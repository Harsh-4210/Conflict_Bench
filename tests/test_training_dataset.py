"""Training dataset: records, token-length filtering, split usage. Uses a fake whitespace tokenizer (no downloads)."""

import logging

import pytest

import training.train_grpo as tg
from generator import ScenarioGenerator
from prompts import build_messages
from scenario_io import loads_scenario
from training.config import config_from_dict
from training.train_grpo import DatasetError, build_datasets, build_split, make_generator, prompt_token_count


class FakeTokenizer:
    """Whitespace 'tokens'. Mimics the two methods the pipeline uses."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        text = "".join(f"<{m['role']}> {m['content']} </{m['role']}>\n" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}


def small_cfg(**over):
    base = dict(train_size=12, val_size=6, train_difficulty_mix={1: 0.5, 2: 0.5}, val_difficulty_mix={1: 0.5, 2: 0.5},
                max_prompt_tokens=100_000, max_completion_tokens=256, max_seq_length=200_000, seed=11)
    base.update(over)
    return config_from_dict(base)


class StubGenerator:
    """Real scenarios, but with a controlled prompt length (in words) per difficulty."""

    def __init__(self, seed, words_by_difficulty):
        self.inner = ScenarioGenerator(seed=seed)
        self.words = words_by_difficulty

    def generate(self, difficulty):
        sc = self.inner.generate(difficulty=difficulty)
        sc.prompt = " ".join(["word"] * self.words[difficulty]) + f" id{self.inner.rng.random()}"
        return sc


def test_prompt_token_count_uses_the_full_chat_templated_prompt():
    sc = ScenarioGenerator(seed=1).generate(difficulty=1)
    msgs = build_messages(sc)
    tok = FakeTokenizer()
    n = prompt_token_count(tok, msgs)
    assert n > len(sc.prompt.split())                 # system prompt + template tokens are counted too
    assert n == len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))["input_ids"])


def test_records_have_prompt_scenario_json_and_difficulty():
    cfg = small_cfg()
    built = build_datasets(cfg, FakeTokenizer())
    assert len(built.train) == 12 and len(built.val) == 6
    for rec in built.train + built.val:
        assert set(rec) == {"prompt", "scenario_json", "difficulty"}
        sc = loads_scenario(rec["scenario_json"])
        assert rec["prompt"] == build_messages(sc)    # exactly what the eval harness will show the model
        assert rec["prompt"][0]["role"] == "system" and rec["prompt"][1]["role"] == "user"
        assert rec["difficulty"] == sc.difficulty


def test_difficulty_mix_is_respected():
    cfg = small_cfg(train_size=20, train_difficulty_mix={1: 0.75, 2: 0.25})
    built = build_datasets(cfg, FakeTokenizer())
    counts = {}
    for r in built.train:
        counts[r["difficulty"]] = counts.get(r["difficulty"], 0) + 1
    assert counts == {1: 15, 2: 5}
    assert built.stats["train"][1].kept == 15 and built.stats["train"][2].target == 5


def test_build_is_deterministic():
    a = build_datasets(small_cfg(), FakeTokenizer())
    b = build_datasets(small_cfg(), FakeTokenizer())
    assert [r["scenario_json"] for r in a.train] == [r["scenario_json"] for r in b.train]
    assert [r["scenario_json"] for r in a.val] == [r["scenario_json"] for r in b.val]


def test_only_train_and_val_splits_are_used(monkeypatch):
    seen = []
    real = tg.make_generator
    monkeypatch.setattr(tg, "make_generator", lambda cfg, split: seen.append(split) or real(cfg, split))
    build_datasets(small_cfg(), FakeTokenizer())
    assert seen == ["train", "val"]


def test_make_generator_refuses_the_test_split():
    with pytest.raises(ValueError, match="test split"):
        make_generator(small_cfg(), "test")
    with pytest.raises(ValueError):
        make_generator(small_cfg(), "anything-else")


def test_train_and_val_scenarios_do_not_overlap():
    built = build_datasets(small_cfg(train_size=30, val_size=30), FakeTokenizer())
    train_prompts = {r["prompt"][1]["content"] for r in built.train}
    assert not train_prompts & {r["prompt"][1]["content"] for r in built.val}


def test_val_prompts_identical_to_train_are_dropped(monkeypatch):
    # until the real train/val template pools exist, a generator that ignores the split must not leak train into val
    monkeypatch.setattr(tg, "make_generator", lambda cfg, split: StubGenerator(5, {1: 20, 2: 20}))
    cfg = small_cfg(train_size=4, val_size=2, train_difficulty_mix={1: 1.0}, val_difficulty_mix={1: 1.0})
    # same seed -> identical prompt sequence in both splits -> every val candidate collides until the stub moves on
    built = build_datasets(cfg, FakeTokenizer())
    assert built.stats["val"][1].dropped_duplicate >= 1
    train_prompts = {r["prompt"][1]["content"] for r in built.train}
    assert not train_prompts & {r["prompt"][1]["content"] for r in built.val}


def test_length_filter_counts_tokens_and_logs_drops(monkeypatch, caplog):
    # difficulty 2 prompts are 150 words, difficulty 1 prompts 20 words; the system prompt adds ~170 tokens
    monkeypatch.setattr(tg, "make_generator", lambda cfg, split: StubGenerator(cfg.seed, {1: 20, 2: 150}))
    tok = FakeTokenizer()
    sc = StubGenerator(1, {1: 20, 2: 150}).generate(1)
    d1_tokens = prompt_token_count(tok, build_messages(sc))
    cfg = small_cfg(train_size=10, train_difficulty_mix={1: 1.0}, max_prompt_tokens=d1_tokens + 5)
    with caplog.at_level(logging.INFO, logger="conflictbench.train"):
        out = build_split(cfg, "train", 10, {1: 1.0}, tok)
    assert out.stats[1].kept == 10 and out.stats[1].dropped_too_long == 0
    assert "dropped_too_long=0" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="conflictbench.train"):
        with pytest.raises(DatasetError) as exc:
            build_split(cfg, "train", 10, {2: 1.0}, tok)
    assert "difficulty 2" in str(exc.value) and "max_prompt_tokens" in str(exc.value)
    assert "dropped_too_long=30" in caplog.text           # budget = 3 x target candidates, all dropped


def test_partial_drops_are_tolerated_down_to_half(monkeypatch):
    class Alternating(StubGenerator):
        n = 0

        def generate(self, difficulty):
            sc = super().generate(difficulty)
            Alternating.n += 1
            if Alternating.n % 2 == 0:                      # every second candidate is far too long
                sc.prompt = "word " * 5000 + f"id{Alternating.n}"
            return sc

    monkeypatch.setattr(tg, "make_generator", lambda cfg, split: Alternating(1, {1: 20}))
    cfg = small_cfg(max_prompt_tokens=2000)
    out = build_split(cfg, "train", 10, {1: 1.0}, FakeTokenizer())
    assert out.stats[1].kept == 10                          # replaced by further candidates
    assert out.stats[1].dropped_too_long >= 9


def test_fails_when_less_than_half_of_a_difficulty_survives(monkeypatch):
    class Mostly(StubGenerator):
        n = 0

        def generate(self, difficulty):
            sc = super().generate(difficulty)
            Mostly.n += 1
            if Mostly.n % 8 != 0:                           # only 1 in 8 fits; budget 30 candidates -> keeps 3 of 10
                sc.prompt = "word " * 5000 + f"id{Mostly.n}"
            return sc

    monkeypatch.setattr(tg, "make_generator", lambda cfg, split: Mostly(1, {1: 20}))
    cfg = small_cfg(max_prompt_tokens=2000)
    with pytest.raises(DatasetError, match="kept 3/10"):
        build_split(cfg, "train", 10, {1: 1.0}, FakeTokenizer())


def test_error_names_the_difficulty_that_failed_not_the_others(monkeypatch):
    monkeypatch.setattr(tg, "make_generator", lambda cfg, split: StubGenerator(cfg.seed, {1: 20, 2: 20, 3: 5000}))
    cfg = small_cfg(train_size=9, train_difficulty_mix={1: 1 / 3, 2: 1 / 3, 3: 1 / 3}, max_prompt_tokens=1000)
    with pytest.raises(DatasetError) as exc:
        build_split(cfg, "train", 9, cfg.train_difficulty_mix, FakeTokenizer())
    msg = str(exc.value)
    assert "difficulty 3" in msg and "difficulty 1:" not in msg and "difficulty 2:" not in msg
