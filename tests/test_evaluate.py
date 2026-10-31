"""Tests for scripts/evaluate.py and scripts/backends.py. No model, no network."""

import json
import urllib.error

import pytest

from scenario_io import load_scenarios
from scripts import backends, evaluate

SUMMARY_KEYS = {
    "run_name", "backend", "model", "adapter", "data_path", "data_sha256", "git_commit",
    "timestamp_utc", "composite", "per_difficulty", "per_rubric", "parse_failure_rate",
    "n", "skipped_too_long",
}


def _run(tmp_path, *argv):
    return evaluate.main(["--out-dir", str(tmp_path), *argv])


# --- strategy backend, end to end -------------------------------------------

def test_strategy_end_to_end(tmp_path):
    gt = _run(tmp_path, "--strategy", "ground_truth", "--generate", "9", "--seed", "5", "--run-name", "gt")
    empty = _run(tmp_path, "--strategy", "empty", "--generate", "9", "--seed", "5", "--run-name", "empty")
    assert gt["composite"]["mean"] == pytest.approx(1.0)
    assert empty["composite"]["mean"] == pytest.approx(0.0)
    assert gt["n"] == 9 and empty["parse_failure_rate"] == 0.0
    assert gt["data_sha256"] == empty["data_sha256"]
    assert set(gt["per_difficulty"]) == {"1", "2", "3"}
    assert all(v["n"] == 3 for v in gt["per_difficulty"].values())

    rows = [json.loads(l) for l in (tmp_path / "gt" / "completions.jsonl").read_text().splitlines()]
    assert len(rows) == 9
    assert {"scenario_id", "difficulty", "completion", "score"} <= set(rows[0])
    assert {"composite", "format_ok", "correct_final_state"} <= set(rows[0]["score"])


def test_summary_schema_and_limit(tmp_path):
    s = _run(tmp_path, "--strategy", "follow_all", "--generate", "6", "--seed", "1", "--limit", "4", "--run-name", "r")
    assert SUMMARY_KEYS <= set(s)
    assert set(s["composite"]) >= {"mean", "ci95_low", "ci95_high"}
    assert set(s["per_rubric"]) == set(evaluate.RUBRICS)
    assert s["n"] == 4 and len(s["data_sha256"]) == 64
    on_disk = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert on_disk == s


def test_data_path_inside_repo_is_stored_relative(tmp_path):
    path = evaluate.ROOT / "results" / "data" / "_test_relpath.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((evaluate.generate_dataset(3, 1, tmp_path)).read_bytes())
        s = _run(tmp_path, "--strategy", "empty", "--data", str(path), "--run-name", "rel")
        assert s["data_path"] == "results/data/_test_relpath.jsonl"
    finally:
        path.unlink(missing_ok=True)


def test_generate_is_even_and_reproducible(tmp_path):
    p1 = evaluate.generate_dataset(10, 3, tmp_path)
    p2 = evaluate.generate_dataset(10, 3, tmp_path)
    assert p1 == p2
    scs = load_scenarios(p1)
    counts = sorted(sum(1 for s in scs if s.difficulty == d) for d in (1, 2, 3))
    assert counts == [3, 3, 4]


def test_data_arg_and_exclusive_data(tmp_path):
    path = evaluate.generate_dataset(3, 2, tmp_path)
    s = _run(tmp_path, "--strategy", "ground_truth", "--data", str(path), "--run-name", "d")
    assert s["n"] == 3
    with pytest.raises(SystemExit):
        _run(tmp_path, "--strategy", "empty")
    with pytest.raises(SystemExit):
        _run(tmp_path, "--strategy", "empty", "--data", str(path), "--generate", "3")


def test_backend_choice_must_be_exactly_one(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, "--generate", "3")


# --- bootstrap ----------------------------------------------------------------

def test_bootstrap_deterministic_and_contains_mean():
    vals = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4]
    a = evaluate.bootstrap_ci(vals)
    assert a == evaluate.bootstrap_ci(vals)
    mean = sum(vals) / len(vals)
    assert a[0] <= mean <= a[1]
    assert evaluate.bootstrap_ci([0.5] * 10) == (0.5, 0.5)
    assert evaluate.bootstrap_ci([]) == (0.0, 0.0)


# --- model backend with fakes ---------------------------------------------------

class FakeTensor:
    """Just enough of a tensor: .shape, .to(), and [:, n:] slicing on a 2-D list."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.device_seen = None

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def to(self, device):
        self.device_seen = device
        return self

    def __getitem__(self, idx):
        rows_idx, cols_idx = idx
        assert rows_idx == slice(None)
        return FakeTensor([r[cols_idx] for r in self.rows])

    def __iter__(self):
        return iter(self.rows)


class FakeBatch(dict):
    """BatchEncoding stand-in: dict with attribute access."""

    @property
    def input_ids(self):
        return self["input_ids"]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, batch=False, prompt_len=5):
        self.batch = batch
        self.prompt_len = prompt_len
        self.seen_messages = None
        self.seen_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.seen_messages, self.seen_kwargs = messages, kwargs
        ids = FakeTensor([[7] * self.prompt_len])
        return FakeBatch(input_ids=ids) if self.batch else ids

    def decode(self, row, skip_special_tokens=False):
        assert skip_special_tokens
        return "".join(chr(t) for t in row)


class FakeModel:
    device = "fake-device"

    class config:
        max_position_embeddings = 100

    def __init__(self, answer_for):
        self.answer_for = answer_for
        self.kwargs = None

    def generate(self, **kw):
        self.kwargs = kw
        ids = kw["input_ids"]
        assert ids.device_seen == "fake-device"
        k = kw.get("num_return_sequences", 1)
        prompt = list(ids.rows[0])
        completion = [ord(c) for c in self.answer_for]
        return FakeTensor([prompt + completion for _ in range(k)])


@pytest.fixture
def data_and_scenarios(tmp_path):
    path = evaluate.generate_dataset(3, 9, tmp_path)
    return path, load_scenarios(path)


@pytest.mark.parametrize("batch", [False, True])
def test_model_backend_plumbing(data_and_scenarios, batch):
    scenarios = data_and_scenarios[1]
    tok, model = FakeTokenizer(batch=batch), FakeModel("PLAIN")
    be = backends.ModelBackend("fake", model=model, tokenizer=tok, max_new_tokens=77)
    out = be.complete(scenarios[0])
    assert out == ["PLAIN"]  # prompt tokens are not decoded back into the completion
    assert [m["role"] for m in tok.seen_messages] == ["system", "user"]
    assert tok.seen_messages[1]["content"] == scenarios[0].prompt
    assert tok.seen_kwargs["add_generation_prompt"] is True and tok.seen_kwargs["return_tensors"] == "pt"
    assert model.kwargs["do_sample"] is False and model.kwargs["max_new_tokens"] == 77
    assert "num_return_sequences" not in model.kwargs


def test_model_backend_samples_and_scoring(tmp_path, data_and_scenarios):
    data, scenarios = data_and_scenarios
    gt = backends.StrategyBackend("ground_truth").complete(scenarios[0])[0]
    be = backends.ModelBackend("fake", samples=3, model=FakeModel(gt), tokenizer=FakeTokenizer())
    assert len(be.complete(scenarios[0])) == 3 and be.model.kwargs["do_sample"] is True
    be = backends.ModelBackend("fake", samples=2, model=FakeModel("not json"), tokenizer=FakeTokenizer())
    s = evaluate.run_evaluation(be, scenarios, "m", tmp_path, data)
    assert s["samples"] == 2 and s["n_completions"] == 6 and s["n"] == 3
    assert s["parse_failure_rate"] == 1.0
    assert s["pass_at_k"]["value"] == 0.0 and s["pass_at_k"]["k"] == 2


def test_model_backend_skips_too_long_prompts(tmp_path, data_and_scenarios):
    data, scenarios = data_and_scenarios
    be = backends.ModelBackend("fake", model=FakeModel("x"), tokenizer=FakeTokenizer(prompt_len=50),
                               max_prompt_tokens=10)
    assert be.complete(scenarios[0]) is None
    s = evaluate.run_evaluation(be, scenarios, "long", tmp_path, data)
    assert s["skipped_too_long"] == 3 and s["n"] == 0


def test_model_backend_default_prompt_limit_from_context(data_and_scenarios):
    scenarios = data_and_scenarios[1]
    be = backends.ModelBackend("fake", model=FakeModel("x"), tokenizer=FakeTokenizer(prompt_len=50),
                               max_new_tokens=60)  # context 100 - 60 = 40 < 50
    assert be.max_prompt_tokens == 40 and be.complete(scenarios[0]) is None


# --- API backend ------------------------------------------------------------------

SECRET = "sk-test-SECRET-123456"


def test_api_backend_end_to_end_and_key_never_written(tmp_path, data_and_scenarios):
    data, scenarios = data_and_scenarios
    calls = []
    gt = backends.StrategyBackend("ground_truth")

    def post(url, headers, payload, timeout):
        calls.append((url, headers, payload))
        sc = next(s for s in scenarios if s.prompt == payload["messages"][1]["content"])
        return {"choices": [{"message": {"content": gt.complete(sc)[0]}}]}

    be = backends.ApiBackend("https://user:pw@api.example.com/v1/", "m1", api_key=SECRET, post=post)
    s = evaluate.run_evaluation(be, scenarios, "api", tmp_path, data)
    assert s["composite"]["mean"] == pytest.approx(1.0)
    url, headers, payload = calls[0]
    assert url.endswith("/v1/chat/completions")
    assert headers["Authorization"] == f"Bearer {SECRET}"
    assert payload["temperature"] == 0 and payload["model"] == "m1"
    assert s["api_base"] == "https://api.example.com/v1"  # no credentials stored
    for p in (tmp_path / "api").iterdir():
        text = p.read_text()
        assert SECRET not in text and "pw@" not in text


def test_api_key_from_env_and_missing(monkeypatch):
    monkeypatch.delenv(backends.API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError):
        backends.ApiBackend("http://x", "m")
    monkeypatch.setenv(backends.API_KEY_ENV, SECRET)
    be = backends.ApiBackend("http://x", "m", post=lambda *a: {})
    assert be.describe()["backend"] == "api"
    assert SECRET not in json.dumps(be.describe())


def test_api_retries_with_backoff(data_and_scenarios):
    scenarios = data_and_scenarios[1]
    attempts, sleeps = [], []

    def post(url, headers, payload, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(url, 429, "slow down", {}, None)
        return {"choices": [{"message": {"content": "ok"}}]}

    be = backends.ApiBackend("http://x", "m", api_key=SECRET, post=post, sleep=sleeps.append, backoff=1.0)
    assert be.complete(scenarios[0]) == ["ok"]
    assert sleeps == [1.0, 2.0]


def test_api_gives_up_and_hides_key(data_and_scenarios):
    scenarios = data_and_scenarios[1]

    def post(url, headers, payload, timeout):
        raise urllib.error.URLError(f"boom {headers}")

    be = backends.ApiBackend("http://x", "m", api_key=SECRET, post=post, sleep=lambda s: None, retries=2)
    with pytest.raises(RuntimeError) as e:
        be.complete(scenarios[0])
    assert SECRET not in str(e.value)


def test_api_client_error_not_retried(data_and_scenarios):
    scenarios = data_and_scenarios[1]
    n = []

    def post(url, headers, payload, timeout):
        n.append(1)
        raise urllib.error.HTTPError(url, 401, "no", {}, None)

    be = backends.ApiBackend("http://x", "m", api_key=SECRET, post=post, sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="401"):
        be.complete(scenarios[0])
    assert len(n) == 1


def test_api_malformed_response_scores_as_failure(data_and_scenarios):
    scenarios = data_and_scenarios[1]
    be = backends.ApiBackend("http://x", "m", api_key=SECRET, post=lambda *a: {"oops": 1})
    assert be.complete(scenarios[0]) == [""]
