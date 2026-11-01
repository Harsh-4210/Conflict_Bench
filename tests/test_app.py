"""Tests for the Gradio demo that need neither torch nor a GPU."""

import contextlib
import json
import sys

import pytest

from generator import ScenarioGenerator


def test_import_app_does_not_load_torch_or_models():
    already_loaded = "torch" in sys.modules
    import app
    assert app._state["loaded"] is False
    assert app._state["model"] is None
    if not already_loaded:
        assert "torch" not in sys.modules
    assert app.demo is not None


def _entry(**kw):
    base = {"instruction_a": "INS-1", "instruction_b": "INS-2", "conflict_type": "direct",
            "resolution": "INS-1", "reasoning": "ok"}
    base.update(kw)
    return base


def _out(conflicts, plan=("INS-1",), over=("INS-2",)):
    return json.dumps({"identified_conflicts": conflicts,
                       "execution_plan": list(plan),
                       "overridden_instructions": list(over)})


def test_script_in_reasoning_is_escaped():
    import app
    html_out = app.format_json_html(_out([_entry(reasoning="<script>alert(1)</script>")]))
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_every_model_string_is_escaped():
    import app
    evil = "<img src=x onerror=alert(1)>"
    html_out = app.format_json_html(_out(
        [_entry(instruction_a=evil, instruction_b=evil, conflict_type=evil, resolution=evil)],
        plan=[evil], over=[evil]))
    assert "<img" not in html_out
    assert "&lt;img" in html_out


def test_parse_failure_escapes_raw_text():
    import app
    html_out = app.format_json_html("<script>alert(1)</script> not json")
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


@pytest.mark.parametrize("junk", [
    None, 42, "just a string", ["a", "list"], {"reasoning": ["not", "a", "str"]},
    {"instruction_a": None, "reasoning": 123, "resolution": {"x": 1}},
])
def test_format_json_html_survives_junk_conflict_entries(junk):
    import app
    out = app.format_json_html(_out([junk, _entry()]))
    assert "jv-root" in out


def test_non_list_fields_do_not_crash():
    import app
    raw = json.dumps({"identified_conflicts": "oops", "execution_plan": ["INS-1"],
                      "overridden_instructions": ["INS-2"]})
    assert "jv-root" in app.format_json_html(raw)


def test_unparseable_shapes_do_not_crash():
    import app
    for raw in ("", "[]", "[1,2,3]", "null", "{", None, 7):
        assert isinstance(app.format_json_html(raw), str)


def test_format_score_html_handles_missing_and_bad_values():
    import app
    assert "score-wrap" in app.format_score_html({})
    assert "score-wrap" in app.format_score_html({"composite": None, "efficiency": "x"})
    assert "score-wrap" in app.format_score_html({"composite": 7.0})  # clamped


def test_ground_truth_tab_renders_and_lists_ground_truth_ids():
    import app
    sc = ScenarioGenerator(seed=3).generate(difficulty=2)
    out = app.format_ground_truth_html(sc)
    assert "Parse Failed" not in out
    assert "jv-root" in out
    for ins_id in sc.ground_truth_followed:
        assert ins_id in out
    assert "Generate a scenario first" in app.format_ground_truth_html(None)


def test_generate_scenario_handler_returns_nine_outputs():
    import app
    res = app.generate_scenario(2)
    assert len(res) == 9
    assert res[1] == res[0].prompt


def test_false_claims_removed_from_page_text():
    import app
    src = open(app.__file__, encoding="utf-8").read().lower()
    assert "never stated" not in src
    assert "non-gameable" not in src
    assert "not yet re-measured" in src
    for stale in ("0.5003", "0.239", "checkpoint-200", "87%"):
        assert stale not in src


# --- adapter on/off logic, with fakes (no torch needed) -------------------

class _FakeModel:
    def __init__(self):
        self.adapter_enabled = True

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True


@pytest.fixture
def fake_loaded(monkeypatch):
    import app
    model = _FakeModel()
    monkeypatch.setitem(app._state, "loaded", True)
    monkeypatch.setitem(app._state, "model", model)
    monkeypatch.setitem(app._state, "tokenizer", object())
    monkeypatch.setitem(app._state, "adapter_loaded", True)
    monkeypatch.setitem(app._state, "adapter_error", None)
    seen = []

    def fake_generate(m, tok, scenario, sample=False, max_new_tokens=0):
        seen.append(m.adapter_enabled)
        return "{}"

    monkeypatch.setattr(app, "generate_text", fake_generate)
    return model, seen


def test_base_runs_without_adapter_trained_runs_with_it(fake_loaded):
    import app
    _, seen = fake_loaded
    sc = ScenarioGenerator(seed=1).generate(difficulty=1)
    app.run_model("base", sc)
    app.run_model("trained", sc)
    assert seen == [False, True]


def test_trained_without_adapter_is_an_error_not_a_silent_base_run(fake_loaded, monkeypatch):
    import app
    monkeypatch.setitem(app._state, "adapter_loaded", False)
    monkeypatch.setitem(app._state, "adapter_error", "boom")
    sc = ScenarioGenerator(seed=1).generate(difficulty=1)
    with pytest.raises(RuntimeError):
        app.run_model("trained", sc)
    out_html, score_html = app.run_trained_model(sc)
    assert "Inference failed" in out_html


def test_prompt_too_long_shows_error_instead_of_truncating(monkeypatch):
    import app

    def too_long(*a, **k):
        raise app.PromptTooLongError("5000 tokens")

    monkeypatch.setattr(app, "run_model", too_long)
    sc = ScenarioGenerator(seed=1).generate(difficulty=1)
    out_html, _ = app.run_base_model(sc)
    assert "Prompt too long" in out_html


def test_to_input_ids_accepts_tensor_or_batch_encoding():
    import app

    class Tensorish:  # no .keys()
        pass

    t = Tensorish()
    assert app._to_input_ids(t) is t
    assert app._to_input_ids({"input_ids": 5, "attention_mask": 1}) == 5
