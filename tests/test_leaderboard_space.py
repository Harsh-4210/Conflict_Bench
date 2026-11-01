"""Tests for leaderboard_space/app.py: scoring, validation, escaping, leaderboard file, no network."""

import importlib.util
import json
import socket
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from scenario_io import dumps_scenario, load_scenarios
from scripts.baselines import ground_truth

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "leaderboard_space" / "app.py"
DATA = ROOT / "data" / "test_v1.jsonl"


# ---------------------------------------------------------------------------
# Nothing in this module may touch the network or the Hub.
# ---------------------------------------------------------------------------

class _NoHub(types.ModuleType):
    def __getattr__(self, name):
        raise AssertionError(f"huggingface_hub.{name} used without a network-free test double")


@pytest.fixture(scope="module", autouse=True)
def no_network():
    mp = pytest.MonkeyPatch()

    def boom(*a, **k):
        raise AssertionError("network access attempted")

    real_connect, real_create, real_getaddrinfo = socket.socket.connect, socket.create_connection, socket.getaddrinfo

    def is_local(host):
        return host in ("127.0.0.1", "::1", "localhost", "")

    # asyncio on Windows opens a loopback socket pair; only non-loopback traffic counts as network.
    def guarded_connect(self, address, *a, **k):
        if not (isinstance(address, tuple) and is_local(address[0])):
            boom()
        return real_connect(self, address, *a, **k)

    def guarded_create(address, *a, **k):
        if not is_local(address[0]):
            boom()
        return real_create(address, *a, **k)

    def guarded_getaddrinfo(host, *a, **k):
        if not is_local(host):
            boom()
        return real_getaddrinfo(host, *a, **k)

    mp.setattr(urllib.request, "urlopen", boom)
    mp.setattr(socket.socket, "connect", guarded_connect)
    mp.setattr(socket, "create_connection", guarded_create)
    mp.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    mp.setitem(sys.modules, "huggingface_hub", _NoHub("huggingface_hub"))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def app(no_network):
    spec = importlib.util.spec_from_file_location("leaderboard_space_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["leaderboard_space_app"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_set(tmp_path_factory):
    """30 scenarios (10 per difficulty) in the private full format."""
    scenarios = load_scenarios(DATA)[::20]
    path = tmp_path_factory.mktemp("private") / "test_full.jsonl"
    path.write_text("".join(dumps_scenario(s) + "\n" for s in scenarios), encoding="utf-8")
    return path, scenarios


@pytest.fixture
def env(app, small_set, monkeypatch, tmp_path):
    path, _ = small_set
    monkeypatch.setenv("CONFLICTBENCH_PRIVATE_SET", str(path))
    monkeypatch.delenv("CONFLICTBENCH_PRIVATE_REPO", raising=False)
    monkeypatch.setenv("CONFLICTBENCH_LEADERBOARD", str(tmp_path / "lb.json"))
    app.reset_cache()
    return tmp_path / "lb.json"


def jsonl(rows):
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


def perfect(scenarios):
    return jsonl([{"scenario_id": s.scenario_id, "completion": ground_truth(s, None)} for s in scenarios])


def empty(scenarios):
    return jsonl([{"scenario_id": s.scenario_id, "completion": ""} for s in scenarios])


def upload(tmp_path, raw, name="p.jsonl"):
    p = tmp_path / name
    p.write_bytes(raw)
    return str(p)


# ---------------------------------------------------------------------------
# Import and code location
# ---------------------------------------------------------------------------

def test_importable_without_network_and_without_hub(app):
    assert Path(app.REPO_ROOT) == ROOT
    assert callable(app.process_submission)
    assert isinstance(sys.modules["huggingface_hub"], _NoHub)  # the app did not need it


def test_the_network_guard_is_active():
    """Sanity check of this module's own guard, so the 'no network' claims are not vacuous."""
    with pytest.raises(AssertionError, match="network"):
        urllib.request.urlopen("https://huggingface.co")
    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("huggingface.co", 443))
    with pytest.raises(AssertionError, match="used without"):
        import huggingface_hub
        huggingface_hub.hf_hub_download


def test_validate_ref_refuses_branch_names(app):
    for bad in ("", "main", "master", "HEAD", "REPLACE_WITH_TAG_OR_COMMIT", "-x", "a b", "a;rm"):
        with pytest.raises(RuntimeError):
            app.validate_ref(bad)
    assert app.validate_ref("v1.0.0") == "v1.0.0"
    assert app.validate_ref("04a6658") == "04a6658"


def test_repo_is_not_fetched_when_present(app, monkeypatch):
    monkeypatch.setenv("CONFLICTBENCH_REF", "v1")
    monkeypatch.setattr(app, "fetch_repo_at_ref", lambda *a, **k: pytest.fail("fetched"))
    assert app.find_repo_root() == ROOT


def test_repo_is_fetched_only_at_a_pinned_ref(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "HERE", tmp_path / "space")
    monkeypatch.setenv("CONFLICTBENCH_REPO_DIR", str(tmp_path / "missing"))
    monkeypatch.delenv("CONFLICTBENCH_REF", raising=False)
    with pytest.raises(RuntimeError, match="not found"):
        app.find_repo_root()
    monkeypatch.setenv("CONFLICTBENCH_REF", "main")
    with pytest.raises(RuntimeError, match="tag or commit"):
        app.find_repo_root()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_perfect_submission_scores_one(app, env, small_set):
    _, scenarios = small_set
    result = app.evaluate_submission(perfect(scenarios), "perfect", app.load_private_set())
    assert result.ok, result.errors
    c = result.summary["composite"]
    assert (c["mean"], c["ci95_low"], c["ci95_high"]) == (1.0, 1.0, 1.0)
    assert result.summary["n"] == 30
    assert {d: v["mean"] for d, v in result.summary["per_difficulty"].items()} == {"1": 1.0, "2": 1.0, "3": 1.0}
    assert all(v == 1.0 for k, v in result.summary["per_rubric"].items()
               if k in ("correct_final_state", "no_contradictions", "conflict_identification"))
    assert result.entry["composite"] == 1.0 and result.entry["parse_failure_rate"] == 0.0
    assert result.entry["set_sha256"] == app.load_private_set().sha256


def test_empty_submission_scores_zero(app, env, small_set):
    _, scenarios = small_set
    result = app.evaluate_submission(empty(scenarios), "empty", app.load_private_set())
    assert result.ok
    assert result.summary["composite"]["mean"] == 0.0
    assert result.summary["parse_failure_rate"] == 1.0
    empty_lists = jsonl([{"scenario_id": s.scenario_id,
                          "completion": '{"identified_conflicts":[],"execution_plan":[],"overridden_instructions":[]}'}
                         for s in scenarios])
    assert app.evaluate_submission(empty_lists, "lists", app.load_private_set()).summary["composite"]["mean"] == 0.0


def test_partial_credit_is_between(app, env, small_set):
    _, scenarios = small_set
    rows = [{"scenario_id": s.scenario_id, "completion": ground_truth(s, None) if i % 2 == 0 else ""}
            for i, s in enumerate(scenarios)]
    result = app.evaluate_submission(jsonl(rows), "half", app.load_private_set())
    assert result.ok and 0.4 < result.summary["composite"]["mean"] < 0.6
    assert result.summary["composite"]["ci95_low"] < result.summary["composite"]["ci95_high"]


def test_bom_crlf_and_blank_lines_are_accepted(app, env, small_set):
    _, scenarios = small_set
    raw = b"\xef\xbb\xbf" + perfect(scenarios).replace(b"\n", b"\r\n\r\n")
    assert app.evaluate_submission(raw, "m", app.load_private_set()).ok


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _errors(app, raw, name="m"):
    result = app.evaluate_submission(raw, name, app.load_private_set())
    assert not result.ok and result.summary is None and result.entry is None
    return " | ".join(result.errors)


def test_missing_ids_rejected(app, env, small_set):
    _, scenarios = small_set
    raw = perfect(scenarios[:-3])
    assert "3 of 30 scenarios have no prediction" in _errors(app, raw)


def test_duplicate_ids_rejected(app, env, small_set):
    _, scenarios = small_set
    rows = [{"scenario_id": s.scenario_id, "completion": "x"} for s in scenarios]
    msg = _errors(app, jsonl(rows + [rows[0]]))
    assert "already appeared on line 1" in msg and "Line 31" in msg


def test_unknown_ids_rejected(app, env, small_set):
    _, scenarios = small_set
    rows = [{"scenario_id": s.scenario_id, "completion": "x"} for s in scenarios]
    msg = _errors(app, jsonl(rows + [{"scenario_id": "not-a-scenario", "completion": "x"}]))
    assert "unknown scenario_id 'not-a-scenario'" in msg


def test_malformed_lines_rejected(app, env, small_set):
    raw = b'{"scenario_id": "a", "completion": "x"}\nthis is not json\n[1, 2]\n{"scenario_id": "a"}\n' \
          b'{"completion": "x"}\n{"scenario_id": 5, "completion": "x"}\n{"scenario_id": "a", "completion": 7}\n'
    msg = _errors(app, raw)
    for needle in ("Line 1: unknown", "Line 2: not valid JSON", "Line 3: expected a JSON object",
                   "Line 4: completion is missing", "Line 5: scenario_id is missing",
                   "Line 6: scenario_id is missing", "Line 7: completion is missing"):
        assert needle in msg, needle


def test_file_level_errors(app, env, monkeypatch):
    assert "empty" in _errors(app, b"  \n\n")
    assert "UTF-8" in _errors(app, b"\xff\xfe\x00bad")
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", 100)
    assert "too large" in _errors(app, b"x" * 101)


def test_oversize_completion_rejected(app, env, small_set):
    _, scenarios = small_set
    rows = [{"scenario_id": s.scenario_id, "completion": "x"} for s in scenarios]
    rows[0]["completion"] = "x" * (app.MAX_COMPLETION_CHARS + 1)
    assert "Line 1: completion is longer" in _errors(app, jsonl(rows))


def test_error_flood_is_capped(app, env):
    msg_list = app.evaluate_submission(b"junk\n" * 1000, "m", app.load_private_set()).errors
    assert len(msg_list) <= app.MAX_ERRORS_COLLECTED + 2
    html = app.render_errors(msg_list)
    assert html.count("<li>") == app.MAX_ERRORS_SHOWN + 1 and "more." in html


@pytest.mark.parametrize("name,fragment", [("", "Enter a model name"), ("   ", "Enter a model name"),
                                           (None, "Enter a model name"), ("x" * 81, "too long"),
                                           ("bad\x00name", "control characters"), ("a\nb", "control characters")])
def test_bad_model_names(app, env, small_set, name, fragment):
    _, scenarios = small_set
    assert fragment in _errors(app, perfect(scenarios), name=name)


def test_rejected_submission_leaves_leaderboard_untouched(app, env, tmp_path):
    status, result, board = app.process_submission(upload(tmp_path, b"junk\n"), "m")
    assert "Submission rejected" in status and result == ""
    assert not env.exists()
    assert "No submissions yet" in board


def test_no_file_and_missing_file_do_not_crash(app, env, tmp_path):
    assert "Choose a predictions file" in app.process_submission(None, "m")[0]
    assert "could not be read" in app.process_submission(str(tmp_path / "nope.jsonl"), "m")[0]


def test_oversize_file_is_not_read(app, env, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", 10)
    status, _, _ = app.process_submission(upload(tmp_path, b"x" * 50), "m")
    assert "too large" in status


# ---------------------------------------------------------------------------
# HTML escaping
# ---------------------------------------------------------------------------

EVIL = '<script>alert("x")</script><img src=x onerror=alert(1)> & "q"'


def test_model_name_is_escaped_everywhere(app, env, small_set, tmp_path):
    _, scenarios = small_set
    status, result, board = app.process_submission(upload(tmp_path, perfect(scenarios)), EVIL)
    assert "Scored and added" in status
    for html in (status, result, board):
        assert "<script>" not in html and "<img" not in html and "onerror=alert(1)>" not in html
    assert "&lt;script&gt;" in result and "&lt;script&gt;" in board
    assert "<script>" not in app.render_leaderboard(json.loads(env.read_text()))


def test_error_text_from_user_input_is_escaped(app, env, tmp_path):
    raw = jsonl([{"scenario_id": EVIL, "completion": "x"}])
    status, _, _ = app.process_submission(upload(tmp_path, raw), "m")
    assert "<script>" not in status and "<img" not in status
    assert "&lt;" in status


def test_stored_entries_with_html_are_escaped_on_render(app):
    html = app.render_leaderboard([{"model": EVIL, "composite": 0.5, "submitted_utc": EVIL}])
    assert "<script>" not in html and "<img" not in html


# ---------------------------------------------------------------------------
# Leaderboard file
# ---------------------------------------------------------------------------

def _entry(model, composite, ts="2026-01-01T00:00:00+00:00", sha="s"):
    return {"model": model, "composite": composite, "ci95_low": composite, "ci95_high": composite,
            "per_difficulty": {"1": composite, "2": composite, "3": composite}, "parse_failure_rate": 0.0,
            "n": 30, "submitted_utc": ts, "submission_sha256": sha, "set_sha256": "set"}


def test_append_and_sort(app, env):
    app.append_entry(_entry("low", 0.2, sha="1"))
    app.append_entry(_entry("high", 0.9, sha="2"))
    app.append_entry(_entry("tie-late", 0.5, ts="2026-03-01T00:00:00+00:00", sha="3"))
    table = app.append_entry(_entry("tie-early", 0.5, ts="2026-02-01T00:00:00+00:00", sha="4"))
    assert [e["model"] for e in table] == ["high", "tie-early", "tie-late", "low"]
    assert [e["model"] for e in app.read_leaderboard()] == ["low", "high", "tie-late", "tie-early"]  # stored in order
    html = app.render_leaderboard(app.read_leaderboard())
    assert html.index("high") < html.index("tie-early") < html.index("tie-late") < html.index("low")
    assert not env.with_name("lb.json.tmp").exists()


def test_same_file_for_same_model_is_refused(app, env):
    app.append_entry(_entry("m", 0.5, sha="same"))
    with pytest.raises(app.LeaderboardError, match="already on the leaderboard"):
        app.append_entry(_entry("m", 0.5, sha="same"))
    app.append_entry(_entry("other", 0.5, sha="same"))  # another name is a different row
    assert len(app.read_leaderboard()) == 2


def test_submission_end_to_end_appends_a_row(app, env, small_set, tmp_path):
    _, scenarios = small_set
    app.process_submission(upload(tmp_path, perfect(scenarios), "a.jsonl"), "model-a")
    status, result, board = app.process_submission(upload(tmp_path, empty(scenarios), "b.jsonl"), "model-b")
    assert "Scored and added" in status and "model-b" in result
    rows = app.read_leaderboard()
    assert [e["model"] for e in rows] == ["model-a", "model-b"]
    assert board.index("model-a") < board.index("model-b")  # sorted by score
    again, _, _ = app.process_submission(upload(tmp_path, empty(scenarios), "b.jsonl"), "model-b")
    assert "already on the leaderboard" in again and len(app.read_leaderboard()) == 2


def test_corrupt_leaderboard_is_a_message_not_a_crash(app, env, small_set, tmp_path):
    _, scenarios = small_set
    env.write_text("{not json")
    status, result, board = app.process_submission(upload(tmp_path, perfect(scenarios)), "m")
    assert "unreadable" in status and "m" in result  # the score is still shown
    assert "unreadable" in board
    assert env.read_text() == "{not json"  # never overwritten


def test_unexpected_errors_are_swallowed(app, env, small_set, tmp_path, monkeypatch):
    _, scenarios = small_set
    monkeypatch.setattr(app, "score", lambda *a, **k: 1 / 0)
    status, _, _ = app.process_submission(upload(tmp_path, perfect(scenarios)), "m")
    assert "Something went wrong" in status and "ZeroDivision" not in status


# ---------------------------------------------------------------------------
# Private set loading
# ---------------------------------------------------------------------------

def test_private_set_not_configured(app, monkeypatch):
    monkeypatch.delenv("CONFLICTBENCH_PRIVATE_SET", raising=False)
    monkeypatch.delenv("CONFLICTBENCH_PRIVATE_REPO", raising=False)
    app.reset_cache()
    with pytest.raises(app.PrivateSetError, match="not configured"):
        app.load_private_set()
    status, _, _ = app.process_submission(__file__, "m")
    assert "not configured" in status


def test_private_set_from_folder_and_bad_paths(app, small_set, monkeypatch, tmp_path):
    path, scenarios = small_set
    monkeypatch.delenv("CONFLICTBENCH_PRIVATE_REPO", raising=False)
    monkeypatch.setenv("CONFLICTBENCH_PRIVATE_SET", str(path.parent))
    app.reset_cache()
    assert len(app.load_private_set().scenarios) == len(scenarios)
    monkeypatch.setenv("CONFLICTBENCH_PRIVATE_SET", str(tmp_path / "gone.jsonl"))
    app.reset_cache()
    with pytest.raises(app.PrivateSetError, match="not found"):
        app.load_private_set()
    junk = tmp_path / "junk.jsonl"
    junk.write_text("not json\n")
    monkeypatch.setenv("CONFLICTBENCH_PRIVATE_SET", str(junk))
    app.reset_cache()
    with pytest.raises(app.PrivateSetError, match="could not be read"):
        app.load_private_set()


def test_private_set_from_hub_uses_lazy_import(app, small_set, monkeypatch):
    path, scenarios = small_set
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(path)

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.delenv("CONFLICTBENCH_PRIVATE_SET", raising=False)
    monkeypatch.setenv("CONFLICTBENCH_PRIVATE_REPO", "someone/private-set")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    app.reset_cache()
    loaded = app.load_private_set()
    assert len(loaded.scenarios) == len(scenarios)
    assert calls == [{"repo_id": "someone/private-set", "filename": "test_full.jsonl",
                      "repo_type": "dataset", "token": "hf_secret_value"}]
    app.load_private_set()
    assert len(calls) == 1  # cached


def test_hub_branch_errors_do_not_leak_the_token(app, monkeypatch):
    def failing(**kwargs):
        raise OSError(f"401 for token {kwargs['token']}")

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = failing
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.delenv("CONFLICTBENCH_PRIVATE_SET", raising=False)
    monkeypatch.setenv("CONFLICTBENCH_PRIVATE_REPO", "someone/private-set")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    app.reset_cache()
    with pytest.raises(app.PrivateSetError) as info:
        app.load_private_set()
    assert "hf_secret_value" not in str(info.value)
    monkeypatch.delenv("HF_TOKEN")
    app.reset_cache()
    with pytest.raises(app.PrivateSetError, match="HF_TOKEN"):
        app.load_private_set()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def test_ui_builds_without_launching_or_network(app):
    gr = pytest.importorskip("gradio")
    demo = app.build_ui()
    assert isinstance(demo, gr.Blocks)
    assert demo.analytics_enabled is False
