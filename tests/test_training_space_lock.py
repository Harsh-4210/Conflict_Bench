"""The public training Space must never start a paid GPU run without the password."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

APP_PATH = Path(__file__).resolve().parent.parent / "training_space" / "app.py"


def _load_app(monkeypatch, password):
    if password is None:
        monkeypatch.delenv("TRAIN_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("TRAIN_PASSWORD", password)
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, *a, **k: None)

    # If the lock ever fails, this fake train_script records the call instead of loading unsloth.
    started = []
    fake = types.ModuleType("train_script")
    fake.run_training = lambda progress_callback=None: started.append(True) or []
    monkeypatch.setitem(sys.modules, "train_script", fake)

    spec = importlib.util.spec_from_file_location("training_space_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, started


def test_disabled_without_env_password(monkeypatch):
    app, started = _load_app(monkeypatch, None)
    first = next(app.start_training("anything"))
    assert "disabled" in first[0]
    assert started == []


def test_wrong_password_refused(monkeypatch):
    app, started = _load_app(monkeypatch, "s3cret")
    first = next(app.start_training("wrong"))
    assert "Wrong password" in first[0]
    assert started == []


def test_correct_password_starts(monkeypatch):
    app, started = _load_app(monkeypatch, "s3cret")
    list(app.start_training("s3cret"))  # drain the generator until the thread finishes
    assert started == [True]
    assert not app.TRAINING_LOCK.locked()


def test_config_table_matches_the_yaml_the_space_runs(monkeypatch):
    """The UI table (read from train_script.py constants) must equal configs/grpo_a10g.yaml, the run it launches."""
    from training.config import load_config

    app, _ = _load_app(monkeypatch, None)
    train_script = _load_train_script_module()
    cfg = load_config(Path(__file__).resolve().parent.parent / train_script.CONFIG_FILE)
    assert train_script.check_constants_match(cfg) == []
    for name, field in train_script.CONSTANT_TO_FIELD.items():
        assert app.CFG[name] == getattr(cfg, field), name


def test_train_script_detects_a_stale_constant():
    from training.config import config_from_dict

    train_script = _load_train_script_module()
    cfg = config_from_dict({"train_size": 123})
    assert "TRAIN_SCENARIOS" in train_script.check_constants_match(cfg, {
        **{n: getattr(cfg, f) for n, f in train_script.CONSTANT_TO_FIELD.items()}, "TRAIN_SCENARIOS": 600})


def _load_train_script_module():
    """Import the real train_script.py (it has no heavy imports at module level)."""
    path = APP_PATH.parent / "train_script.py"
    spec = importlib.util.spec_from_file_location("real_train_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_script_has_no_training_code_of_its_own():
    """F-34: the Space must call the shared pipeline, not carry a copy of it."""
    text = (APP_PATH.parent / "train_script.py").read_text(encoding="utf-8")
    assert "from training.train_grpo import" in text
    for forbidden in ("GRPOTrainer(", "GRPOConfig(", "FastLanguageModel", "def build_reward_fn", "def build_dataset"):
        assert forbidden not in text


def test_dockerfile_is_pinned():
    text = (APP_PATH.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "git+https://github.com/unslothai" not in text          # no unsloth from git HEAD
    assert "CONFLICTBENCH_REF" in text and "checkout --detach" in text
    assert "requirements-train.txt" in text
