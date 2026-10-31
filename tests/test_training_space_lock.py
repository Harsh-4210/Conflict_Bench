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


def test_config_table_reads_real_script_values(monkeypatch):
    app, _ = _load_app(monkeypatch, None)
    assert app.CFG["TRAIN_SCENARIOS"] == 300
    assert app.CFG["EVAL_SCENARIOS"] == 10
    assert app.CFG["NUM_GENERATIONS"] == 4
