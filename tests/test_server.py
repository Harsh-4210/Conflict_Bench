"""
Tests for the OpenEnv HTTP/WebSocket server (P1-4).

openenv-core builds a *fresh* environment per plain-HTTP request, so a real
reset-then-step episode over HTTP needs a factory that hands back one shared
instance. The tests below use that for the HTTP routes and the real, default
factory (``server.app.app``) for the WebSocket session.
"""

import json

import pytest
from fastapi.testclient import TestClient

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import State

from server.app import app as default_app
from server.conflictbench_environment import ConflictBenchEnvironment
from server.models import ConflictBenchAction, ConflictBenchObservation


@pytest.fixture
def shared_env():
    """One environment instance served by every HTTP request (stateful HTTP)."""
    return ConflictBenchEnvironment(auto_scale_difficulty=False, seed=3)


@pytest.fixture
def http(shared_env):
    shared_app = create_app(
        lambda: shared_env, ConflictBenchAction, ConflictBenchObservation,
        env_name="conflictbench-test",
    )
    with TestClient(shared_app) as client:
        yield client


def _gt_action(env: ConflictBenchEnvironment) -> str:
    return env._env.sample_action()


def test_app_imports_and_exposes_routes():
    paths = {r.path for r in default_app.routes}
    assert {"/reset", "/step", "/state", "/health", "/ws"} <= paths


def test_reset_returns_prompt_text(http):
    r = http.post("/reset", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is False
    obs = body["observation"]
    assert len(obs["text"]) > 200
    assert obs["num_conflicts"] > 0
    assert obs["domain"] and obs["scenario_id"]
    assert obs["error"] is None


def test_step_with_ground_truth_scores_one(http, shared_env):
    http.post("/reset", json={})
    r = http.post("/step", json={"action": {"text": _gt_action(shared_env)}})
    assert r.status_code == 200
    body = r.json()
    assert body["reward"] == 1.0
    assert body["done"] is True
    assert body["observation"]["score_breakdown"]["composite"] == 1.0


def test_step_with_garbage_scores_zero(http):
    http.post("/reset", json={})
    r = http.post("/step", json={"action": {"text": "this is not json"}})
    assert r.status_code == 200
    body = r.json()
    assert body["reward"] == 0.0
    assert body["done"] is True
    assert body["observation"]["error"] is None


def test_step_before_reset_is_a_clear_error_not_an_exception(http):
    r = http.post("/step", json={"action": {"text": "{}"}})
    assert r.status_code == 200
    body = r.json()
    assert body["reward"] == 0.0
    assert "reset" in body["observation"]["error"].lower()


def test_step_after_done_is_not_rescored(http, shared_env):
    http.post("/reset", json={})
    action = {"action": {"text": _gt_action(shared_env)}}
    first = http.post("/step", json=action).json()
    second = http.post("/step", json=action)
    assert first["reward"] == 1.0
    assert second.status_code == 200
    body = second.json()
    assert body["reward"] == 0.0
    assert body["done"] is True
    assert "finished" in body["observation"]["error"].lower()
    assert shared_env._env._episode_count == 1


def test_ground_truth_hidden_by_default_and_revealed_on_flag():
    hidden = ConflictBenchEnvironment(auto_scale_difficulty=False, seed=4)
    hidden.reset()
    assert hidden.step(ConflictBenchAction(text="{}")).ground_truth is None

    shown = ConflictBenchEnvironment(
        reveal_ground_truth=True, auto_scale_difficulty=False, seed=4
    )
    shown.reset()
    gt = shown.step(ConflictBenchAction(text="{}")).ground_truth
    assert gt is not None and {"followed", "overridden", "conflicts"} <= set(gt)


def test_invalid_reset_options_return_error_observation():
    env = ConflictBenchEnvironment()
    assert env.reset(difficulty=9).error
    assert env.reset(domain="Nope").error


def test_each_instance_owns_its_generator():
    a, b = ConflictBenchEnvironment(), ConflictBenchEnvironment()
    assert a._env.generator is not b._env.generator
    assert isinstance(a.state, State)


def _ws_send(ws, payload):
    ws.send_text(json.dumps(payload))
    msg = json.loads(ws.receive_text())
    assert msg["type"] == "observation", msg
    return msg["data"]


def test_websocket_session_is_stateful_and_hides_ground_truth():
    with TestClient(default_app) as client:
        with client.websocket_connect("/ws") as ws:
            # step before reset on a persistent session
            early = _ws_send(ws, {"type": "step", "data": {"text": "{}"}})
            assert early["reward"] == 0.0
            assert "reset" in early["observation"]["error"].lower()

            first = _ws_send(ws, {"type": "reset", "data": {"seed": 11}})
            assert first["observation"]["text"] and first["done"] is False

            scored = _ws_send(ws, {"type": "step", "data": {"text": "garbage"}})
            assert scored["reward"] == 0.0 and scored["done"] is True
            assert scored["observation"]["ground_truth"] is None

            again = _ws_send(ws, {"type": "step", "data": {"text": "garbage"}})
            assert "finished" in again["observation"]["error"].lower()

            second = _ws_send(ws, {"type": "reset", "data": {}})
            assert second["done"] is False
            assert second["observation"]["scenario_id"] != first["observation"]["scenario_id"]
