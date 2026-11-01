"""
Tests for the multi-step OpenEnv server (multistep/server.py).

A real episode needs one persistent session, so these run over the /ws WebSocket
(plain HTTP /step is stateless in openenv-core 0.2.3).
"""

import json

import pytest
from fastapi.testclient import TestClient

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import State

from multistep.env import MultiStepConflictEnv
from multistep.server import MultiStepAction, MultiStepEnvironment, MultiStepObservation, app as default_app
from scripts import evaluate_multistep


def _send(ws, payload):
    ws.send_text(json.dumps(payload))
    msg = json.loads(ws.receive_text())
    assert msg["type"] == "observation", msg
    return msg["data"]


def _step(ws, action):
    return _send(ws, {"type": "step", "data": {"text": json.dumps(action)}})


def _oracle_actions(seed, difficulty):
    """The oracle's whole action list for an episode, found on a local copy of the same episode."""
    local = MultiStepConflictEnv()
    local.reset(seed=seed, difficulty=difficulty)
    actions = []
    for action in evaluate_multistep.oracle(local):
        actions.append(action)
        _, _, done, _ = local.step(action)
        if done:
            return actions
    raise AssertionError("oracle did not finish")


def test_app_exposes_routes():
    paths = {r.path for r in default_app.routes}
    assert {"/reset", "/step", "/state", "/health", "/ws"} <= paths


def test_websocket_full_episode_with_the_oracle_scores_one():
    seed, difficulty = 31, 2
    actions = _oracle_actions(seed, difficulty)
    with TestClient(default_app) as client, client.websocket_connect("/ws") as ws:
        first = _send(ws, {"type": "reset", "data": {"seed": seed, "difficulty": difficulty}})
        assert first["done"] is False and first["reward"] in (0, 0.0)
        obs = first["observation"]
        assert obs["new_messages"] and obs["budget"] > 0 and obs["error"] is None
        assert obs["ground_truth"] is None and obs["reward_breakdown"] is None
        # the served episode is the local one
        local = MultiStepConflictEnv()
        assert local.reset(seed=seed, difficulty=difficulty)["scenario_id"] == obs["scenario_id"]

        last = None
        for i, action in enumerate(actions):
            last = _step(ws, action)
            assert last["observation"]["error"] is None, (action, last["observation"]["error"])
            if i < len(actions) - 1:
                assert last["done"] is False and last["reward"] in (0, 0.0)
                assert last["observation"]["ground_truth"] is None
        assert last["done"] is True and last["reward"] == pytest.approx(1.0)
        final = last["observation"]
        assert final["reward_breakdown"]["accuracy"] == 1.0
        assert final["ground_truth"] is None            # hidden unless the server reveals it

        again = _step(ws, {"tool": "wait"})
        assert "finished" in again["observation"]["error"].lower() and again["reward"] in (0, 0.0)

        second = _send(ws, {"type": "reset", "data": {"seed": seed + 1}})
        assert second["done"] is False
        assert second["observation"]["scenario_id"] != obs["scenario_id"]


def test_websocket_invalid_actions_and_early_step():
    with TestClient(default_app) as client, client.websocket_connect("/ws") as ws:
        early = _send(ws, {"type": "step", "data": {"text": json.dumps({"tool": "wait"})}})
        assert "reset" in early["observation"]["error"].lower() and early["reward"] in (0, 0.0)

        _send(ws, {"type": "reset", "data": {"seed": 4, "difficulty": 1}})
        for bad in ("not json", json.dumps({"tool": "dance"}), json.dumps({"tool": "apply", "id": "INS-NOPE"})):
            out = _send(ws, {"type": "step", "data": {"text": bad}})
            assert out["observation"]["error"] and out["done"] is False
        assert out["observation"]["steps_used"] == 3

        # waiting works and shows the next messages
        nxt = _step(ws, {"tool": "wait"})
        assert nxt["observation"]["error"] is None and nxt["observation"]["new_messages"]


def test_websocket_budget_ends_the_episode():
    with TestClient(default_app) as client, client.websocket_connect("/ws") as ws:
        first = _send(ws, {"type": "reset", "data": {"seed": 6, "difficulty": 1}})
        budget = first["observation"]["budget"]
        out = None
        for _ in range(budget):
            out = _send(ws, {"type": "step", "data": {"text": "garbage"}})
        assert out["done"] is True
        assert out["observation"]["reward_breakdown"]["ended_by"] == "budget"
        assert out["reward"] == pytest.approx(0.0)


def test_websocket_sessions_are_independent():
    with TestClient(default_app) as client:
        with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
            _send(a, {"type": "reset", "data": {"seed": 1}})
            _send(b, {"type": "reset", "data": {"seed": 2}})
            _step(a, {"tool": "wait"})
            out_b = _step(b, {"tool": "finish"})
            assert out_b["done"] is True
            out_a = _step(a, {"tool": "wait"})
            assert out_a["done"] is False and out_a["observation"]["steps_used"] == 2


def test_reveal_ground_truth_flag_only_on_the_final_observation():
    shared = create_app(
        lambda: MultiStepEnvironment(reveal_ground_truth=True), MultiStepAction, MultiStepObservation,
        env_name="conflictbench-multistep-test",
    )
    with TestClient(shared) as client, client.websocket_connect("/ws") as ws:
        _send(ws, {"type": "reset", "data": {"seed": 3, "difficulty": 1}})
        mid = _step(ws, {"tool": "wait"})
        assert mid["observation"]["ground_truth"] is None
        end = _step(ws, {"tool": "finish"})
        truth = end["observation"]["ground_truth"]
        assert truth is not None and {"followed", "overridden", "target_state"} <= set(truth)


def test_invalid_reset_options_return_error_observations():
    env = MultiStepEnvironment()
    assert env.reset(difficulty=99).error
    assert env.reset(difficulty="2").error
    assert env.reset(difficulty=True).error
    assert env.reset(seed="x").error
    ok = env.reset(seed=1, difficulty=1)
    assert ok.error is None and isinstance(env.state, State) and env.state.step_count == 0


def test_environment_wrapper_hides_truth_and_counts_steps():
    env = MultiStepEnvironment()
    env.reset(seed=2, difficulty=1)
    obs = env.step(MultiStepAction(text='{"tool": "finish"}'))
    assert obs.done and obs.ground_truth is None and env.state.step_count == 1
    assert "finished" in env.step(MultiStepAction(text='{"tool": "wait"}')).error.lower()
    assert env.step(MultiStepAction(text='{"tool": "wait"}')).done is True
