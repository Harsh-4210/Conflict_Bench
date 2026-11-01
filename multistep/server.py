"""
OpenEnv server for the multi-step ConflictBench environment.

Wraps ``MultiStepConflictEnv`` in the OpenEnv ``Environment`` interface, the same pattern
as server/conflictbench_environment.py.

Session model (openenv-core 0.2.3): the plain HTTP ``/reset`` and ``/step`` routes build a
*fresh* environment for every request, so they are stateless. A real episode needs one
persistent session: the ``/ws`` WebSocket endpoint (what ``EnvClient`` uses).

Usage:
    uvicorn multistep.server:app --host 0.0.0.0 --port 8001
    python -m multistep.server --port 8001
"""

import os
import sys
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import MAX_DIFFICULTY
from multistep.env import MultiStepConflictEnv


class MultiStepAction(Action):
    """One tool call, as the raw text of a JSON object, e.g. {"tool": "apply", "id": "INS-1A2B3C"}."""

    text: str = Field(description='A JSON tool call such as {"tool": "apply", "id": "INS-..."}')


class MultiStepObservation(Observation):
    """What the agent sees after each call. ``done`` and ``reward`` come from the base class."""

    text: str = Field(default="", description="Readable view of everything below")
    scenario_id: Optional[str] = Field(default=None)
    difficulty: Optional[int] = Field(default=None)
    domain: Optional[str] = Field(default=None)
    turn: int = Field(default=0, description="Index of the latest turn delivered")
    more_messages: bool = Field(default=False, description="True while `wait` will still bring new messages")
    new_messages: List[Dict[str, Any]] = Field(default_factory=list)
    status_updates: List[str] = Field(default_factory=list)
    state: List[Dict[str, Any]] = Field(default_factory=list, description="Current company state, one entry per cell")
    inbox: List[Dict[str, str]] = Field(default_factory=list, description="Every arrived ID and what you did with it")
    steps_used: int = Field(default=0)
    steps_remaining: int = Field(default=0)
    budget: int = Field(default=0)
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    feedback: str = Field(default="", description="Result of the last call")
    error: Optional[str] = Field(default=None, description="Set when the last call was invalid")
    reward_breakdown: Optional[Dict[str, Any]] = Field(default=None, description="Set on the final observation")
    ground_truth: Optional[Dict[str, Any]] = Field(
        default=None, description="Correct answer; only when the server was started with reveal_ground_truth"
    )


class MultiStepEnvironment(Environment[MultiStepAction, MultiStepObservation, State]):
    """OpenEnv wrapper. Every instance owns its own MultiStepConflictEnv (no shared state)."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, reveal_ground_truth: bool = False, seed: Optional[int] = None, **env_kwargs: Any):
        """
        Args:
            reveal_ground_truth: If True, the final observation includes the correct answer.
                Keep False for any server an agent is tested against.
            seed: Seed for the stream of episode seeds (None = nondeterministic).
            **env_kwargs: Passed to MultiStepConflictEnv (budget_factor, max_per_turn, ...).
        """
        super().__init__()
        self.reveal_ground_truth = reveal_ground_truth
        self._env = MultiStepConflictEnv(seed=seed, **env_kwargs)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._started = False
        self._done = False

    @staticmethod
    def _wrap(obs: Dict[str, Any], ground_truth: Optional[Dict[str, Any]] = None) -> MultiStepObservation:
        fields = {k: v for k, v in obs.items() if k in MultiStepObservation.model_fields}
        return MultiStepObservation(**fields, ground_truth=ground_truth)

    @staticmethod
    def _error(message: str, done: bool = False) -> MultiStepObservation:
        return MultiStepObservation(text=message, error=message, done=done, reward=0.0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MultiStepObservation:
        """
        Start a new episode.

        Args:
            seed: Reproducible episode (same seed and difficulty -> same messages, same schedule).
            episode_id: Optional custom episode id; defaults to the scenario id.
            **kwargs: Optional ``difficulty`` (1-MAX_DIFFICULTY).
        """
        difficulty = kwargs.get("difficulty")
        if difficulty is not None and (
            isinstance(difficulty, bool) or not isinstance(difficulty, int)
            or difficulty not in range(1, MAX_DIFFICULTY + 1)
        ):
            return self._error(f"Invalid difficulty {difficulty!r}; expected 1-{MAX_DIFFICULTY}.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            return self._error(f"Invalid seed {seed!r}; expected an integer.")
        obs = self._env.reset(seed=seed, difficulty=difficulty)
        self._started = True
        self._done = False
        self._state = State(episode_id=episode_id or obs["scenario_id"], step_count=0)
        return self._wrap(obs)

    def step(
        self,
        action: MultiStepAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MultiStepObservation:
        """Run one tool call. Per-step reward is 0; the final reward arrives with done=True."""
        if not self._started:
            return self._error("No active episode: call reset before step.")
        if self._done:
            return self._error("Episode already finished: call reset to start a new one.", done=True)
        self._state.step_count += 1
        obs, _, done, info = self._env.step(action.text)
        self._done = done
        truth = info.get("ground_truth") if (done and self.reveal_ground_truth) else None
        return self._wrap(obs, ground_truth=truth)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state


# The environment class is the factory: one fresh, isolated instance per session.
# reveal_ground_truth defaults to False, so served observations never leak answers.
app = create_app(
    MultiStepEnvironment,
    MultiStepAction,
    MultiStepObservation,
    env_name="conflictbench-multistep",
    max_concurrent_envs=10,
)


def main():
    """Entry point for direct execution: python -m multistep.server [--host H] [--port P]."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
