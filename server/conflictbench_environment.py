"""
ConflictBench Environment Implementation for OpenEnv HTTP Server.

Wraps the ConflictBenchEnv gym-style environment into the OpenEnv
Environment interface expected by the HTTP server infrastructure.

Session model (openenv-core 0.2.3): the plain HTTP ``/reset`` and ``/step``
routes build a *fresh* environment instance for every request, so they are
stateless. A real episode (reset then step) needs one persistent session,
i.e. the ``/ws`` WebSocket endpoint (what ``EnvClient`` uses).
"""

import json
import os
import sys
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conflict_bench import ConflictBenchEnv

try:
    from .models import ConflictBenchAction, ConflictBenchObservation
except ImportError:
    from server.models import ConflictBenchAction, ConflictBenchObservation

_DOMAINS = ("HR", "Finance", "IT", "Operations")


class ConflictBenchEnvironment(
    Environment[ConflictBenchAction, ConflictBenchObservation, State]
):
    """
    OpenEnv-compatible wrapper around ConflictBenchEnv.

    Every instance owns its own ConflictBenchEnv (and therefore its own
    ScenarioGenerator); nothing is shared between sessions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        reveal_ground_truth: bool = False,
        auto_scale_difficulty: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            reveal_ground_truth: If True, the post-step observation includes the
                correct answer. Keep False for any server an agent is tested against.
            auto_scale_difficulty: Passed to ConflictBenchEnv.
            seed: Seed for this instance's generator (None = nondeterministic).
        """
        super().__init__()
        self.reveal_ground_truth = reveal_ground_truth
        self._env = ConflictBenchEnv(auto_scale_difficulty=auto_scale_difficulty, seed=seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._started = False
        self._done = False

    @staticmethod
    def _error(message: str, done: bool = False) -> ConflictBenchObservation:
        return ConflictBenchObservation(text=message, error=message, done=done, reward=0.0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ConflictBenchObservation:
        """
        Start a new episode.

        Args:
            seed: Reproducible scenario (a fresh ScenarioGenerator(seed) is used).
            episode_id: Optional custom episode id; defaults to the scenario id.
            **kwargs: Optional ``difficulty`` (1-3) and ``domain`` overrides.
        """
        difficulty = kwargs.get("difficulty")
        domain = kwargs.get("domain")
        if difficulty is not None and difficulty not in (1, 2, 3):
            return self._error(f"Invalid difficulty {difficulty!r}; expected 1, 2 or 3.")
        if domain is not None and domain not in _DOMAINS:
            return self._error(f"Invalid domain {domain!r}; expected one of {list(_DOMAINS)}.")
        if difficulty is not None:
            self._env.fixed_difficulty = difficulty
        if domain is not None:
            self._env.fixed_domain = domain

        obs = self._env.reset(seed=seed)
        self._started = True
        self._done = False
        self._state = State(episode_id=episode_id or obs["scenario_id"], step_count=0)

        return ConflictBenchObservation(
            text=obs["prompt"],
            difficulty=obs["difficulty"],
            domain=obs["domain"],
            num_conflicts=obs["num_conflicts"],
            scenario_id=obs["scenario_id"],
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: ConflictBenchAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ConflictBenchObservation:
        """Score the agent's JSON plan. Single-step episodes."""
        if not self._started:
            return self._error("No active episode: call reset before step.")
        if self._done:
            return self._error(
                "Episode already finished: call reset to start a new one.", done=True
            )

        self._state.step_count += 1
        obs, reward, done, info = self._env.step(action.text)
        self._done = done

        ground_truth = None
        if self.reveal_ground_truth:
            ground_truth = {
                "followed": info["ground_truth_followed"],
                "overridden": info["ground_truth_overridden"],
                "conflicts": info["ground_truth_conflicts"],
            }

        return ConflictBenchObservation(
            text=json.dumps(info["score_breakdown"], indent=2),
            difficulty=obs["difficulty"],
            domain=obs["domain"],
            num_conflicts=obs["num_conflicts"],
            scenario_id=obs["scenario_id"],
            score_breakdown=info["score_breakdown"],
            ground_truth=ground_truth,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
