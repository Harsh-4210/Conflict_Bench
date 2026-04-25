"""
ConflictBench Environment Implementation for OpenEnv HTTP Server.

Wraps the ConflictBenchEnv gym-style environment into the OpenEnv
Environment interface expected by the HTTP server infrastructure.
"""

import json
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State, TextAction, TextObservation

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from conflict_bench import ConflictBenchEnv


class ConflictBenchEnvironment(Environment):
    """
    OpenEnv-compatible wrapper around ConflictBenchEnv.

    Translates the gym-style reset()/step() API into the OpenEnv
    Environment interface used by the HTTP server.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the ConflictBench environment."""
        self._env = ConflictBenchEnv(
            auto_scale_difficulty=True,
            seed=None,
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> TextObservation:
        """
        Reset the environment and generate a new scenario.

        Returns:
            TextObservation with the instruction document as text.
        """
        obs = self._env.reset()
        self._state = State(episode_id=obs["scenario_id"], step_count=0)

        return TextObservation(
            text=obs["prompt"],
            done=False,
            reward=0.0,
            metadata={
                "difficulty": obs["difficulty"],
                "domain": obs["domain"],
                "num_conflicts": obs["num_conflicts"],
                "scenario_id": obs["scenario_id"],
            },
        )

    def step(self, action: TextAction) -> TextObservation:
        """
        Execute one step: score the agent's JSON plan.

        Args:
            action: TextAction containing the agent's JSON response.

        Returns:
            TextObservation with score breakdown and reward.
        """
        self._state.step_count += 1
        _, reward, done, info = self._env.step(action.text)

        return TextObservation(
            text=json.dumps(info["score_breakdown"], indent=2),
            done=True,
            reward=reward,
            metadata=info,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
