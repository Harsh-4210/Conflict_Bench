"""
Action and observation models for the ConflictBench OpenEnv server.

openenv-core 0.2.3 ships no ready-made text action/observation types, so these
subclass the library's pydantic ``Action`` / ``Observation`` base classes.

Note: the HTTP/WebSocket layer serialises an observation as
``{"observation": <all fields except done/reward/metadata>, "reward", "done"}``.
``metadata`` is dropped on the wire, so everything a client needs lives in
explicit fields below.
"""

from typing import Any, Dict, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class ConflictBenchAction(Action):
    """The agent's answer: the raw text of its JSON resolution plan."""

    text: str = Field(description="The agent's response (a JSON resolution plan)")


class ConflictBenchObservation(Observation):
    """What the agent sees. ``done`` and ``reward`` come from the base class."""

    text: str = Field(
        default="",
        description="Instruction document on reset; score summary (JSON) after a step",
    )
    difficulty: Optional[int] = Field(default=None, description="Scenario difficulty 1-3")
    domain: Optional[str] = Field(default=None, description="Business domain")
    num_conflicts: Optional[int] = Field(default=None, description="Number of planted conflicts")
    scenario_id: Optional[str] = Field(default=None, description="Scenario identifier")
    score_breakdown: Optional[Dict[str, float]] = Field(
        default=None, description="Per-rubric scores (set after a scored step)"
    )
    error: Optional[str] = Field(
        default=None, description="Set when the request could not be processed"
    )
    ground_truth: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Correct answer; only present if the server was started with reveal_ground_truth",
    )
