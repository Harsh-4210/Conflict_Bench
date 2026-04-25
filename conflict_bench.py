"""
ConflictBench — OpenEnv Environment
Theme: Long-Horizon Instruction Following (Theme 2) + Scale AI sub-theme bonus.

Environment name: ConflictBenchEnv
An LLM agent is given a realistic business instruction document containing
20–35 directives from multiple stakeholders. Some instructions are in direct
conflict. The agent must identify every conflict, resolve it according to
an implicit authority hierarchy, and produce a clean execution plan.

Follows OpenEnv gym-style API: reset() / step() / state property.
Uses OpenEnv's Rubric system for composable multi-dimensional rewards.
"""

import sys
import os
import random
from typing import Optional, Dict, Any, Tuple

# OpenEnv base import — works with latest OpenEnv release
try:
    from openenv.core.env_server import Environment
    OPENENV_AVAILABLE = True
except ImportError:
    try:
        from openenv import Environment
        OPENENV_AVAILABLE = True
    except ImportError:
        # Graceful fallback for local testing without OpenEnv installed
        OPENENV_AVAILABLE = False
        class Environment:
            pass

# Local stubs for Rubric/RubricSet used in smoke testing
class Rubric:
    def __init__(self, name, fn, weight):
        self.name = name
        self.fn = fn
        self.weight = weight

class RubricSet:
    def __init__(self, rubrics):
        self.rubrics = rubrics

sys.path.insert(0, os.path.dirname(__file__))
from generator import ScenarioGenerator, Scenario
from verifier import (
    score,
    parse_agent_output,
    rubric_correct_final_state,
    rubric_no_contradictions,
    rubric_conflict_identification,
    rubric_efficiency,
    rubric_format_compliance,
    ScoreBreakdown,
)


class ConflictBenchEnv(Environment):
    """
    ConflictBench: Instruction Priority Resolution Environment.

    Each episode:
      - reset() generates a fresh business instruction scenario
      - The agent observes a formatted instruction document as text
      - step(action) receives the agent's JSON plan, scores it, returns reward
      - Episode is single-step (one plan per scenario)

    Difficulty auto-scales with episode count during training.
    """

    metadata = {
        "name": "ConflictBenchEnv",
        "version": "1.0.0",
        "theme": "Long-Horizon Instruction Following",
        "sub_theme": "Scale AI — Long-Horizon Workflows (non-code, business setting)",
        "authors": ["Harsh Jain"],
        "description": (
            "An LLM environment where the agent receives a realistic business "
            "instruction document with embedded conflicts and must resolve them "
            "according to an implicit stakeholder authority hierarchy."
        ),
    }

    def __init__(
        self,
        auto_scale_difficulty: bool = True,
        fixed_difficulty: Optional[int] = None,
        fixed_domain: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            auto_scale_difficulty: If True, difficulty increases as training progresses.
            fixed_difficulty: Override difficulty (1, 2, or 3). Ignored if auto_scale.
            fixed_domain: Fix domain to "HR", "Finance", "IT", or "Operations".
            seed: Random seed for reproducibility.
        """
        self.auto_scale_difficulty = auto_scale_difficulty
        self.fixed_difficulty = fixed_difficulty
        self.fixed_domain = fixed_domain

        self.generator = ScenarioGenerator(seed=seed)
        self._episode_count = 0
        self._current_scenario: Optional[Scenario] = None
        self._current_score: Optional[ScoreBreakdown] = None
        self._done = False

        # Register OpenEnv rubrics if available
        if OPENENV_AVAILABLE:
            self._rubric_set = self._build_rubric_set()

    # ------------------------------------------------------------------
    # OpenEnv Rubric construction
    # ------------------------------------------------------------------

    def _build_rubric_set(self) -> RubricSet:
        """Construct OpenEnv RubricSet with all 5 composable rubrics."""

        def _make_rubric_fn(rubric_func):
            """Wrap a rubric function so it receives (output_str, scenario)."""
            def fn(agent_output: str, **kwargs) -> float:
                if self._current_scenario is None:
                    return 0.0
                return rubric_func(parse_agent_output(agent_output), self._current_scenario)
            return fn

        rubrics = [
            Rubric(
                name="correct_final_state",
                fn=_make_rubric_fn(rubric_correct_final_state),
                weight=0.35,
            ),
            Rubric(
                name="no_contradictions",
                fn=_make_rubric_fn(rubric_no_contradictions),
                weight=0.25,
            ),
            Rubric(
                name="conflict_identification",
                fn=_make_rubric_fn(rubric_conflict_identification),
                weight=0.20,
            ),
            Rubric(
                name="efficiency",
                fn=_make_rubric_fn(rubric_efficiency),
                weight=0.10,
            ),
            Rubric(
                name="format_compliance",
                fn=_make_rubric_fn(rubric_format_compliance),
                weight=0.10,
            ),
        ]
        return RubricSet(rubrics)

    # ------------------------------------------------------------------
    # Difficulty scaling
    # ------------------------------------------------------------------

    def _get_difficulty(self) -> int:
        """
        Auto-scale difficulty with episode count.
        Episodes 0–199: difficulty 1 (2 conflicts, warm-up)
        Episodes 200–499: difficulty 2 (4 conflicts)
        Episodes 500+: difficulty 3 (6 conflicts, hardest)
        """
        if not self.auto_scale_difficulty or self.fixed_difficulty is not None:
            return self.fixed_difficulty or 2

        if self._episode_count < 200:
            return 1
        elif self._episode_count < 500:
            return 2
        else:
            return 3

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """
        Start a new episode. Generates a fresh scenario.

        Returns:
            observation dict with:
              - "prompt": full formatted instruction document (str)
              - "difficulty": current difficulty level (int)
              - "domain": business domain (str)
              - "num_conflicts": number of conflicts in this scenario (int)
              - "scenario_id": unique episode ID (str)
        """
        difficulty = self._get_difficulty()
        self._current_scenario = self.generator.generate(
            difficulty=difficulty,
            domain=self.fixed_domain,
        )
        self._current_score = None
        self._done = False

        return {
            "prompt": self._current_scenario.prompt,
            "difficulty": self._current_scenario.difficulty,
            "domain": self._current_scenario.domain,
            "num_conflicts": len(self._current_scenario.conflicts),
            "scenario_id": self._current_scenario.scenario_id,
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step: score the agent's plan.

        Args:
            action: Raw string output from the LLM (should be a JSON plan).

        Returns:
            Tuple of (observation, reward, done, info):
              - observation: same as reset() observation (episode is single-step)
              - reward: composite float in [0.0, 1.0]
              - done: always True after one step
              - info: dict with score breakdown + ground truth
        """
        if self._current_scenario is None:
            raise RuntimeError("Must call reset() before step()")

        self._current_score = score(action, self._current_scenario)
        self._episode_count += 1
        self._done = True

        observation = {
            "prompt": self._current_scenario.prompt,
            "difficulty": self._current_scenario.difficulty,
            "domain": self._current_scenario.domain,
            "num_conflicts": len(self._current_scenario.conflicts),
            "scenario_id": self._current_scenario.scenario_id,
        }

        info = {
            "score_breakdown": self._current_score.to_dict(),
            "ground_truth_followed": self._current_scenario.ground_truth_followed,
            "ground_truth_overridden": self._current_scenario.ground_truth_overridden,
            "ground_truth_conflicts": [
                {
                    "pair": [c.instruction_a_id, c.instruction_b_id],
                    "resolution": c.resolution_id,
                    "explanation": c.explanation,
                }
                for c in self._current_scenario.conflicts
            ],
            "episode_count": self._episode_count,
        }

        return observation, self._current_score.composite, True, info

    @property
    def state(self) -> Dict[str, Any]:
        """
        Current environment state (OpenEnv required property).
        Returns the current observation + last score if available.
        """
        if self._current_scenario is None:
            return {"status": "not_started"}

        state = {
            "scenario_id": self._current_scenario.scenario_id,
            "domain": self._current_scenario.domain,
            "difficulty": self._current_scenario.difficulty,
            "num_instructions": len(self._current_scenario.instructions),
            "num_conflicts": len(self._current_scenario.conflicts),
            "episode_count": self._episode_count,
            "done": self._done,
        }

        if self._current_score is not None:
            state["last_score"] = self._current_score.to_dict()

        return state

    def close(self):
        """Clean up resources (no-op for this environment)."""
        pass

    # ------------------------------------------------------------------
    # Utility — useful for demos and testing
    # ------------------------------------------------------------------

    def sample_action(self) -> str:
        """
        Return the ground truth answer as a sample action.
        Useful for sanity checks and baseline evaluation.
        """
        if self._current_scenario is None:
            raise RuntimeError("Must call reset() first.")
        sc = self._current_scenario
        return {
            "identified_conflicts": [
                {
                    "instruction_a": c.instruction_a_id,
                    "instruction_b": c.instruction_b_id,
                    "conflict_type": c.conflict_type,
                    "resolution": c.resolution_id,
                    "reasoning": c.explanation,
                }
                for c in sc.conflicts
            ],
            "execution_plan": sc.ground_truth_followed,
            "overridden_instructions": sc.ground_truth_overridden,
        }

    def render(self) -> str:
        """Return a human-readable render of the current scenario state."""
        if self._current_scenario is None:
            return "No active scenario. Call reset() first."

        sc = self._current_scenario
        lines = [
            f"=== ConflictBench Episode ===",
            f"Domain: {sc.domain} | Difficulty: {sc.difficulty} | Conflicts: {len(sc.conflicts)}",
            f"Instructions: {len(sc.instructions)}",
            "",
            sc.business_context,
            "",
        ]
        for ins in sc.instructions:
            lines.append(f"  [{ins.id}] {ins.source} (authority {ins.source_priority}): {ins.text[:80]}...")
        if self._current_score:
            lines.append("")
            lines.append(f"Last score: {self._current_score.composite:.3f}")
            for k, v in self._current_score.to_dict().items():
                if k != "composite":
                    lines.append(f"  {k}: {v:.3f}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== ConflictBench Smoke Test ===\n")
    env = ConflictBenchEnv(auto_scale_difficulty=False, fixed_difficulty=2, seed=42)

    obs = env.reset()
    print(f"Domain: {obs['domain']} | Difficulty: {obs['difficulty']} | Conflicts: {obs['num_conflicts']}")
    print(f"\n--- Prompt preview (first 800 chars) ---")
    print(obs["prompt"][:800])
    print("...\n")

    # Test with ground truth answer
    gt_action = json.dumps(env.sample_action())
    _, reward, done, info = env.step(gt_action)
    print(f"Ground truth score: {reward:.4f}")
    print(f"Breakdown: {info['score_breakdown']}")

    # Test with blank (zero) answer
    env.reset()
    _, reward, done, info = env.step("{}")
    print(f"\nBlank response score: {reward:.4f}")
    print(f"State: {env.state}")
