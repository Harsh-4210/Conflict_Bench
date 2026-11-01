"""
ConflictBench multi-step: the final reward.

    reward = clip( accuracy - revert_penalty - waste_penalty , 0 , 1 )

accuracy
    Compare the final company state with the target state from resolve(). Take every
    cell either state has (a cell is (topic, scope[, task])). A cell is correct when
    both states have it and hold the same value. accuracy = correct cells / all cells.
    A cell that exists only in your state (something applied that should not be) is wrong,
    and so is a cell that exists only in the target (something left out).

revert_penalty = min(REVERT_PENALTY * k, REVERT_PENALTY_CAP)
    k = reverts of an instruction that was already outranked (or void) on the information
    available when it was applied. The agent should have known. A revert forced by a
    message that arrived later costs nothing.

waste_penalty = min(WASTE_PENALTY * w, WASTE_PENALTY_CAP)
    w = invalid tool calls (malformed, unknown ID, already applied, wait with an empty
    inbox, ...) plus reverts of an instruction that was still a winner on the information
    available at the time of the revert.

Per-step reward is 0. The reward is paid once, when the agent calls ``finish`` or the step
budget runs out.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

from .state import Cell

REVERT_PENALTY = 0.03
REVERT_PENALTY_CAP = 0.15
WASTE_PENALTY = 0.01
WASTE_PENALTY_CAP = 0.10


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    accuracy: float
    cells_correct: int
    cells_total: int
    foreseeable_reverts: int
    wasted_steps: int
    revert_penalty: float
    waste_penalty: float
    steps_used: int
    budget: int
    ended_by: str          # "finish" | "budget"

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def state_accuracy(actual: Dict[Cell, str], target: Dict[Cell, str]) -> Tuple[int, int]:
    """(correct cells, all cells) over the union of both states."""
    cells = set(actual) | set(target)
    if not cells:
        return 0, 0
    correct = sum(1 for c in cells if c in actual and c in target and actual[c] == target[c])
    return correct, len(cells)


def final_reward(
    actual: Dict[Cell, str],
    target: Dict[Cell, str],
    foreseeable_reverts: int,
    wasted_steps: int,
    steps_used: int,
    budget: int,
    ended_by: str,
) -> RewardBreakdown:
    correct, total = state_accuracy(actual, target)
    accuracy = correct / total if total else 1.0
    revert_penalty = min(REVERT_PENALTY * max(0, foreseeable_reverts), REVERT_PENALTY_CAP)
    waste_penalty = min(WASTE_PENALTY * max(0, wasted_steps), WASTE_PENALTY_CAP)
    reward = min(1.0, max(0.0, accuracy - revert_penalty - waste_penalty))
    return RewardBreakdown(
        reward=reward, accuracy=accuracy, cells_correct=correct, cells_total=total,
        foreseeable_reverts=foreseeable_reverts, wasted_steps=wasted_steps,
        revert_penalty=revert_penalty, waste_penalty=waste_penalty,
        steps_used=steps_used, budget=budget, ended_by=ended_by,
    )
