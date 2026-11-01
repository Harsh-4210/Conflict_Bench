"""
ConflictBench multi-step: company state, episode schedule, target state.

Nothing here talks to an agent. It holds the three plain-data pieces the
environment is built on:

  * ``CompanyState``     what has been applied so far and what the company looks like now
  * ``build_schedule``   when each message (and each status fact) arrives, seeded
  * ``target_cells``     the state the scenario's ground truth implies (from resolve())

Ground truth comes only from ``resolver.resolve``; this module never re-derives it.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from generator import Instruction, Scenario
from resolver import resolve

try:  # fact phrasings live in the generator; fall back to a plain sentence if they move
    from generator import V2_FACTS
except ImportError:  # pragma: no cover
    V2_FACTS = {}

# (topic, scope, slot). ``slot`` is "" for an ordinary instruction and the instruction's
# own ID for a contingent follow-up (see cell_of).
Cell = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Company state
# ---------------------------------------------------------------------------

def cell_of(ins: Instruction) -> Cell:
    """
    The state cell an instruction writes to: (topic, scope, slot).

    Ordinary instructions share a cell when topic and scope match, so a later write
    overwrites an earlier one. A contingent follow-up (``depends_on`` set) is a separate
    task ("inform the works council"), not a setting, so it gets its own cell keyed by its
    ID. Without that, applying a cancelled follow-up would score the same as applying the
    live one, because both write "carry_out" to the same ("follow_up", "all") cell.
    """
    scope = getattr(ins, "action_scope", "") or "all"
    slot = ins.id if getattr(ins, "depends_on", None) else ""
    return (ins.action_key, scope, slot)


def cell_to_dict(cell: Cell, value: str, set_by: str) -> Dict[str, Any]:
    out = {"topic": cell[0], "scope": cell[1], "value": value, "set_by": set_by}
    if cell[2]:
        out["task"] = cell[2]
    return out


class CompanyState:
    """
    The company's current settings, as a function of the instructions applied so far.

    Applied instructions are kept in the order they were applied. A cell's value is the
    one written by the most recently applied instruction on that cell, so applying a
    newer instruction overwrites an older one, and reverting the newer one brings the
    older value back.
    """

    def __init__(self) -> None:
        self._applied: List[Instruction] = []

    def apply(self, ins: Instruction) -> None:
        self._applied.append(ins)

    def revert(self, instruction_id: str) -> bool:
        for k, ins in enumerate(self._applied):
            if ins.id == instruction_id:
                del self._applied[k]
                return True
        return False

    def is_applied(self, instruction_id: str) -> bool:
        return any(ins.id == instruction_id for ins in self._applied)

    def applied_ids(self) -> List[str]:
        return [ins.id for ins in self._applied]

    def cells(self) -> Dict[Cell, Tuple[str, str]]:
        """cell -> (value, id of the instruction that set it)."""
        out: Dict[Cell, Tuple[str, str]] = {}
        for ins in self._applied:
            out[cell_of(ins)] = (ins.action_value, ins.id)
        return out

    def values(self) -> Dict[Cell, str]:
        return {cell: v for cell, (v, _) in self.cells().items()}

    def as_list(self) -> List[Dict[str, Any]]:
        """JSON-friendly view for observations, sorted by (topic, scope, task)."""
        return [cell_to_dict(c, v, i) for c, (v, i) in sorted(self.cells().items())]


def target_cells(scenario: Scenario, order: Optional[Sequence[str]] = None) -> Dict[Cell, str]:
    """
    The state implied by ground truth: apply every instruction ``resolve(scenario)``
    says to follow, in arrival order, and read off the cells.
    """
    followed = set(resolve(scenario)[0])
    by_id = {ins.id: ins for ins in scenario.instructions}
    ids = list(order) if order is not None else [ins.id for ins in scenario.instructions]
    state = CompanyState()
    for iid in ids:
        if iid in followed:
            state.apply(by_id[iid])
    return state.values()


# ---------------------------------------------------------------------------
# Episode schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    index: int
    message_ids: Tuple[str, ...]
    fact_keys: Tuple[str, ...] = ()   # status facts announced on this turn


@dataclass(frozen=True)
class Schedule:
    seed: int
    turns: Tuple[Turn, ...]

    @property
    def order(self) -> Tuple[str, ...]:
        return tuple(i for t in self.turns for i in t.message_ids)


def build_schedule(scenario: Scenario, seed: int, max_per_turn: int = 3) -> Schedule:
    """
    Split a scenario's instructions into turns, deterministically for a given seed.

    Messages arrive in ``effective_date`` order (undated ones keep their presentation
    order; equal dates are shuffled by the seed). Each turn delivers 1..max_per_turn
    messages. Every status fact in ``scenario.facts`` is announced on one turn, drawn
    uniformly over all turns, so a conditional instruction can arrive before or after the
    fact that decides it.
    """
    rng = random.Random(f"conflictbench-multistep:{seed}")
    ins_list = list(scenario.instructions)
    jitter = {ins.id: rng.random() for ins in ins_list}
    ordered = sorted(ins_list, key=lambda i: (getattr(i, "effective_date", None) or "", jitter[i.id]))

    chunks: List[List[str]] = []
    k = 0
    while k < len(ordered):
        size = rng.randint(1, max(1, max_per_turn))
        chunks.append([i.id for i in ordered[k:k + size]])
        k += size
    if not chunks:
        chunks = [[]]

    facts_at: Dict[int, List[str]] = {t: [] for t in range(len(chunks))}
    for key in sorted(getattr(scenario, "facts", None) or {}):
        facts_at[rng.randrange(len(chunks))].append(key)

    turns = tuple(Turn(t, tuple(chunks[t]), tuple(facts_at[t])) for t in range(len(chunks)))
    return Schedule(seed=seed, turns=turns)


# ---------------------------------------------------------------------------
# Text for observations
# ---------------------------------------------------------------------------

def fact_line(key: str, value: bool) -> str:
    """The status sentence for a fact, as the single-step prompt phrases it."""
    if key in V2_FACTS:
        return V2_FACTS[key][1] if value else V2_FACTS[key][2]
    return f"Status: {key.replace('_', ' ')} is {'true' if value else 'false'}."


def message_dict(ins: Instruction, turn: int) -> Dict[str, Any]:
    """What the agent sees of an instruction. No source tier, no polarity, no verdict."""
    return {
        "id": ins.id,
        "from": getattr(ins, "author", "") or ins.source,
        "title": getattr(ins, "title", "") or ins.source,
        "date": getattr(ins, "effective_date", None),
        "text": ins.text,
        "arrived_turn": turn,
    }


def standing_notices(scenario: Scenario) -> List[str]:
    """
    The scenario's standing notices (final-say rules and delegations) as sentences.

    Taken verbatim from the STANDING NOTICES section of the single-step prompt when it has
    one (so names and titles match the messages). Otherwise rendered from the structured
    rules by office name.
    """
    lines: List[str] = []
    in_section = False
    for raw in (getattr(scenario, "prompt", "") or "").splitlines():
        line = raw.rstrip()
        if line.strip() == "STANDING NOTICES:":
            in_section = True
            continue
        if in_section:
            if line.startswith("- "):
                lines.append(line[2:].strip())
            else:
                break
    if lines:
        return lines
    out = []
    for r in getattr(scenario, "authority_rules", None) or []:
        out.append(f"{r.get('office')} has the final say on {str(r.get('action_key', '')).replace('_', ' ')}.")
    for d in getattr(scenario, "delegations", None) or []:
        out.append(
            f"{d.get('from_office')} has delegated their authority on "
            f"{str(d.get('action_key', '')).replace('_', ' ')} to {d.get('to_office')}."
        )
    return out
