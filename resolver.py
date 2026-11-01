"""
ConflictBench — Deterministic resolver
The single place where ground truth comes from (FIX_BLUEPRINT.md P3-4).

``resolve(scenario)`` reads only the structured fields of a scenario: each
instruction's action_key / action_value / action_scope / source /
source_priority / effective_date / condition / depends_on, plus the scenario's
facts, authority_rules and delegations. It never looks at text, template order
or presentation order (except to order its output deterministically).

Rules, in order:
  1. Void: an instruction whose ``condition`` names a fact that is not True in
     ``scenario.facts`` is not in force. It is overridden and takes part in no
     conflict.
  2. Conflict: two in-force instructions conflict when they share an
     action_key, their action_values differ, and their scopes overlap ("all"
     overlaps everything; otherwise scopes must be equal).
  3. Authority on a topic (lower rank wins):
       - an authority rule naming the instruction's office for that topic -> rank 0
       - else the raw tier, lowered to the delegator's tier if the office holds a
         delegation for that topic
  4. Equal rank: the later effective_date wins (supersession). Equal rank and no
     usable dates -> UnresolvableConflict.
  5. Chains: an instruction that ``depends_on`` another one is cancelled
     (overridden) when that one is not followed. Repeated until stable.

Works unchanged on legacy scenarios (no scopes, dates, facts or notices).
"""

from typing import Dict, List, Optional, Sequence, Tuple

from generator import CONFLICT_TYPE_LABELS, SOURCE_PRIORITY, ConflictPair, Instruction, Scenario


class UnresolvableConflict(ValueError):
    """Two conflicting instructions have equal authority and no dates to separate them."""


ALL_SCOPE = "all"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def scopes_overlap(a: str, b: str) -> bool:
    """'all' (or empty, for old data) overlaps everything; otherwise scopes must match."""
    a = a or ALL_SCOPE
    b = b or ALL_SCOPE
    return a == ALL_SCOPE or b == ALL_SCOPE or a == b


def in_force(ins: Instruction, facts: Dict[str, bool]) -> bool:
    """An unconditional instruction is always in force; a conditional one only if its fact is True."""
    return ins.condition is None or facts.get(ins.condition) is True


def would_conflict(a: Instruction, b: Instruction) -> bool:
    """Same topic, different direction, overlapping scope. (Ignores conditions and chains.)"""
    return (
        a.action_key == b.action_key
        and a.action_value != b.action_value
        and scopes_overlap(getattr(a, "action_scope", ALL_SCOPE), getattr(b, "action_scope", ALL_SCOPE))
    )


def effective_rank(ins: Instruction, scenario: Scenario) -> Tuple[int, str]:
    """(rank, reason) of an instruction on its own topic. Lower rank = more authority."""
    for rule in getattr(scenario, "authority_rules", None) or []:
        if rule.get("action_key") == ins.action_key and rule.get("office") == ins.source:
            return 0, "rule"
    rank, reason = ins.source_priority, "tier"
    for d in getattr(scenario, "delegations", None) or []:
        if d.get("action_key") == ins.action_key and d.get("to_office") == ins.source:
            delegated = SOURCE_PRIORITY[d["from_office"]]
            if delegated < rank:
                rank, reason = delegated, "delegation"
    return rank, reason


def pick_winner(a: Instruction, b: Instruction, scenario: Scenario) -> Tuple[Instruction, str]:
    """Winner of a conflicting pair and the rule that decided it ('tier'|'rule'|'delegation'|'time')."""
    (ra, why_a), (rb, why_b) = effective_rank(a, scenario), effective_rank(b, scenario)
    if ra != rb:
        return (a, why_a) if ra < rb else (b, why_b)
    da, db = getattr(a, "effective_date", None), getattr(b, "effective_date", None)
    if da and db and da != db:
        return (a if da > db else b), "time"
    raise UnresolvableConflict(
        f"{a.id} and {b.id} have equal authority (rank {ra}) on {a.action_key} and no dates to separate them"
    )


def _explain(winner: Instruction, loser: Instruction, how: str, scenario: Scenario) -> str:
    topic = winner.action_key.replace("_", " ")
    if how == "rule":
        return f"{winner.source} has final authority on {topic} under a standing rule, so it overrides {loser.source}."
    if how == "delegation":
        d = next(
            d for d in scenario.delegations
            if d.get("action_key") == winner.action_key and d.get("to_office") == winner.source
        )
        return (
            f"{winner.source} holds {d['from_office']}'s delegated authority on {topic}, "
            f"so it overrides {loser.source}."
        )
    if how == "time":
        return (
            f"{winner.source} and {loser.source} have equal authority on {topic}; the later instruction "
            f"({winner.effective_date}) supersedes the earlier one ({loser.effective_date})."
        )
    return (
        f"{winner.source} (authority level {winner.source_priority}) overrides "
        f"{loser.source} (authority level {loser.source_priority}) on {topic}."
    )


def _conflict_type(a: Instruction, b: Instruction, how: str) -> str:
    if how == "time":
        return CONFLICT_TYPE_LABELS["time_ordered"]
    if a.condition is not None or b.condition is not None:
        return CONFLICT_TYPE_LABELS["conditional"]
    return CONFLICT_TYPE_LABELS.get(a.instruction_type, "direct")


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

def find_conflicts(instructions: Sequence[Instruction]) -> List[Tuple[Instruction, Instruction]]:
    """Every conflicting pair among `instructions`, in presentation order (a before b)."""
    pairs = []
    for i, a in enumerate(instructions):
        for b in instructions[i + 1:]:
            if would_conflict(a, b):
                pairs.append((a, b))
    return pairs


def resolve(scenario: Scenario) -> Tuple[List[str], List[str], List[ConflictPair]]:
    """
    Ground truth from structured fields only.

    Returns (followed_ids, overridden_ids, conflicts): both ID lists sorted;
    conflicts in presentation order, each with instruction_a before instruction_b
    as they appear in ``scenario.instructions``.
    """
    instructions = list(scenario.instructions)
    facts = getattr(scenario, "facts", None) or {}
    void = {ins.id for ins in instructions if not in_force(ins, facts)}

    while True:
        active = [ins for ins in instructions if ins.id not in void]
        decided = []
        losers = set()
        for a, b in find_conflicts(active):
            winner, how = pick_winner(a, b, scenario)
            loser = b if winner is a else a
            decided.append((a, b, winner, loser, how))
            losers.add(loser.id)
        followed = {ins.id for ins in active} - losers
        cancelled = {
            ins.id for ins in active
            if getattr(ins, "depends_on", None) and ins.depends_on not in followed
        }
        if not cancelled:
            break
        void |= cancelled

    conflicts = [
        ConflictPair(
            instruction_a_id=a.id,
            instruction_b_id=b.id,
            conflict_type=_conflict_type(a, b, how),
            resolution_id=winner.id,
            explanation=_explain(winner, loser, how, scenario),
        )
        for a, b, winner, loser, how in decided
    ]
    overridden = void | losers
    return sorted(followed), sorted(overridden), conflicts


def resolution_methods(scenario: Scenario) -> Dict[frozenset, str]:
    """Unordered conflict pair -> the rule that decided it. For tests and analysis."""
    by_id = {ins.id: ins for ins in scenario.instructions}
    out = {}
    for c in resolve(scenario)[2]:
        a, b = by_id[c.instruction_a_id], by_id[c.instruction_b_id]
        out[frozenset((a.id, b.id))] = pick_winner(a, b, scenario)[1]
    return out


def matches_ground_truth(scenario: Scenario) -> Optional[str]:
    """None if the stored ground truth equals resolve(); otherwise a short description of the difference."""
    followed, overridden, conflicts = resolve(scenario)
    if sorted(scenario.ground_truth_followed) != followed:
        return f"followed differs: stored {sorted(scenario.ground_truth_followed)} vs resolved {followed}"
    if sorted(scenario.ground_truth_overridden) != overridden:
        return f"overridden differs: stored {sorted(scenario.ground_truth_overridden)} vs resolved {overridden}"
    stored = {frozenset((c.instruction_a_id, c.instruction_b_id)): c.resolution_id for c in scenario.conflicts}
    got = {frozenset((c.instruction_a_id, c.instruction_b_id)): c.resolution_id for c in conflicts}
    if stored != got:
        return f"conflicts differ: stored {stored} vs resolved {got}"
    return None
