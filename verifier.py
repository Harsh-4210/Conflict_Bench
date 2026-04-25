"""
ConflictBench — Deterministic Verifier
Scores LLM output against ground truth. No LLM judge. Fully rule-based.
5 independent rubric functions, each returning a float in [0.0, 1.0].
"""

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from generator import Scenario


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    raw: str
    parsed_ok: bool
    identified_conflicts: List[Dict]   # list of {instruction_a, instruction_b, conflict_type, resolution, reasoning}
    execution_plan: List[str]          # list of instruction IDs
    overridden_instructions: List[str] # list of instruction IDs
    parse_error: Optional[str] = None


def parse_agent_output(raw: str) -> AgentOutput:
    """
    Extract and parse the JSON block from LLM output.
    Handles common LLM formatting issues (markdown fences, trailing text).
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try to extract first JSON object if there's trailing text
    brace_match = re.search(r"\{[\s\S]*\}", cleaned)
    if brace_match:
        cleaned = brace_match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return AgentOutput(
            raw=raw,
            parsed_ok=False,
            identified_conflicts=[],
            execution_plan=[],
            overridden_instructions=[],
            parse_error=str(e),
        )

    # Validate required top-level keys
    required_keys = {"identified_conflicts", "execution_plan", "overridden_instructions"}
    missing = required_keys - set(data.keys())
    if missing:
        return AgentOutput(
            raw=raw,
            parsed_ok=False,
            identified_conflicts=[],
            execution_plan=[],
            overridden_instructions=[],
            parse_error=f"Missing required keys: {missing}",
        )

    return AgentOutput(
        raw=raw,
        parsed_ok=True,
        identified_conflicts=data.get("identified_conflicts", []),
        execution_plan=[str(x) for x in data.get("execution_plan", [])],
        overridden_instructions=[str(x) for x in data.get("overridden_instructions", [])],
    )


# ---------------------------------------------------------------------------
# Rubric 1 — Correct final state
# Did the agent follow the right instructions and skip the right ones?
# Weight in composite: 0.35
# ---------------------------------------------------------------------------

def rubric_correct_final_state(output: AgentOutput, scenario: Scenario) -> float:
    """
    F1 score comparing agent's execution plan to ground truth followed instructions.
    Penalizes both false follows (following a loser) and false skips (skipping a winner).
    """
    if not output.parsed_ok:
        return 0.0

    gt_followed = set(scenario.ground_truth_followed)
    agent_followed = set(output.execution_plan)

    if not gt_followed and not agent_followed:
        return 1.0

    tp = len(gt_followed & agent_followed)
    fp = len(agent_followed - gt_followed)
    fn = len(gt_followed - agent_followed)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


# ---------------------------------------------------------------------------
# Rubric 2 — No contradictory actions in the execution plan
# Even if the agent gets some IDs wrong, its plan must be self-consistent.
# Weight: 0.25
# ---------------------------------------------------------------------------

def rubric_no_contradictions(output: AgentOutput, scenario: Scenario) -> float:
    """
    Check that no two instructions in the agent's execution plan
    share the same action_key (i.e. no conflicting actions co-exist in the plan).
    Score = 1 - (contradictions_found / max_possible_contradictions).
    """
    if not output.parsed_ok:
        return 0.0

    # Build a map from instruction ID to action_key + action_value
    id_to_instr = {ins.id: ins for ins in scenario.instructions}

    # Collect the action_keys present in the agent's plan
    seen_keys: Dict[str, str] = {}  # action_key -> first action_value seen
    contradictions = 0
    total_pairs_checked = 0

    for instr_id in output.execution_plan:
        if instr_id not in id_to_instr:
            continue
        instr = id_to_instr[instr_id]
        key = instr.action_key
        val = instr.action_value

        if key in seen_keys:
            total_pairs_checked += 1
            if seen_keys[key] != val:
                contradictions += 1
        else:
            seen_keys[key] = val

    if total_pairs_checked == 0:
        return 1.0  # no opportunity for contradiction

    return round(1.0 - (contradictions / total_pairs_checked), 4)


# ---------------------------------------------------------------------------
# Rubric 3 — Conflict identification accuracy
# Did the agent correctly identify which instruction pairs conflict?
# Weight: 0.20
# ---------------------------------------------------------------------------

def rubric_conflict_identification(output: AgentOutput, scenario: Scenario) -> float:
    """
    F1 score on conflict pair identification.
    A conflict is a frozenset of two instruction IDs — order doesn't matter.
    """
    if not output.parsed_ok:
        return 0.0

    gt_conflict_pairs = set(
        frozenset([c.instruction_a_id, c.instruction_b_id])
        for c in scenario.conflicts
    )

    agent_conflict_pairs = set()
    for c in output.identified_conflicts:
        if "instruction_a" in c and "instruction_b" in c:
            agent_conflict_pairs.add(frozenset([c["instruction_a"], c["instruction_b"]]))

    if not gt_conflict_pairs and not agent_conflict_pairs:
        return 1.0

    tp = len(gt_conflict_pairs & agent_conflict_pairs)
    fp = len(agent_conflict_pairs - gt_conflict_pairs)
    fn = len(gt_conflict_pairs - agent_conflict_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)

    # Bonus: correct resolution pick within identified conflicts
    resolution_bonus = 0.0
    correctly_resolved = 0
    agent_conflict_map = {}
    for c in output.identified_conflicts:
        if "instruction_a" in c and "instruction_b" in c and "resolution" in c:
            pair = frozenset([c["instruction_a"], c["instruction_b"]])
            agent_conflict_map[pair] = c["resolution"]

    gt_resolution_map = {
        frozenset([c.instruction_a_id, c.instruction_b_id]): c.resolution_id
        for c in scenario.conflicts
    }

    matched_pairs = gt_conflict_pairs & agent_conflict_pairs
    for pair in matched_pairs:
        if agent_conflict_map.get(pair) == gt_resolution_map.get(pair):
            correctly_resolved += 1

    if matched_pairs:
        resolution_bonus = 0.15 * (correctly_resolved / len(matched_pairs))

    return round(min(1.0, f1 + resolution_bonus), 4)


# ---------------------------------------------------------------------------
# Rubric 4 — Efficiency
# Fewer unnecessary instructions in the plan = better.
# An efficient plan contains exactly the needed instructions, nothing extra.
# Weight: 0.10
# ---------------------------------------------------------------------------

def rubric_efficiency(output: AgentOutput, scenario: Scenario) -> float:
    """
    Score based on plan compactness. Penalizes including unnecessary extra IDs.
    Optimal plan = exactly the ground truth followed set.
    """
    if not output.parsed_ok:
        return 0.0

    gt_followed = set(scenario.ground_truth_followed)
    agent_followed = set(output.execution_plan)

    extra = len(agent_followed - gt_followed)      # false follows (bloat)
    missed = len(gt_followed - agent_followed)      # false skips
    total_gt = len(gt_followed)

    if total_gt == 0:
        return 1.0

    penalty = (extra + missed) / (total_gt + extra)
    return round(max(0.0, 1.0 - penalty), 4)


# ---------------------------------------------------------------------------
# Rubric 5 — Format compliance
# Valid JSON with correct structure = full score.
# Partial structure gives partial credit.
# Weight: 0.10
# ---------------------------------------------------------------------------

def rubric_format_compliance(output: AgentOutput, scenario: Scenario) -> float:
    """
    Check JSON structure compliance. Awards partial credit for partial structure.
    """
    if not output.parsed_ok:
        return 0.0  # no JSON at all

    score = 0.4  # base: valid JSON

    # Check for required top-level keys
    if isinstance(output.identified_conflicts, list):
        score += 0.2
    if isinstance(output.execution_plan, list):
        score += 0.2
    if isinstance(output.overridden_instructions, list):
        score += 0.1

    # Check conflict entries have expected sub-keys
    if output.identified_conflicts:
        sample = output.identified_conflicts[0]
        required_subkeys = {"instruction_a", "instruction_b", "conflict_type", "resolution", "reasoning"}
        if isinstance(sample, dict) and required_subkeys.issubset(sample.keys()):
            score += 0.1

    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

RUBRIC_WEIGHTS = {
    "correct_final_state":    0.35,
    "no_contradictions":      0.25,
    "conflict_identification": 0.20,
    "efficiency":             0.10,
    "format_compliance":      0.10,
}


@dataclass
class ScoreBreakdown:
    correct_final_state: float
    no_contradictions: float
    conflict_identification: float
    efficiency: float
    format_compliance: float
    composite: float

    def to_dict(self) -> dict:
        return {
            "correct_final_state":     self.correct_final_state,
            "no_contradictions":       self.no_contradictions,
            "conflict_identification": self.conflict_identification,
            "efficiency":              self.efficiency,
            "format_compliance":       self.format_compliance,
            "composite":               self.composite,
        }


def score(raw_output: str, scenario: Scenario) -> ScoreBreakdown:
    """
    Full scoring pipeline. Parse output then run all 5 rubrics.
    Returns ScoreBreakdown with individual + composite score.
    """
    output = parse_agent_output(raw_output)

    r1 = rubric_correct_final_state(output, scenario)
    r2 = rubric_no_contradictions(output, scenario)
    r3 = rubric_conflict_identification(output, scenario)
    r4 = rubric_efficiency(output, scenario)
    r5 = rubric_format_compliance(output, scenario)

    composite = (
        RUBRIC_WEIGHTS["correct_final_state"]     * r1 +
        RUBRIC_WEIGHTS["no_contradictions"]        * r2 +
        RUBRIC_WEIGHTS["conflict_identification"]  * r3 +
        RUBRIC_WEIGHTS["efficiency"]               * r4 +
        RUBRIC_WEIGHTS["format_compliance"]        * r5
    )

    return ScoreBreakdown(
        correct_final_state=r1,
        no_contradictions=r2,
        conflict_identification=r3,
        efficiency=r4,
        format_compliance=r5,
        composite=round(composite, 4),
    )
