"""
ConflictBench — Deterministic Verifier
Scores LLM output against ground truth. No LLM judge. Fully rule-based.

Scoring (Phase 2, see FIX_BLUEPRINT.md P2-1..P2-6):
  composite = 0.45 x correct_final_state
            + 0.30 x conflict_identification
            + 0.25 x no_contradictions
  then the format gate: no parseable answer -> 0; schema errors -> x 0.9.

`efficiency` and `format_compliance` are still computed and reported, but only
for logging. They are not part of the composite.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from generator import Scenario


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

REQUIRED_KEYS = ("identified_conflicts", "execution_plan", "overridden_instructions")

# ID-like fields inside one identified_conflicts entry.
_CONFLICT_ID_FIELDS = ("instruction_a", "instruction_b", "resolution")

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n?([\s\S]*?)```")
_DECODER = json.JSONDecoder()


@dataclass
class AgentOutput:
    raw: str
    parsed_ok: bool
    identified_conflicts: List[Dict]   # list of {instruction_a, instruction_b, conflict_type, resolution, reasoning}
    execution_plan: List[str]          # list of instruction IDs (normalised: stripped, upper case)
    overridden_instructions: List[str] # list of instruction IDs (normalised: stripped, upper case)
    parse_error: Optional[str] = None
    schema_errors: List[str] = field(default_factory=list)  # type problems found while normalising


def _norm_id(value: Any) -> str:
    """Canonical form of an instruction ID: stripped, upper case."""
    return str(value).strip().upper()


def _is_id_scalar(value: Any) -> bool:
    # bool is a subclass of int, but True/False are never valid IDs.
    return isinstance(value, (str, int)) and not isinstance(value, bool)


def _coerce_text(raw: Any) -> str:
    """Turn whatever the caller passed (str, TRL message list, other) into text."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        if not raw:
            return ""
        last = raw[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # OpenAI-style content parts
                return "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            return str(content)
        return last if isinstance(last, str) else str(last)
    return str(raw)


def _scan_objects(text: str) -> List[dict]:
    """
    Every JSON object that can be decoded starting at some '{' in `text`, in order.
    An object holding all required keys is taken whole (its insides are skipped);
    any other object is kept but scanning continues inside it, so an answer nested
    in a wrapper object is still found.
    """
    found: List[dict] = []
    pos = text.find("{")
    while pos != -1:
        try:
            obj, end = _DECODER.raw_decode(text, pos)
        except (ValueError, RecursionError):
            pos = text.find("{", pos + 1)
            continue
        if isinstance(obj, dict):
            found.append(obj)
            span = text[pos:end]
            # Skip the object's insides if it is the answer, or if no answer can be
            # nested inside it (keeps the scan linear on large non-answer objects).
            if all(k in obj for k in REQUIRED_KEYS) or not all(f'"{k}"' in span for k in REQUIRED_KEYS):
                pos = text.find("{", end)
                continue
        pos = text.find("{", pos + 1)
    return found


def _find_answer(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Pick the answer object. Order: fenced blocks, last block first; then the whole
    text. Within a region the LAST object with all required keys wins (models often
    correct themselves at the end). Returns (object, None) or (None, error).
    """
    regions = [m.group(1) for m in _FENCE_RE.finditer(text)][::-1]
    regions.append(text)

    any_object = False
    for region in regions:
        objects = _scan_objects(region)
        any_object = any_object or bool(objects)
        for obj in reversed(objects):
            if all(k in obj for k in REQUIRED_KEYS):
                return obj, None

    if any_object:
        return None, f"No JSON object has all required keys: {list(REQUIRED_KEYS)}"
    return None, "No JSON object found"


def _as_id_list(value: Any, field_name: str, errors: List[str]) -> List[str]:
    """None -> []; str -> [str]; list -> its str/int items; anything else -> [] + error."""
    if value is None:
        return []
    if _is_id_scalar(value):
        ident = _norm_id(value)
        return [ident] if ident else []
    if isinstance(value, list):
        out: List[str] = []
        for i, item in enumerate(value):
            if _is_id_scalar(item):
                ident = _norm_id(item)
                if ident:
                    out.append(ident)
                else:
                    errors.append(f"{field_name}[{i}]: empty ID")
            else:
                errors.append(f"{field_name}[{i}]: expected an ID string, got {type(item).__name__}")
        return out
    errors.append(f"{field_name}: expected a list, got {type(value).__name__}")
    return []


def _as_conflict_list(value: Any, errors: List[str]) -> List[Dict]:
    """Keep only dict entries whose ID fields are scalars; normalise those IDs."""
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"identified_conflicts: expected a list, got {type(value).__name__}")
        return []

    out: List[Dict] = []
    for i, entry in enumerate(value):
        where = f"identified_conflicts[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{where}: expected an object, got {type(entry).__name__}")
            continue
        clean = dict(entry)
        bad = False
        for key in _CONFLICT_ID_FIELDS:
            if key not in entry:
                errors.append(f"{where}: missing '{key}'")
            elif _is_id_scalar(entry[key]):
                clean[key] = _norm_id(entry[key])
            else:
                errors.append(f"{where}.{key}: expected an ID string, got {type(entry[key]).__name__}")
                bad = True
        if bad:
            continue
        for key in ("conflict_type", "reasoning"):
            if key in clean and not isinstance(clean[key], str):
                clean[key] = str(clean[key])
        out.append(clean)
    return out


def parse_agent_output(raw: Any) -> AgentOutput:
    """
    Extract and parse the JSON answer from LLM output. Never raises.
    Accepts a str, a TRL list of message dicts, or anything else (via str()).
    """
    try:
        text = _coerce_text(raw)
    except Exception as e:  # a hostile __str__ must not crash the scorer
        return AgentOutput("", False, [], [], [], parse_error=f"Unreadable input: {e!r}")

    data, error = _find_answer(text)
    if data is None:
        return AgentOutput(
            raw=text,
            parsed_ok=False,
            identified_conflicts=[],
            execution_plan=[],
            overridden_instructions=[],
            parse_error=error,
        )

    errors: List[str] = []
    return AgentOutput(
        raw=text,
        parsed_ok=True,
        identified_conflicts=_as_conflict_list(data["identified_conflicts"], errors),
        execution_plan=_as_id_list(data["execution_plan"], "execution_plan", errors),
        overridden_instructions=_as_id_list(
            data["overridden_instructions"], "overridden_instructions", errors
        ),
        schema_errors=errors,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _f1(predicted: Set[str], truth: Set[str]) -> float:
    """Set F1. Both empty -> 1.0 (nothing to find, nothing claimed)."""
    if not predicted and not truth:
        return 1.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    return 2 * tp / (len(predicted) + len(truth))


def _ids(values: Iterable[str]) -> Set[str]:
    return {_norm_id(v) for v in values}


# ---------------------------------------------------------------------------
# Rubric 1 — Correct final state
# Did the agent follow the right instructions AND override the right ones?
# Weight in composite: 0.45
# ---------------------------------------------------------------------------

def rubric_correct_final_state(output: AgentOutput, scenario: Scenario) -> float:
    """
    Mean of followed-F1 (plan vs GT followed) and overridden-F1 (overridden list vs
    GT overridden).
      - An ID that is not in the scenario (made up) counts as a false positive in
        BOTH lists, wherever it appears.
      - An ID in both lists multiplies the result by (1 - overlap / n_instructions).
    """
    if not output.parsed_ok:
        return 0.0

    valid = _ids(ins.id for ins in scenario.instructions)
    plan = set(output.execution_plan)
    overridden = set(output.overridden_instructions)

    hallucinated = (plan | overridden) - valid
    followed_f1 = _f1(plan | hallucinated, _ids(scenario.ground_truth_followed))
    overridden_f1 = _f1(overridden | hallucinated, _ids(scenario.ground_truth_overridden))

    overlap = len(plan & overridden)
    n = max(1, len(scenario.instructions))
    split_factor = max(0.0, 1.0 - overlap / n)

    return round((followed_f1 + overridden_f1) / 2 * split_factor, 4)


# ---------------------------------------------------------------------------
# Rubric 2 — No contradictions
# On every real clash the plan must make exactly one choice.
# Weight: 0.25
# ---------------------------------------------------------------------------

def _stated_resolutions(output: AgentOutput) -> Dict[frozenset, Set[str]]:
    """Unordered pair -> every resolution the agent stated for it."""
    stated: Dict[frozenset, Set[str]] = {}
    for c in output.identified_conflicts:
        if "instruction_a" not in c or "instruction_b" not in c:
            continue
        pair = frozenset((c["instruction_a"], c["instruction_b"]))
        stated.setdefault(pair, set()).add(c.get("resolution", ""))
    return stated


def rubric_no_contradictions(output: AgentOutput, scenario: Scenario) -> float:
    """
    Is the answer self-consistent on every real clash? For each ground-truth
    conflict pair, credit 1 only if
      - exactly one of its two IDs is in the execution plan (both = contradiction,
        neither = clash not handled), AND
      - the agent listed that pair, and the resolution it stated is the side that
        is in the plan (saying "A wins" and then executing B is a contradiction).
    Score = mean over GT conflicts; an empty plan scores 0. This rewards a
    consistent, explicit choice, right or wrong; correctness is R1 and R3.
    """
    if not output.parsed_ok:
        return 0.0

    plan = set(output.execution_plan)
    if not scenario.conflicts:
        return 1.0 if plan else 0.0

    stated = _stated_resolutions(output)
    consistent = 0
    for c in scenario.conflicts:
        pair = frozenset((_norm_id(c.instruction_a_id), _norm_id(c.instruction_b_id)))
        chosen = plan & pair
        if len(chosen) == 1 and stated.get(pair) == chosen:
            consistent += 1
    return round(consistent / len(scenario.conflicts), 4)


# ---------------------------------------------------------------------------
# Rubric 3 — Conflict identification x resolution accuracy
# Weight: 0.30
# ---------------------------------------------------------------------------

def rubric_conflict_identification(output: AgentOutput, scenario: Scenario) -> float:
    """
    pair F1 x resolution accuracy.
      - Pairs are unordered; a pair listed twice counts once.
      - Resolution accuracy is over the correctly identified pairs. A pair counts as
        correctly resolved only if every listing of it names the GT winner (listing
        a pair twice with both winners does not hedge).
    """
    if not output.parsed_ok:
        return 0.0

    gt_resolution = {
        frozenset((_norm_id(c.instruction_a_id), _norm_id(c.instruction_b_id))): _norm_id(c.resolution_id)
        for c in scenario.conflicts
    }

    agent_resolutions = _stated_resolutions(output)

    gt_pairs = set(gt_resolution)
    agent_pairs = set(agent_resolutions)
    if not gt_pairs and not agent_pairs:
        return 1.0

    pair_f1 = _f1(agent_pairs, gt_pairs)
    matched = gt_pairs & agent_pairs
    if not matched:
        return 0.0

    correct = sum(1 for p in matched if agent_resolutions[p] == {gt_resolution[p]})
    return round(pair_f1 * correct / len(matched), 4)


# ---------------------------------------------------------------------------
# Rubric 4 — Efficiency (logging only, NOT in the composite)
# It is the Jaccard overlap of the plan with GT followed, i.e. a monotone function
# of followed-F1, so weighting it would count R1 twice (FIX_BLUEPRINT.md C.2).
# ---------------------------------------------------------------------------

def rubric_efficiency(output: AgentOutput, scenario: Scenario) -> float:
    """Jaccard overlap of the plan with the ground-truth followed set. Logging only."""
    if not output.parsed_ok:
        return 0.0

    gt_followed = _ids(scenario.ground_truth_followed)
    agent_followed = set(output.execution_plan)

    extra = len(agent_followed - gt_followed)      # false follows (bloat)
    missed = len(gt_followed - agent_followed)      # false skips
    total_gt = len(gt_followed)

    if total_gt == 0:
        return 1.0

    penalty = (extra + missed) / (total_gt + extra)
    return round(max(0.0, 1.0 - penalty), 4)


# ---------------------------------------------------------------------------
# Rubric 5 — Format (a gate, NOT a weighted rubric)
# ---------------------------------------------------------------------------

SCHEMA_ERROR_FACTOR = 0.9


def rubric_format_compliance(output: AgentOutput, scenario: Scenario) -> float:
    """
    The format gate's multiplier: 0.0 if no answer object was found, 0.9 if it was
    found but had schema errors, 1.0 if clean. Logged; applied as a multiplier.
    """
    if not output.parsed_ok:
        return 0.0
    return SCHEMA_ERROR_FACTOR if output.schema_errors else 1.0


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

# The ONLY copy of the weights. Everything else must import this.
RUBRIC_WEIGHTS = {
    "correct_final_state":     0.45,
    "conflict_identification": 0.30,
    "no_contradictions":       0.25,
}


@dataclass
class ScoreBreakdown:
    correct_final_state: float
    no_contradictions: float
    conflict_identification: float
    efficiency: float          # logging only
    format_compliance: float   # logging only: the format-gate multiplier
    composite: float
    format_ok: bool = False    # an answer object with all required keys was found
    schema_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "correct_final_state":     self.correct_final_state,
            "no_contradictions":       self.no_contradictions,
            "conflict_identification": self.conflict_identification,
            "efficiency":              self.efficiency,
            "format_compliance":       self.format_compliance,
            "composite":               self.composite,
            "format_ok":               self.format_ok,
        }


def score(raw_output: Any, scenario: Scenario) -> ScoreBreakdown:
    """
    Full scoring pipeline. Parse output, run the rubrics, combine them with
    RUBRIC_WEIGHTS, then apply the format gate. Never raises on agent output.
    """
    output = parse_agent_output(raw_output)

    parts = {
        "correct_final_state":     rubric_correct_final_state(output, scenario),
        "no_contradictions":       rubric_no_contradictions(output, scenario),
        "conflict_identification": rubric_conflict_identification(output, scenario),
    }
    gate = rubric_format_compliance(output, scenario)

    composite = sum(weight * parts[name] for name, weight in RUBRIC_WEIGHTS.items()) * gate
    composite = min(1.0, max(0.0, composite))

    return ScoreBreakdown(
        correct_final_state=parts["correct_final_state"],
        no_contradictions=parts["no_contradictions"],
        conflict_identification=parts["conflict_identification"],
        efficiency=rubric_efficiency(output, scenario),
        format_compliance=gate,
        composite=round(composite, 4),
        format_ok=output.parsed_ok,
        schema_errors=list(output.schema_errors),
    )
