# ConflictBench — Reward Function Specification

## Design Philosophy

The reward function must satisfy three requirements simultaneously:

1. **Correctness** — it must reward genuinely better conflict resolution, not surface features
2. **Density** — it must provide a useful gradient signal at every training step, not just sparse successes
3. **Ungameability** — a model should not be able to score high on the rubrics without actually solving the task

All three requirements are met through the five-rubric composite design with F1 scoring and programmatic ground truth.

---

## Rubric Specifications

### R1: Correct Final State (weight = 0.35)

**What it measures:** Whether the model's execution plan correctly identifies which instructions should be followed and which should be overridden.

**How it is computed:**

```python
predicted_followed  = set(parse_followed_ids(completion))
predicted_overridden = set(parse_overridden_ids(completion))

precision_f = |predicted_followed ∩ true_followed| / |predicted_followed|
recall_f    = |predicted_followed ∩ true_followed| / |true_followed|
f1_followed = 2 * precision_f * recall_f / (precision_f + recall_f)

# Same for overridden
f1_overridden = ...

score_r1 = (f1_followed + f1_overridden) / 2
```

**Why F1 and not accuracy:** F1 gives partial credit for partially correct plans. A model that correctly identifies 4 of 6 followed instructions receives a meaningful reward, not zero. This dense signal is critical for GRPO to make progress in early training.

---

### R2: No Contradictions (weight = 0.25)

**What it measures:** Whether the execution plan contains two instructions that cannot coexist.

**How it is computed:**

```python
executed_actions = extract_action_key_values(completion)
contradiction_pairs = find_conflicting_actions(executed_actions)
score_r2 = 1.0 if len(contradiction_pairs) == 0 else 0.0
```

**Why binary:** Contradictions are binary failures. A plan that partially contradicts itself is still a failed plan — it cannot be executed. The high weight (0.25) ensures this is a strong training signal.

**Common failure mode caught:** A model that attempts to satisfy both sides of a hiring-freeze vs. hiring-expansion conflict — executing both "freeze hiring" and "proceed with hiring" — receives 0.0 on this rubric regardless of how well-formatted or detailed the explanation is.

---

### R3: Conflict Identification (weight = 0.20)

**What it measures:** Whether the model explicitly identifies the conflict pairs and resolves each in the correct direction.

**How it is computed:**

```python
predicted_pairs = parse_conflict_pairs(completion)  # [(id_a, id_b), ...]
true_pairs      = scenario.conflicts

f1_pairs = compute_f1(predicted_pairs, true_pairs)

# Direction accuracy: for each correctly identified pair,
# did the model name the correct winner?
direction_accuracy = mean([
    1.0 if predicted_winner(pair) == true_winner(pair) else 0.0
    for pair in correctly_identified_pairs
])

score_r3 = f1_pairs * direction_accuracy
```

**Why multiply F1 by direction accuracy:** A model that identifies all conflict pairs but always names the wrong winner is not learning the authority hierarchy — it is pattern-matching on conflict structure. Multiplying ensures both components must be correct for a high score.

---

### R4: Efficiency (weight = 0.10)

**What it measures:** Whether the execution plan includes unnecessary instructions that were not part of any conflict and should simply be executed as-is.

**How it is computed:**

```python
necessary_instructions = true_followed | true_overridden
unnecessary_included = predicted_plan - necessary_instructions
penalty = min(len(unnecessary_included) / total_instructions, 1.0)
score_r4 = 1.0 - penalty
```

**Why include this:** Without an efficiency penalty, a model can trivially improve its Final State score by including all instructions in the "followed" list. The efficiency rubric closes this loophole.

---

### R5: Format Compliance (weight = 0.10)

**What it measures:** Whether the completion is valid JSON with the required schema.

**Required schema:**
```json
{
  "followed": ["INS-XXXX", "INS-YYYY"],
  "overridden": ["INS-ZZZZ"],
  "conflicts": [
    {
      "instruction_a": "INS-XXXX",
      "instruction_b": "INS-ZZZZ",
      "winner": "INS-XXXX",
      "rationale": "Legal outranks VP Engineering"
    }
  ]
}
```

**How it is computed:**

```python
try:
    parsed = json.loads(extract_json(completion))
    has_followed  = isinstance(parsed.get("followed"), list)
    has_overridden = isinstance(parsed.get("overridden"), list)
    has_conflicts = isinstance(parsed.get("conflicts"), list)
    score_r5 = 1.0 if all([has_followed, has_overridden, has_conflicts]) else 0.5
except json.JSONDecodeError:
    score_r5 = 0.0
```

---

## Composite Score

```
Composite = 0.35 × R1 + 0.25 × R2 + 0.20 × R3 + 0.10 × R4 + 0.10 × R5
```

Range: [0.0, 1.0]

A model that achieves 1.0 on all rubrics has: produced a perfectly correct execution plan, included no contradictions, identified all conflict pairs and resolved each in the correct direction, included no unnecessary instructions, and output valid JSON.

---

## Observed Score Distributions

| Model State | Mean Composite | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|
| Qwen2.5-3B baseline (zero-shot) | 0.14 | 0.11 | 0.31 | 0.08 | 0.62 | 0.65 |
| After GRPO Run 2 (step 250) | 0.50 | 0.48 | 0.74 | 0.39 | 0.71 | 0.88 |

The largest gains are in R2 (no contradictions) and R3 (conflict identification), which suggests the model is learning the structural task — identify conflicts, pick a winner — before it learns the authority content — which source level wins.
