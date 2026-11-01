# ConflictBench — Reward Specification

This document describes the scorer as it is implemented in [`verifier.py`](../verifier.py). If the two ever disagree, the code is right and this file is out of date. `tests/test_docs.py` checks that the weight table below matches `verifier.RUBRIC_WEIGHTS`.

The scorer is deterministic and rule-based: no LLM judge. The correct answer for every scenario comes from [`resolver.py`](../resolver.py) and is stored in the scenario (`ground_truth_followed`, `ground_truth_overridden`, `conflicts`).

---

## 1. What the model must output

The system prompt ([`prompts.py`](../prompts.py)) and the end of every scenario prompt ask for one JSON object with exactly these keys:

```json
{
  "identified_conflicts": [
    {
      "instruction_a": "<ID>",
      "instruction_b": "<ID>",
      "conflict_type": "direct|resource|temporal|conditional",
      "resolution": "<winning ID>",
      "reasoning": "<one sentence>"
    }
  ],
  "execution_plan": ["<ID>", "..."],
  "overridden_instructions": ["<ID>", "..."]
}
```

- `execution_plan`: every instruction to carry out.
- `overridden_instructions`: every other instruction (losers of conflicts, void conditional instructions, cancelled contingent instructions).
- `identified_conflicts`: every pair that really conflicts, each with its winner in `resolution`.
- `conflict_type` and `reasoning` are **not scored**. They are kept for readability and analysis.

## 2. Parsing (never raises)

`parse_agent_output()` accepts a string, a TRL-style list of chat messages (it reads the last message), or anything else via `str()`.

1. **Finding the answer.** Fenced code blocks are tried first, last block first; then the whole text. Inside each region every position that starts a JSON object is decoded, and the **last** object that has all three keys wins (models often correct themselves at the end). An answer nested inside a wrapper object is still found.
2. **Normalising IDs.** Every ID is stripped and upper-cased, so `" ins-ab12cd "` matches `INS-AB12CD`.
3. **Normalising types.** For the two ID lists: `null` → `[]`; a single string or integer → a one-item list; a list keeps its string/integer items and drops the rest; anything else → `[]`. For `identified_conflicts`: only objects are kept; an entry whose `instruction_a`, `instruction_b` or `resolution` is not a string/integer is dropped. Each problem found here is recorded as a **schema error**.
4. `format_ok` (`parsed_ok`) is true when an object with all three keys was found.

## 3. The composite

<!-- rubric-weights:start -->
| Rubric | Key in `RUBRIC_WEIGHTS` | Weight |
|---|---|---:|
| Correct final state | `correct_final_state` | 0.45 |
| Conflict identification | `conflict_identification` | 0.30 |
| No contradictions | `no_contradictions` | 0.25 |
<!-- rubric-weights:end -->

```
weighted  = 0.45 × correct_final_state + 0.30 × conflict_identification + 0.25 × no_contradictions
composite = clamp(weighted × format_gate, 0, 1), rounded to 4 decimals

format_gate = 0.0   no JSON object with all three keys was found  (every rubric is 0 as well)
            = 0.9   an object was found but had schema errors      (SCHEMA_ERROR_FACTOR)
            = 1.0   clean
```

`RUBRIC_WEIGHTS` in `verifier.py` is the only copy of the weights. The training pipeline imports it; nothing else defines its own weights.

## 4. The rubrics

Notation: `P` = set of IDs in `execution_plan`, `O` = set of IDs in `overridden_instructions`, `F*` / `O*` = the correct followed / overridden sets, `n` = number of instructions in the scenario. `F1(A, B) = 2|A ∩ B| / (|A| + |B|)`, with `F1(∅, ∅) = 1`.

### R1 — correct_final_state (0.45)

```
H        = (P ∪ O) − {IDs that exist in the scenario}        # made-up IDs
followed   = F1(P ∪ H, F*)
overridden = F1(O ∪ H, O*)
split      = max(0, 1 − |P ∩ O| / n)
R1 = (followed + overridden) / 2 × split
```

**Why it exists:** this is the end result: what actually gets done. Scoring both lists means "follow everything" and "override everything" each lose about half of this rubric. A made-up ID costs points in both lists, and putting an ID in both lists (hedging) shrinks the whole rubric.

### R3 — conflict_identification (0.30)

```
G = set of correct conflict pairs (unordered)
A = set of pairs the model listed (unordered; a pair listed twice counts once)
if G and A are both empty: R3 = 1
matched = G ∩ A;  if matched is empty: R3 = 0
correct = pairs in matched where EVERY listing of that pair names the correct winner
R3 = F1(A, G) × |correct| / |matched|
```

**Why it exists:** R1 can be partly right by luck. R3 checks that the model found the actual clashes and resolved each one the right way. Multiplying (instead of adding) pair F1 and resolution accuracy means finding every clash but always naming the wrong winner scores 0. Listing a pair twice with both winners does not count as right.

### R2 — no_contradictions (0.25)

```
for each correct conflict pair {a, b}:
    credit 1 if exactly one of a, b is in P
              AND the model listed the pair {a, b}
              AND every resolution it stated for that pair is that same ID
    else 0
R2 = mean credit over the correct conflicts
(a scenario with no conflicts: R2 = 1 if P is non-empty, else 0)
```

**Why it exists:** a usable plan makes one explicit, consistent choice on every clash. Carrying out both sides is a contradiction; carrying out neither leaves the clash unhandled; saying "A wins" and then executing B is a contradiction too. R2 rewards consistency whether or not the choice is right (R1 and R3 check rightness), and an empty plan gets nothing.

## 5. Logged but not in the composite

`ScoreBreakdown` also reports two numbers that are **not** weighted:

- `efficiency = 1 − (extra + missed) / (|F*| + extra)`, where `extra = |P − F*|` and `missed = |F* − P|`. This is the Jaccard overlap of the plan with `F*`.
- `format_compliance` = the format-gate value (0, 0.9 or 1).

**Why efficiency was removed.** For any single answer, F1 = 2J / (1 + J), where J is the Jaccard overlap, so efficiency is just another monotone function of the followed-F1 that R1 already contains. Weighting it counted the same thing twice. It also showed that some per-rubric numbers reported for the hackathon run are impossible (`FIX_BLUEPRINT.md`, C.2): a mean F1 can never be below the mean Jaccard of the same answers.

**Why format is a gate.** In the first version, any JSON with the three keys earned 10% of the reward, which is part of why an empty answer scored 0.34. Format is a requirement, not an achievement: without a parseable answer nothing else can score.

## 6. Reward hacks: before and after

Mean composite over 600 scenarios (200 per level 1–3). No model is involved.

| Strategy | Old task, old reward | Old task, new reward | New task (test_v1), new reward | Limit in tests |
|---|---:|---:|---:|---|
| `empty` | 0.340 | 0.000 | 0.000 | ≤ 0.05 |
| `follow_all` | 0.429 | 0.177 | 0.189 | ≤ 0.35 |
| `override_all` | 0.340 | 0.116 | 0.097 | ≤ 0.25 |
| `random_single` | 0.410 | 0.037 | 0.024 | none |
| `random_half` | 0.411 | 0.222 | 0.207 | ≤ 0.35 |
| `hallucinated_ids` | 0.340 | 0.000 | 0.000 | ≤ 0.05 |
| `keyword_restrictive` (old template matching) | 0.951 | 0.921 | 0.189 | ≤ 0.35 (test split) |
| `oracle_lookup_v1` (old template matching + tier) | 1.000 | 1.000 | 0.189 | ≤ 0.6 (test split) |
| `keyword_restrictive_v2` (hidden fields + strict side) | — | — | 0.669 | ≤ 0.75 (test split) |
| `latest_wins` (hidden fields + later date) | — | — | 0.730 | ≤ 0.8 (test split) |
| `oracle_lookup` (hidden fields + seniority) | — | — | 0.808 | ≤ 0.9 (test split) |
| `gt_one_conflict_flipped` | — | 0.805 | 0.841 | level 1: 0.4–0.85 |
| `ground_truth` | 1.000 | 1.000 | 1.000 | = 1.0 |

Sources: `results/baselines.json` (legacy generator + the old verifier; its lookup row was then called `oracle_lookup`), `results/baselines_v2.json` (legacy generator + this verifier), and the `data/test_v1.jsonl` table of [`results/leaderboard.md`](../results/leaderboard.md). The limits are asserted in `tests/test_reward_hacks.py` (generated scenarios) and `tests/test_difficulty.py` (test split). `tests/test_reward_hacks.py` also checks monotonicity: over 500 generated scenarios, starting from the correct answer, each extra random error never raises the score, and the first error always lowers it.

Reading the table:

- The **new reward** alone removed the free points: empty, made-up and "all or nothing" answers fell to near zero or below 0.25 on the old task. It could not fix the keyword rule (0.921) or the lookup (1.000), because on the old task those shortcuts really did give the right answer.
- The **new task** removed those shortcuts: winners are balanced, the hierarchy is hidden, and standing rules, delegations, dates, conditions and chains change who wins. The `*_v2`, `latest_wins` and `oracle_lookup` rows are handed the hidden topic and scope fields that a model never sees, and still fall well short of 1.0.

## 7. What changed from the hackathon version

The previous version of this document described formulas and a JSON schema (`followed` / `overridden` / `conflicts`) that the code never used. The code itself changed as follows (`git log`, commit 896827c):

| | Hackathon version | Now |
|---|---|---|
| Weights | 0.35 final state, 0.25 contradictions, 0.20 identification, 0.10 efficiency, 0.10 format | 0.45 / 0.25 / 0.30, efficiency and format not weighted |
| Format | a weighted rubric (10%); any JSON with the three keys scored at least 0.9 on it | gate: 0, 0.9 or 1 |
| No contradictions | an empty plan scored 1.0 | needs exactly one side in the plan and a matching stated resolution, per real conflict |
| Final state | F1 of the plan only; the overridden list was never scored | mean of plan F1 and overridden F1, with penalties for made-up and double-listed IDs |
| Identification | 60% pair F1 + 40% resolution accuracy | pair F1 × resolution accuracy |
| Parser | first code fence, then a greedy `{...}`; crashed on `null`, numbers and nested IDs | last valid object; never raises |
