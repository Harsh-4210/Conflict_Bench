# Multi-step ConflictBench

The single-step task shows the agent every instruction at once and asks for one answer. The
multi-step task (`multistep/`) plays the same scenarios as an inbox: instructions arrive a few
at a time, the agent acts on what it has seen, and a later message can overturn an earlier
decision. Ground truth is still `resolver.resolve(scenario)` and nothing else.

## The episode

1. `reset(seed)` generates a scenario (or plays one you pass in) and splits it into turns.
   Messages arrive in `effective_date` order, 1 to 3 per turn, with the split decided by the
   seed. Same seed, same difficulty, same generator config give the same episode.
2. Every status fact in `scenario.facts` ("the acquisition is still pending") is announced as a
   status update on one turn, drawn by the seed over all turns. A conditional instruction
   ("If the audit is still open, ...") can arrive before or after the fact that decides it.
3. Standing notices (final-say rules, delegations) and the authority rules are shown on the
   first observation and stay fixed.
4. The agent sends one JSON tool call per step. The episode ends at `finish` or when the
   step budget (3 x the number of instructions) runs out.

Each observation holds the new messages, status updates, the company state, an inbox of
what the agent did with each message, steps used and left, and the tool list. It never holds
ground truth. The final observation adds the reward breakdown. With `reveal_ground_truth=True`
(server option, off by default) the final observation also holds the correct answer.
`more_messages` tells the agent whether `wait` will still bring something.

### The company state

A dict of cells. A cell is `(topic, scope)`, for example `("hiring_status", "all")`, with the
value and the ID of the instruction that set it. Applying an instruction writes its value to
its cell. The newest applied instruction on a cell wins, and reverting it brings back the one
under it. A department-scoped instruction and a company-wide one on the same topic sit in
different cells, so overwriting does not hide the loser: it has to be reverted.

A contingent follow-up (`depends_on` set) gets a cell of its own, keyed by its ID, because it
is a separate task and not a setting. Without this, applying a cancelled follow-up would write
the same `("follow_up", "all")` cell as a live one and score the same.

### Where the long horizon comes from

* A later message from higher authority (or later in date at equal authority) outranks an
  instruction already applied.
* A status update can void a conditional instruction already applied.
* A contingent follow-up dies when the instruction it hangs on is overridden.

The agent cannot see the future, so it has to act on what it knows and correct course.

## Tools

| Call | Effect |
|---|---|
| `{"tool":"apply","id":"INS-..."}` | Writes the instruction's value to the state. |
| `{"tool":"reject","id":"INS-...","reason":"..."}` | Records a decision not to act. No state change. |
| `{"tool":"revert","id":"INS-..."}` | Removes an applied instruction; the cell falls back. |
| `{"tool":"escalate","ids":["INS-...",...]}` | Flags arrived instructions for a human. No state change. |
| `{"tool":"wait"}` | Delivers the next turn. Invalid when nothing is left. |
| `{"tool":"finish"}` | Ends the episode. |

An invalid call (bad JSON, unknown tool, unknown or not-yet-arrived ID, applying twice,
reverting something not applied, `wait` with an empty inbox, ...) returns an observation with
`error` set, changes nothing, and uses a step. `reject` and `escalate` are bookkeeping: the
reward looks only at the final state.

## Reward

Per-step reward is 0. The reward is paid once, at `finish` or when the budget runs out (the
step that uses up the budget still runs).

```
reward = clip( accuracy - revert_penalty - waste_penalty , 0 , 1 )
```

* **accuracy.** Compare the final state with the target state: apply every instruction
  `resolve()` says to follow, in arrival order. Over the union of both states' cells, a cell is
  correct when both have it with the same value. `accuracy = correct / all cells`. An extra
  cell (something applied that should not be) and a missing cell both count as wrong.
* **revert_penalty = min(0.03 x k, 0.15).** `k` counts reverts of an instruction that was
  already outranked or void *on the information available when it was applied*. That is what
  "the agent should have known" means here: `resolve()` run on the messages delivered so far.
* **waste_penalty = min(0.01 x w, 0.10).** `w` counts invalid calls plus reverts of an
  instruction that was still a winner on the information available at the time of the revert.
* A revert forced by something that arrived after the apply costs nothing.

"Available information" is `env.known_scenario()`: delivered messages only; a fact that has not
been announced is assumed true; a contingent instruction whose parent has not arrived is
assumed fine. The agent gets the benefit of the doubt, so it is never charged for something
that could not be known.

The constants live in `multistep/scoring.py`.

## Baselines

`scripts/evaluate_multistep.py` plays six scripted strategies on `data/test_v1.jsonl` (600
scenarios, episode seeds 0..599, budget factor 3, at most 3 messages per turn):

| Strategy | What it does |
|---|---|
| `oracle` | Sees the future. Applies exactly the final winners. |
| `cautious_resolver` | Each turn runs `resolve()` on what is known, applies winners, reverts anything that stopped being one. |
| `apply_then_fix` | Applies every message on arrival, then reverts what `resolve()` on the known messages says is outranked. |
| `apply_all_never_revert` | Applies every message that is not already known to lose. Never reverts. |
| `greedy_latest` | Applies every message the moment it arrives. Last writer wins. Never reverts. |
| `do_nothing` | Calls `finish` at once. |

Measured results (`results/multistep_<strategy>/summary.json`, 600 episodes each, mean reward
with a seeded 95% bootstrap CI over episodes):

| Strategy | Reward | 95% CI | D1 | D2 | D3 | Accuracy | Foreseeable reverts / ep | Reverts / ep | Steps / ep |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | 1.000 | 1.000-1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 25.5 |
| cautious_resolver | 1.000 | 1.000-1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 3.33 | 32.1 |
| apply_then_fix | 0.922 | 0.918-0.925 | 0.953 | 0.919 | 0.893 | 1.000 | 2.68 | 6.00 | 37.5 |
| apply_all_never_revert | 0.885 | 0.879-0.890 | 0.902 | 0.878 | 0.874 | 0.885 | 0.00 | 0.00 | 28.5 |
| greedy_latest | 0.762 | 0.755-0.769 | 0.780 | 0.750 | 0.756 | 0.762 | 0.00 | 0.00 | 31.2 |
| do_nothing | 0.000 | 0.000-0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | 1.0 |

Episodes average 20.6 instructions over 10.6 turns. `cautious_resolver` reverts at least once
in 94.8% of the episodes (3.33 reverts per episode), so the correction the task asks for really
happens, and no baseline hit the step budget.

**Why `cautious_resolver` scores 1.000.** It only applies what `resolve()` marks as a winner
on the information it has, so it never applies something already outranked. Every revert it
makes is forced by news that arrived later, and the rules say that costs nothing. Its
`accuracy` is 1.000 because after the last turn it knows everything. So the best agent
without foresight ties the oracle on reward; it only needs more steps (32.1 against 25.5).
The foresight penalty is aimed at agents that apply an instruction they can already see losing,
and `apply_then_fix` shows it: same final state as the oracle, 0.078 lower reward.

## Run it

```
python scripts/evaluate_multistep.py                              # every strategy on data/test_v1.jsonl
python scripts/evaluate_multistep.py --strategy oracle --n 50     # one strategy, first 50 scenarios
python scripts/evaluate_multistep.py --generate 30 --seed 5       # 30 freshly generated test-split scenarios
```

In Python:

```python
from multistep import MultiStepConflictEnv
env = MultiStepConflictEnv()
obs = env.reset(seed=7, difficulty=2)
obs, reward, done, info = env.step({"tool": "apply", "id": obs["new_messages"][0]["id"]})
```

Serve it on OpenEnv (own app, separate from `server/app.py`):

```
uvicorn multistep.server:app --port 8001
```

Plain HTTP `/reset` and `/step` are stateless in openenv-core 0.2.3 (each request builds a new
environment). Run full episodes over the `/ws` WebSocket, as `EnvClient` does;
`tests/test_multistep_server.py` does this end to end.

Tests: `python -m pytest -q tests/test_multistep_*.py`.

## Limits

* The scripted baselines and the scoring rule use the same `resolve()` on the known messages,
  so `cautious_resolver` is the best-case agent for this rule, not a measurement of any model.
  No language model has been run on this environment: not yet measured.
* Standing notices are not delivered over time; only messages and status facts are.
* `reject` and `escalate` have no effect on the reward. All conflicts in the generator have a
  resolvable winner, so there is nothing the agent must escalate.
* `multistep/` is not listed in `pyproject.toml`'s package map, and the `results/multistep_*`
  summaries use a prefixed `data_sha256` so `scripts/make_leaderboard.py` keeps them in their own
  group.
* A scenario whose full set of instructions has an unresolvable conflict (equal authority, no
  dates) makes `reset` raise `UnresolvableConflict`. The generator does not produce these.
