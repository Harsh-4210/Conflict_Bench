"""
ConflictBench — multi-step ("long-horizon") environment, framework-free core.

One scenario's instructions arrive over several turns, like an inbox. The agent acts on
what it has seen with tool calls (apply / reject / revert / escalate / wait / finish) and
the environment keeps a small simulated company state. A later message can outrank an
instruction that was already applied, a status update can void a conditional one, and a
contingent follow-up dies when the instruction it hangs on is overridden, so the agent
must act without seeing the future and correct course.

Ground truth is ``resolver.resolve`` applied to the whole scenario. It is never in an
observation. The reward (multistep/scoring.py) is paid once at ``finish`` or when the step
budget runs out.

Gym-style API (no OpenEnv dependency; multistep/server.py wraps this class):

    env = MultiStepConflictEnv()
    obs = env.reset(seed=7)
    obs, reward, done, info = env.step({"tool": "apply", "id": "INS-..."})

Reference solvers (the oracle and the baselines in scripts/evaluate_multistep.py) use the
privileged helpers ``ground_truth()``, ``known_followed()`` and ``known_scenario()``. They
are not part of the observation and are not served over OpenEnv.
"""

import dataclasses
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from generator import GeneratorConfig, Instruction, Scenario, ScenarioGenerator, clamp_difficulty
from resolver import UnresolvableConflict, resolve

from .scoring import RewardBreakdown, final_reward
from .state import (
    CompanyState, Schedule, build_schedule, cell_of, fact_line, message_dict,
    standing_notices, target_cells,
)
from .tools import TOOLS, ActionError, ToolCall, parse_action

DEFAULT_BUDGET_FACTOR = 3
DEFAULT_MAX_PER_TURN = 3
DEFAULT_DIFFICULTY = 2

RULES_TEXT = """You are a business operations coordinator. Instructions from several people arrive over time, a few at a time. You keep the company's settings (the COMPANY STATE) correct by applying the instructions that should be carried out, and by undoing earlier ones when later information shows they should not be.

AUTHORITY:
- Legal, compliance and regulatory roles outrank everyone else, including the CEO.
- Otherwise seniority decides: C-suite (CEO, President, Chief ... Officer) > vice presidents (VP, SVP, EVP) > directors > managers > team leads and coordinators. Titles at the same level have equal authority.
- STANDING NOTICES can give one person the final say on one topic, or pass a senior person's authority on one topic to someone else. A notice applies only to the topic it names.
- Between instructions of equal authority, the later date wins. A later date never beats higher authority.

CONFLICTS:
- Two instructions conflict only if they give different directions on the same topic and their scopes overlap. Company-wide (or all systems) overlaps everything; two different departments or two different systems do not overlap. Instructions that agree never conflict.
- The instruction that loses a conflict must not be in the final state.
- An instruction that starts with "If" applies only when its condition is true. Status updates tell you when a condition is true or false; until then you do not know. If the condition is false the instruction is void.
- An instruction contingent on another one is cancelled when that other instruction is not carried out.

YOUR TURN: send ONE JSON tool call per step. Reverting an instruction that you already knew was outranked when you applied it costs points; reverting because a new message changed things does not. Invalid calls use up a step. The score is the company state when you call finish (or when the step budget runs out)."""


class MultiStepConflictEnv:
    """
    Multi-step ConflictBench. One instance holds one episode at a time.

    Args:
        config: GeneratorConfig for the scenarios (None = the default task).
        difficulty: default difficulty for reset() (1..MAX_DIFFICULTY).
        budget_factor: step budget = budget_factor * number of instructions.
        max_per_turn: at most this many messages arrive per turn.
        seed: seeds the stream of episode seeds used when reset() gets no seed.
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        difficulty: int = DEFAULT_DIFFICULTY,
        budget_factor: int = DEFAULT_BUDGET_FACTOR,
        max_per_turn: int = DEFAULT_MAX_PER_TURN,
        seed: Optional[int] = None,
    ):
        if budget_factor < 1:
            raise ValueError("budget_factor must be >= 1")
        if max_per_turn < 1:
            raise ValueError("max_per_turn must be >= 1")
        self.config = config
        self.difficulty = clamp_difficulty(difficulty)
        self.budget_factor = budget_factor
        self.max_per_turn = max_per_turn
        self._seed_rng = random.Random(seed)
        self._started = False
        self._done = False
        self.scenario: Optional[Scenario] = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: Optional[int] = None,
        scenario: Optional[Scenario] = None,
    ) -> Dict[str, Any]:
        """
        Start an episode and return the first observation.

        ``seed`` fixes the scenario and the arrival schedule (same seed, same difficulty,
        same config -> the same episode). ``scenario`` plays a ready-made scenario instead
        (the schedule is then seeded by ``seed``, or by the scenario ID).
        """
        if seed is None:
            seed = self._seed_rng.randrange(2 ** 31)
        if scenario is None:
            level = clamp_difficulty(difficulty if difficulty is not None else self.difficulty)
            scenario = ScenarioGenerator(seed=seed, config=self.config).generate(level, seed=seed)
        self.scenario = scenario
        self.episode_seed = seed
        self.schedule: Schedule = build_schedule(scenario, seed, self.max_per_turn)
        self._by_id: Dict[str, Instruction] = {ins.id: ins for ins in scenario.instructions}
        self._facts: Dict[str, bool] = dict(getattr(scenario, "facts", None) or {})

        self.budget = self.budget_factor * max(1, len(scenario.instructions))
        self.steps_used = 0
        self.turn = 0
        self._state = CompanyState()
        self._delivered: List[str] = []
        self._announced: Set[str] = set()
        self._status: Dict[str, str] = {}               # arrived id -> undecided|applied|rejected|escalated|reverted
        self._apply_was_foreseeable_loss: Dict[str, bool] = {}
        self.foreseeable_reverts = 0
        self.reverts_total = 0
        self.invalid_calls = 0
        self.unjustified_reverts = 0
        self._known_cache: Dict[int, Set[str]] = {}
        self._breakdown: Optional[RewardBreakdown] = None
        self._truth_followed, self._truth_overridden, _ = resolve(scenario)
        self._target = target_cells(scenario, self.schedule.order)
        self._started = True
        self._done = False

        new_msgs, status_lines = self._deliver(0)
        return self._observation(
            new_messages=new_msgs, status_updates=status_lines,
            feedback="Episode started. These are the first messages.", first=True,
        )

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Run one tool call. Returns (observation, reward, done, info). Reward is 0 until the episode ends."""
        if not self._started:
            return self._error_obs("No active episode: call reset before step."), 0.0, False, {}
        if self._done:
            obs = self._error_obs("Episode already finished: call reset to start a new one.", done=True)
            return obs, 0.0, True, {}

        self.steps_used += 1
        new_msgs: List[Dict[str, Any]] = []
        status_lines: List[str] = []
        error: Optional[str] = None
        feedback = ""
        finished = False

        try:
            call = parse_action(action)
            feedback, new_msgs, status_lines, finished = self._execute(call)
        except ActionError as exc:
            error = str(exc)
            self.invalid_calls += 1

        ended_by = None
        if finished:
            ended_by = "finish"
        elif self.steps_used >= self.budget:
            ended_by = "budget"
            feedback = (feedback + " " if feedback else "") + "Step budget used up: the episode ends."
        reward = 0.0
        info: Dict[str, Any] = {}
        if ended_by is not None:
            self._done = True
            self._breakdown = final_reward(
                actual=self._state.values(), target=self._target,
                foreseeable_reverts=self.foreseeable_reverts,
                wasted_steps=self.invalid_calls + self.unjustified_reverts,
                steps_used=self.steps_used, budget=self.budget, ended_by=ended_by,
            )
            reward = self._breakdown.reward
            info = {
                "reward_breakdown": self._breakdown.to_dict(),
                "ground_truth": self.ground_truth(),
                "stats": {
                    "n_instructions": len(self.scenario.instructions),
                    "turns": len(self.schedule.turns),
                    "turn_reached": self.turn,
                    "reverts": self.reverts_total,
                    "invalid_calls": self.invalid_calls,
                    "unjustified_reverts": self.unjustified_reverts,
                },
            }

        obs = self._observation(
            new_messages=new_msgs, status_updates=status_lines, feedback=feedback,
            error=error, reward=reward,
        )
        return obs, reward, self._done, info

    @property
    def state(self) -> Dict[str, Any]:
        """Bookkeeping about the current episode. No ground truth."""
        return {
            "scenario_id": self.scenario.scenario_id if self.scenario else None,
            "episode_seed": getattr(self, "episode_seed", None),
            "turn": getattr(self, "turn", 0),
            "steps_used": getattr(self, "steps_used", 0),
            "budget": getattr(self, "budget", 0),
            "done": self._done,
        }

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _arrived(self, instruction_id: str) -> Instruction:
        if instruction_id not in self._by_id or instruction_id not in self._status:
            raise ActionError(f"Unknown ID {instruction_id!r}: no message with that ID has arrived.")
        return self._by_id[instruction_id]

    def _execute(self, call: ToolCall) -> Tuple[str, List[Dict[str, Any]], List[str], bool]:
        """Run a parsed call. Returns (feedback, new messages, status lines, finished). Raises ActionError if invalid."""
        if call.tool == "apply":
            ins = self._arrived(call.id)
            if self._state.is_applied(ins.id):
                raise ActionError(f"{ins.id} is already applied.")
            self._apply_was_foreseeable_loss[ins.id] = ins.id not in self.known_followed()
            self._state.apply(ins)
            self._status[ins.id] = "applied"
            cell = cell_of(ins)
            return f"Applied {ins.id}: {cell[0]} ({cell[1]}) is now '{ins.action_value}'.", [], [], False

        if call.tool == "revert":
            ins = self._arrived(call.id)
            if not self._state.is_applied(ins.id):
                raise ActionError(f"{ins.id} is not applied, so there is nothing to revert.")
            if ins.id in self.known_followed():
                self.unjustified_reverts += 1       # still a winner on what is known: a wasted step
            elif self._apply_was_foreseeable_loss.get(ins.id, False):
                self.foreseeable_reverts += 1       # it was already outranked when applied
            self._state.revert(ins.id)
            self._status[ins.id] = "reverted"
            self.reverts_total += 1
            return f"Reverted {ins.id}.", [], [], False

        if call.tool == "reject":
            ins = self._arrived(call.id)
            if self._state.is_applied(ins.id):
                raise ActionError(f"{ins.id} is applied. Use revert to undo it.")
            if self._status[ins.id] == "rejected":
                raise ActionError(f"{ins.id} is already rejected.")
            self._status[ins.id] = "rejected"
            return f"Rejected {ins.id}.", [], [], False

        if call.tool == "escalate":
            for iid in call.ids:
                ins = self._arrived(iid)
                if self._state.is_applied(iid):
                    raise ActionError(f"{iid} is applied. Revert it before escalating.")
                if self._status[iid] == "escalated":
                    raise ActionError(f"{iid} is already escalated.")
            for iid in call.ids:
                self._status[iid] = "escalated"
            return f"Escalated {', '.join(call.ids)}.", [], [], False

        if call.tool == "wait":
            if self.turn >= len(self.schedule.turns) - 1:
                raise ActionError("No more messages will arrive. Call finish when you are done.")
            msgs, lines = self._deliver(self.turn + 1)
            return f"Turn {self.turn}: {len(msgs)} new message(s).", msgs, lines, False

        if call.tool == "finish":
            return "Finished.", [], [], True

        raise ActionError(f"Unknown tool {call.tool!r}.")  # pragma: no cover (parse_action checks)

    def _deliver(self, turn: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        self.turn = turn
        t = self.schedule.turns[turn]
        msgs = []
        for iid in t.message_ids:
            self._delivered.append(iid)
            self._status[iid] = "undecided"
            msgs.append(message_dict(self._by_id[iid], turn))
        lines = []
        for key in t.fact_keys:
            self._announced.add(key)
            lines.append(fact_line(key, self._facts[key]))
        return msgs, lines

    # ------------------------------------------------------------------
    # Privileged helpers for reference solvers (never in observations)
    # ------------------------------------------------------------------

    @property
    def arrived_ids(self) -> List[str]:
        """IDs of every message delivered so far, in arrival order."""
        return list(self._delivered)

    @property
    def current_message_ids(self) -> List[str]:
        """IDs delivered on the current turn."""
        return list(self.schedule.turns[self.turn].message_ids)

    @property
    def more_messages(self) -> bool:
        return self.turn < len(self.schedule.turns) - 1

    def applied_ids(self) -> List[str]:
        return self._state.applied_ids()

    def ground_truth(self) -> Dict[str, Any]:
        """The full-information answer: resolve() on the whole scenario, and the state it implies."""
        by_cell = sorted(self._target.items())
        return {
            "followed": list(self._truth_followed),
            "overridden": list(self._truth_overridden),
            "target_state": [
                {"topic": c[0], "scope": c[1], **({"task": c[2]} if c[2] else {}), "value": v}
                for c, v in by_cell
            ],
        }

    def known_scenario(self) -> Scenario:
        """
        The scenario as it looks right now: only delivered messages, and for facts that
        have not been announced yet, "true" is assumed (the agent is given the benefit of
        the doubt). A contingent instruction whose parent has not arrived is assumed to be
        fine. Stored ground-truth fields on the returned object are stale; use resolve().
        """
        sc = self.scenario
        seen = set(self._delivered)
        visible = []
        for iid in self._delivered:
            ins = self._by_id[iid]
            if getattr(ins, "depends_on", None) and ins.depends_on not in seen:
                ins = dataclasses.replace(ins, depends_on=None)
            visible.append(ins)
        facts = {k: (v if k in self._announced else True) for k, v in self._facts.items()}
        return dataclasses.replace(sc, instructions=visible, facts=facts)

    def known_followed(self) -> Set[str]:
        """IDs that resolve() would follow given only what has arrived so far (cached per turn)."""
        cached = self._known_cache.get(self.turn)
        if cached is None:
            try:
                cached = set(resolve(self.known_scenario())[0])
            except UnresolvableConflict:
                cached = set(self._delivered)   # cannot tell: nothing is known to be outranked
            self._known_cache[self.turn] = cached
        return cached

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _inbox(self) -> List[Dict[str, str]]:
        return [{"id": iid, "status": self._status[iid]} for iid in self._delivered]

    def _observation(
        self,
        new_messages: List[Dict[str, Any]],
        status_updates: List[str],
        feedback: str = "",
        error: Optional[str] = None,
        reward: float = 0.0,
        first: bool = False,
    ) -> Dict[str, Any]:
        sc = self.scenario
        obs: Dict[str, Any] = {
            "scenario_id": sc.scenario_id,
            "difficulty": sc.difficulty,
            "domain": sc.domain,
            "turn": self.turn,
            "more_messages": self.more_messages,
            "new_messages": new_messages,
            "status_updates": status_updates,
            "state": self._state.as_list(),
            "inbox": self._inbox(),
            "steps_used": self.steps_used,
            "steps_remaining": max(0, self.budget - self.steps_used),
            "budget": self.budget,
            "tools": TOOLS,
            "feedback": feedback,
            "error": error,
            "done": self._done,
            "reward": reward,
            "reward_breakdown": self._breakdown.to_dict() if self._done and self._breakdown else None,
        }
        obs["text"] = self._render(obs, first)
        return obs

    def _error_obs(self, message: str, done: bool = False) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id if self.scenario else None,
            "difficulty": self.scenario.difficulty if self.scenario else None,
            "domain": self.scenario.domain if self.scenario else None,
            "turn": getattr(self, "turn", 0), "more_messages": False,
            "new_messages": [], "status_updates": [], "state": [], "inbox": [],
            "steps_used": getattr(self, "steps_used", 0), "steps_remaining": 0,
            "budget": getattr(self, "budget", 0), "tools": TOOLS,
            "feedback": "", "error": message, "done": done, "reward": 0.0,
            "reward_breakdown": None, "text": message,
        }

    def _render(self, obs: Dict[str, Any], first: bool) -> str:
        lines: List[str] = []
        if first:
            lines += [RULES_TEXT, "", f"CONTEXT: {self.scenario.business_context}"]
            notices = standing_notices(self.scenario)
            if notices:
                lines += ["", "STANDING NOTICES:"] + [f"- {n}" for n in notices]
            lines += ["", "TOOLS (send one JSON object):"]
            for t in TOOLS:
                args = ", ".join(f'"{k}": <{v}>' for k, v in t["args"].items())
                example = '{"tool": "%s"%s}' % (t["tool"], (", " + args) if args else "")
                lines.append(f"- {example}: {t['does']}")
            lines.append("")
        lines.append(
            f"TURN {obs['turn']} | steps used {obs['steps_used']}/{obs['budget']} | "
            + ("more messages will arrive" if obs["more_messages"] else "no more messages will arrive")
        )
        if obs["feedback"]:
            lines.append(f"RESULT: {obs['feedback']}")
        if obs["error"]:
            lines.append(f"ERROR: {obs['error']}")
        if obs["status_updates"]:
            lines += ["", "STATUS UPDATES:"] + [f"- {s}" for s in obs["status_updates"]]
        if obs["new_messages"]:
            lines += ["", "NEW MESSAGES:"]
            for m in obs["new_messages"]:
                dated = f", {m['date']}" if m["date"] else ""
                lines.append(f"[{m['id']}] From {m['from']} ({m['title']}){dated}:\n  {m['text']}")
        lines += ["", "COMPANY STATE:"]
        if obs["state"]:
            for c in obs["state"]:
                task = f" [{c['task']}]" if c.get("task") else ""
                lines.append(f"- {c['topic']} ({c['scope']}){task} = {c['value']} (set by {c['set_by']})")
        else:
            lines.append("- (nothing applied yet)")
        if obs["inbox"]:
            lines += ["", "INBOX: " + ", ".join(f"{m['id']}={m['status']}" for m in obs["inbox"])]
        if obs["reward_breakdown"]:
            b = obs["reward_breakdown"]
            lines += ["", f"FINAL SCORE: {b['reward']:.3f} (state accuracy {b['accuracy']:.3f}, "
                          f"foreseeable reverts {b['foreseeable_reverts']}, wasted steps {b['wasted_steps']})"]
        return "\n".join(lines)
