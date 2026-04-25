"""
ConflictBench — Scenario Generator
Dynamically generates instruction conflict scenarios for LLM training.
No hardcoded scenarios. Every episode is freshly generated from templates.
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from copy import deepcopy


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    id: str
    text: str
    source: str          # e.g. "Legal & Compliance", "CEO Office"
    source_priority: int  # 1 = highest authority (Legal), 5 = lowest
    instruction_type: str  # "absolute" | "conditional" | "time_ordered" | "resource"
    action_key: str      # semantic conflict key, e.g. "hiring_status"
    action_value: str    # the directive, e.g. "frozen" | "active"
    condition: Optional[str] = None  # set only for conditional instructions


@dataclass
class ConflictPair:
    instruction_a_id: str
    instruction_b_id: str
    conflict_type: str
    resolution_id: str   # the ID of the instruction that wins (higher priority)
    explanation: str


@dataclass
class Scenario:
    scenario_id: str
    domain: str
    difficulty: int       # 1 = 2 conflicts | 2 = 4 conflicts | 3 = 6 conflicts
    business_context: str
    instructions: List[Instruction]
    conflicts: List[ConflictPair]
    ground_truth_followed: List[str]   # instruction IDs agent SHOULD follow
    ground_truth_overridden: List[str]  # instruction IDs agent SHOULD override
    prompt: str           # fully formatted LLM prompt


# ---------------------------------------------------------------------------
# Authority hierarchy — deterministic resolution basis
# ---------------------------------------------------------------------------

SOURCE_PRIORITY = {
    "Legal & Compliance":      1,
    "Regulatory Affairs":      1,
    "CEO Office":              2,
    "Chief Financial Officer":  2,
    "Chief Technology Officer":  2,
    "Chief Operating Officer":  2,
    "VP Engineering":          3,
    "VP Finance":              3,
    "VP Operations":           3,
    "VP Human Resources":      3,
    "Director of IT":          4,
    "Director of Finance":     4,
    "Engineering Manager":     5,
    "Finance Manager":         5,
    "HR Manager":              5,
    "IT Manager":              5,
    "Team Lead":               6,
    "Department Coordinator":  6,
}

# ---------------------------------------------------------------------------
# Template library — each group shares an action_key.
# Conflict is injected by picking two templates from the same group
# with different action_values and different source priorities.
# ---------------------------------------------------------------------------

TEMPLATE_GROUPS = {
    "hiring_status": {
        "description": "controls whether the company can hire",
        "templates": [
            {
                "action_value": "frozen",
                "sources": ["Legal & Compliance", "CEO Office", "Chief Financial Officer"],
                "texts": [
                    "Due to regulatory constraints, all hiring across {department} is immediately frozen until further notice.",
                    "Effective immediately: hiring is suspended company-wide pending completion of the financial audit.",
                    "All open headcount requests are paused as part of cost containment measures approved by the executive committee.",
                ],
                "type": "absolute",
            },
            {
                "action_value": "active",
                "sources": ["VP Engineering", "VP Human Resources", "Engineering Manager", "HR Manager"],
                "texts": [
                    "Approved headcount expansion: proceed with hiring {n} {role} engineers for the {team} team this quarter.",
                    "Q{quarter} hiring plan approved — begin recruiting for {n} open {role} positions immediately.",
                    "HR is authorized to extend offers to shortlisted candidates for the {role} role in {department}.",
                ],
                "type": "absolute",
            },
        ],
    },

    "deployment_approval": {
        "description": "controls whether production deployment is permitted",
        "templates": [
            {
                "action_value": "blocked",
                "sources": ["Legal & Compliance", "Director of IT", "Chief Technology Officer"],
                "texts": [
                    "No production deployments are permitted until the pending security audit is fully resolved.",
                    "All deployment windows are suspended pending sign-off from the compliance team.",
                    "Production environment is locked for the next {days} days due to the upcoming SOC 2 review.",
                ],
                "type": "absolute",
            },
            {
                "action_value": "approved",
                "sources": ["VP Engineering", "Engineering Manager", "Team Lead"],
                "texts": [
                    "Deployment window for {system} release {version} is approved for {deadline}.",
                    "Engineering team is cleared to push the {system} update to production this {timeframe}.",
                    "Release sign-off received — proceed with rolling out {system} to all environments.",
                ],
                "type": "absolute",
            },
        ],
    },

    "budget_approval_threshold": {
        "description": "sets the maximum spend before additional approval is needed",
        "templates": [
            {
                "action_value": "strict_limit",
                "sources": ["Chief Financial Officer", "Legal & Compliance", "VP Finance"],
                "texts": [
                    "Effective this quarter, all expenditures above ${low_amount} require CFO approval before commitment.",
                    "Cost controls in effect: no purchase orders above ${low_amount} may be issued without executive sign-off.",
                    "Budget freeze: department spending limited to ${low_amount} per line item until end-of-quarter review.",
                ],
                "type": "resource",
            },
            {
                "action_value": "standard_limit",
                "sources": ["Director of Finance", "Finance Manager", "VP Operations"],
                "texts": [
                    "Standard procurement policy: managers may approve expenses up to ${high_amount} without additional review.",
                    "Department heads retain discretionary budget authority up to ${high_amount} per transaction this quarter.",
                    "Approved budget allocation for {department}: up to ${high_amount} available for operational expenditures.",
                ],
                "type": "resource",
            },
        ],
    },

    "remote_work_policy": {
        "description": "governs where employees are permitted to work",
        "templates": [
            {
                "action_value": "mandatory_office",
                "sources": ["CEO Office", "Chief Operating Officer", "VP Human Resources"],
                "texts": [
                    "All employees in {department} are required to be on-site five days per week starting {deadline}.",
                    "Return-to-office mandate: remote work privileges are revoked for {team} effective {deadline}.",
                    "Executive directive: {department} must maintain full in-office attendance during the product launch phase.",
                ],
                "type": "absolute",
            },
            {
                "action_value": "flexible",
                "sources": ["HR Manager", "Engineering Manager", "Team Lead"],
                "texts": [
                    "The {team} team is authorized to follow a flexible hybrid schedule of {n} days in-office per week.",
                    "Approved: {department} may continue operating on a remote-first basis through end of {timeframe}.",
                    "Team leads may set individual remote arrangements for their direct reports as they see fit.",
                ],
                "type": "absolute",
            },
        ],
    },

    "overtime_authorization": {
        "description": "controls whether overtime work is authorized",
        "templates": [
            {
                "action_value": "restricted",
                "sources": ["Chief Financial Officer", "VP Finance", "Legal & Compliance"],
                "texts": [
                    "Overtime is prohibited company-wide until the next budget cycle begins. All requests must be denied.",
                    "Regulatory advisory: overtime hours for {department} must not exceed {n} hours per employee this period.",
                    "Cost control directive: no overtime approvals will be processed until Q{quarter} budget reconciliation is complete.",
                ],
                "type": "absolute",
            },
            {
                "action_value": "authorized",
                "sources": ["VP Engineering", "Engineering Manager", "VP Operations"],
                "texts": [
                    "Overtime is approved for the {team} team to meet the {deadline} milestone — up to {n} hours per engineer.",
                    "All-hands crunch approved: {department} staff are authorized for overtime through the product launch.",
                    "Project delivery requirements justify overtime. Team leads should coordinate hours with HR.",
                ],
                "type": "absolute",
            },
        ],
    },

    "vendor_contract_approval": {
        "description": "controls who can approve new vendor agreements",
        "templates": [
            {
                "action_value": "executive_only",
                "sources": ["Legal & Compliance", "Chief Financial Officer", "Chief Operating Officer"],
                "texts": [
                    "All new vendor contracts, regardless of value, must be reviewed and signed by a C-suite executive.",
                    "Procurement policy update: no new vendor agreements may be executed without legal review and CFO signature.",
                    "Vendor onboarding is suspended pending completion of the third-party risk assessment framework.",
                ],
                "type": "absolute",
            },
            {
                "action_value": "manager_approved",
                "sources": ["VP Operations", "Director of Finance", "Finance Manager"],
                "texts": [
                    "Directors and above may sign vendor contracts valued below ${high_amount} without further approval.",
                    "Standard delegation of authority: {department} managers may execute agreements under ${high_amount}.",
                    "Proceed with onboarding {vendor} — the contract is within the approved delegation threshold.",
                ],
                "type": "absolute",
            },
        ],
    },

    "system_access_level": {
        "description": "controls data and system access permissions",
        "templates": [
            {
                "action_value": "restricted",
                "sources": ["Legal & Compliance", "Director of IT", "Regulatory Affairs"],
                "texts": [
                    "Access to {system} is restricted to senior staff only. Revoke credentials for all junior employees immediately.",
                    "Pending data protection review: all external access to {system} must be suspended.",
                    "GDPR compliance: {system} access logs must be audited and permissions reduced to the minimum necessary.",
                ],
                "type": "absolute",
            },
            {
                "action_value": "expanded",
                "sources": ["VP Engineering", "IT Manager", "Engineering Manager"],
                "texts": [
                    "Onboard the full {team} team to {system} — access provisioning should be completed by {deadline}.",
                    "New project requires {department} to have read and write access to all {system} environments.",
                    "Approved: grant {system} admin credentials to {n} additional engineers to support the migration.",
                ],
                "type": "absolute",
            },
        ],
    },

    "travel_policy": {
        "description": "governs employee business travel",
        "templates": [
            {
                "action_value": "suspended",
                "sources": ["Chief Financial Officer", "CEO Office", "VP Finance"],
                "texts": [
                    "All non-essential business travel is suspended until further notice to manage Q{quarter} costs.",
                    "Travel freeze in effect: no flights or hotel bookings may be expensed outside of {location} through {deadline}.",
                    "Executive directive: travel spend must be reduced by {pct}% this quarter. Cancel planned trips where possible.",
                ],
                "type": "absolute",
            },
            {
                "action_value": "approved",
                "sources": ["VP Operations", "Director of Finance", "Engineering Manager"],
                "texts": [
                    "Travel to {location} for the {event} is approved for the {team} team representatives.",
                    "Attendance at {event} has been budgeted — team leads should proceed with booking arrangements.",
                    "Client visit to {location} approved. Book flights and accommodation within the standard per-diem limits.",
                ],
                "type": "absolute",
            },
        ],
    },

    "security_patch_schedule": {
        "description": "controls timing of security updates",
        "templates": [
            {
                "action_value": "immediate",
                "sources": ["Legal & Compliance", "Chief Technology Officer", "Director of IT"],
                "texts": [
                    "Critical vulnerability disclosure requires all systems to be patched within {n} hours. No exceptions.",
                    "Security incident response: apply emergency patches to {system} immediately, regardless of other schedules.",
                    "Compliance mandate: all endpoints must run the latest security update before {deadline}. Escalate blockers.",
                ],
                "type": "time_ordered",
            },
            {
                "action_value": "scheduled",
                "sources": ["IT Manager", "Team Lead", "Engineering Manager"],
                "texts": [
                    "Security patches for {system} are scheduled for the next maintenance window on {deadline} to avoid disruption.",
                    "Planned patching cycle: apply updates to {system} during the {timeframe} downtime window as scheduled.",
                    "Patch deployment to {system} will proceed according to the standard quarterly update schedule.",
                ],
                "type": "time_ordered",
            },
        ],
    },

    "data_retention": {
        "description": "governs how long data is stored",
        "templates": [
            {
                "action_value": "delete_immediately",
                "sources": ["Legal & Compliance", "Regulatory Affairs"],
                "texts": [
                    "GDPR deletion request received: all {data_type} data for the specified user IDs must be purged within {n} hours.",
                    "Regulatory order: delete all {data_type} records older than {timeframe} from {system} immediately.",
                    "Right-to-erasure request confirmed. Legal requires immediate deletion of the flagged {data_type} records.",
                ],
                "type": "absolute",
            },
            {
                "action_value": "retain",
                "sources": ["VP Engineering", "Engineering Manager", "IT Manager"],
                "texts": [
                    "Retain all {data_type} records in {system} for ongoing model training — do not purge this dataset.",
                    "The {data_type} data pipeline depends on historical records going back {timeframe}. Do not delete.",
                    "Product analytics requires {data_type} data to be preserved for the next {n} months for reporting.",
                ],
                "type": "absolute",
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Dynamic fill parameters
# ---------------------------------------------------------------------------

FILL_PARAMS = {
    "department": ["Engineering", "Finance", "Operations", "Product", "Marketing", "Sales", "HR"],
    "team": ["Platform", "Backend", "Frontend", "Data", "Security", "Infrastructure", "Analytics"],
    "role": ["senior", "staff", "principal", "junior", "mid-level"],
    "n": ["2", "3", "4", "5", "6", "8", "10"],
    "quarter": ["1", "2", "3", "4"],
    "deadline": ["Monday", "end of this week", "next Friday", "end of month", "this Thursday"],
    "timeframe": ["this sprint", "this quarter", "next month", "the next two weeks", "this fiscal year"],
    "days": ["7", "10", "14", "21", "30"],
    "system": ["Jira", "Confluence", "Salesforce", "GitHub", "the data warehouse", "AWS production", "the analytics platform"],
    "version": ["v2.1", "v3.0", "v1.4.2", "v2.5"],
    "low_amount": ["5,000", "10,000", "15,000", "8,000"],
    "high_amount": ["25,000", "50,000", "75,000", "100,000"],
    "vendor": ["the new SaaS provider", "the cloud services vendor", "the contract staffing agency"],
    "data_type": ["user activity", "transaction", "PII", "behavioral", "log"],
    "location": ["New York", "London", "Singapore", "Bangalore", "San Francisco"],
    "event": ["the industry conference", "the client summit", "the annual review", "the partner meeting"],
    "pct": ["20", "30", "40", "50"],
}

BUSINESS_CONTEXTS = {
    "HR": [
        "Apex Technologies is a mid-size SaaS company currently undergoing a financial audit following an unexpected shortfall in Q{quarter} revenue. The CFO has initiated cost containment measures while the HR team is simultaneously executing an approved growth plan from earlier in the year.",
        "GlobalSoft is preparing for its Series B fundraise and has recently received conflicting guidance from its legal counsel and board regarding headcount management during the due diligence period.",
        "Meridian Corp is navigating a workforce restructuring following its recent acquisition. Multiple directives from the acquiring entity, local management, and compliance teams are active simultaneously.",
    ],
    "Finance": [
        "Vertex Capital is mid-quarter with a new CFO who has introduced revised approval thresholds that conflict with pre-existing delegation-of-authority agreements approved by previous leadership.",
        "NovaTech is preparing its year-end close while simultaneously onboarding a major new client. Finance and operations have each issued guidance that conflicts with the other's spending authority.",
        "Axiom Financial Services is subject to a regulatory examination. The compliance team has issued emergency spending restrictions that clash with pre-committed project budgets.",
    ],
    "IT": [
        "CloudBase Inc. has discovered a critical vulnerability in its production environment. The security team and the engineering team have issued conflicting instructions about remediation timing and deployment freezes.",
        "DataStream Corp is mid-migration to a new cloud infrastructure. A compliance audit has been triggered simultaneously, generating conflicting access and deployment directives from multiple stakeholders.",
        "StrataTech is rolling out a major platform update while also undergoing a SOC 2 Type II audit, creating conflicts between its deployment schedule and its auditor-mandated change freeze.",
    ],
    "Operations": [
        "PackageFlow Logistics is managing a peak season surge while also implementing new vendor approval processes ordered by its legal team following a supply chain incident.",
        "BuildRight Construction is executing multiple concurrent projects while a new corporate travel policy — issued after a budget overrun — conflicts with approved project site visit schedules.",
        "SupplyBridge Corp has received a priority client escalation that requires immediate resource reallocation, in direct conflict with a cost-freeze directive issued by the CFO two weeks earlier.",
    ],
}

# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """
    Generates fully dynamic instruction conflict scenarios.
    No two generated scenarios are identical.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def _fill(self, text: str) -> str:
        """Fill template slots with randomly chosen values."""
        for key, values in FILL_PARAMS.items():
            placeholder = "{" + key + "}"
            while placeholder in text:
                text = text.replace(placeholder, random.choice(values), 1)
        return text

    def _make_instruction(
        self,
        group_key: str,
        template_entry: dict,
        existing_ids: set,
    ) -> Instruction:
        """Construct a single Instruction from a template entry."""
        instr_id = f"INS-{str(uuid.uuid4())[:6].upper()}"
        while instr_id in existing_ids:
            instr_id = f"INS-{str(uuid.uuid4())[:6].upper()}"

        source = random.choice(template_entry["sources"])
        raw_text = random.choice(template_entry["texts"])
        text = self._fill(raw_text)

        return Instruction(
            id=instr_id,
            text=text,
            source=source,
            source_priority=SOURCE_PRIORITY[source],
            instruction_type=template_entry["type"],
            action_key=group_key,
            action_value=template_entry["action_value"],
        )

    def _generate_conflict_pair(
        self, group_key: str, existing_ids: set
    ) -> Tuple[Instruction, Instruction, ConflictPair]:
        """
        Generate one conflict pair from a template group.
        Always picks two templates with DIFFERENT action_values and
        DIFFERENT source priorities (no ties — every conflict has a clear winner).
        Returns (winner_instruction, loser_instruction, conflict_pair).
        """
        group = TEMPLATE_GROUPS[group_key]
        templates = group["templates"]

        # Pick two templates from different action_value buckets
        t_a = templates[0]
        t_b = templates[1]

        # Retry until we get sources with different priority tiers
        for _ in range(20):
            instr_a = self._make_instruction(group_key, t_a, existing_ids)
            instr_b = self._make_instruction(group_key, t_b, existing_ids)
            if instr_a.source_priority != instr_b.source_priority:
                break
        # After retry loop, IDs are committed
        existing_ids.add(instr_a.id)
        existing_ids.add(instr_b.id)

        # Resolve: strictly lower priority number = higher authority = wins
        if instr_a.source_priority < instr_b.source_priority:
            winner, loser = instr_a, instr_b
        else:
            winner, loser = instr_b, instr_a

        explanation = (
            f"{winner.source} (authority level {winner.source_priority}) overrides "
            f"{loser.source} (authority level {loser.source_priority}) on {group_key.replace('_', ' ')}."
        )

        conflict = ConflictPair(
            instruction_a_id=instr_a.id,
            instruction_b_id=instr_b.id,
            conflict_type=instr_a.instruction_type,
            resolution_id=winner.id,
            explanation=explanation,
        )

        return winner, loser, conflict

    def _generate_filler_instructions(
        self, n: int, existing_ids: set, existing_keys: set
    ) -> List[Instruction]:
        """
        Generate n non-conflicting filler instructions to pad the scenario.
        These are always 'follow' instructions with no opposing directive.
        """
        fillers = []
        available_groups = [k for k in TEMPLATE_GROUPS if k not in existing_keys]
        random.shuffle(available_groups)

        for i in range(min(n, len(available_groups))):
            group_key = available_groups[i]
            group = TEMPLATE_GROUPS[group_key]
            # Pick either template but only one per group (no conflict)
            t = random.choice(group["templates"])
            instr = self._make_instruction(group_key, t, existing_ids)
            existing_ids.add(instr.id)
            fillers.append(instr)

        return fillers

    def _format_prompt(self, context: str, instructions: List[Instruction]) -> str:
        """Format the full LLM prompt for the scenario."""
        random.shuffle(instructions)  # randomize presentation order

        instruction_lines = []
        for ins in instructions:
            instruction_lines.append(
                f"[{ins.id}] From {ins.source}:\n  {ins.text}"
            )
        instruction_block = "\n\n".join(instruction_lines)

        prompt = f"""You are a business operations coordinator. Resolve instruction conflicts using the authority hierarchy below.

AUTHORITY HIERARCHY (lower number = higher authority):
1. Legal & Compliance, Regulatory Affairs
2. CEO Office, CFO, CTO, COO
3. VP Engineering, VP Finance, VP Operations, VP Human Resources
4. Director of IT, Director of Finance
5. Engineering Manager, Finance Manager, HR Manager, IT Manager
6. Team Lead, Department Coordinator
When two instructions conflict, the one from the higher-authority source (lower tier number) ALWAYS wins.

CONTEXT: {context}

INSTRUCTIONS:
{instruction_block}

Output ONLY a JSON object. No preamble, no explanation outside the JSON. Keep "reasoning" to one short sentence.
{{"identified_conflicts":[{{"instruction_a":"<ID>","instruction_b":"<ID>","conflict_type":"direct|resource|temporal","resolution":"<winning ID>","reasoning":"<one sentence>"}}],"execution_plan":["<ID>",...],"overridden_instructions":["<ID>",...]}}"""
        return prompt

    def generate(self, difficulty: int = 2, domain: Optional[str] = None) -> Scenario:
        """
        Generate one complete scenario.

        Args:
            difficulty: 1 = 2 conflicts, 2 = 4 conflicts, 3 = 6 conflicts
            domain: "HR" | "Finance" | "IT" | "Operations" | None (random)

        Returns:
            A fully populated Scenario object.
        """
        difficulty = max(1, min(3, difficulty))
        num_conflicts = {1: 2, 2: 4, 3: 6}[difficulty]

        if domain is None:
            domain = random.choice(list(BUSINESS_CONTEXTS.keys()))

        # Select business context
        context_template = random.choice(BUSINESS_CONTEXTS[domain])
        context = self._fill(context_template)

        # Select template groups for conflicts (no repetition)
        all_groups = list(TEMPLATE_GROUPS.keys())
        random.shuffle(all_groups)
        conflict_groups = all_groups[:num_conflicts]

        existing_ids: set = set()
        all_instructions: List[Instruction] = []
        all_conflicts: List[ConflictPair] = []
        ground_truth_followed: List[str] = []
        ground_truth_overridden: List[str] = []

        for group_key in conflict_groups:
            winner, loser, conflict_pair = self._generate_conflict_pair(group_key, existing_ids)
            all_instructions.append(winner)
            all_instructions.append(loser)
            all_conflicts.append(conflict_pair)
            ground_truth_followed.append(winner.id)
            ground_truth_overridden.append(loser.id)

        # Add non-conflicting filler instructions (makes scenario realistic)
        used_keys = set(conflict_groups)
        filler_count = random.randint(2, 4)
        fillers = self._generate_filler_instructions(filler_count, existing_ids, used_keys)
        all_instructions.extend(fillers)
        ground_truth_followed.extend([f.id for f in fillers])

        prompt = self._format_prompt(context, all_instructions)

        return Scenario(
            scenario_id=str(uuid.uuid4()),
            domain=domain,
            difficulty=difficulty,
            business_context=context,
            instructions=all_instructions,
            conflicts=all_conflicts,
            ground_truth_followed=sorted(ground_truth_followed),
            ground_truth_overridden=sorted(ground_truth_overridden),
            prompt=prompt,
        )

    def generate_batch(
        self, n: int, difficulty: Optional[int] = None, domain: Optional[str] = None
    ) -> List[Scenario]:
        """Generate a batch of n scenarios with optionally fixed difficulty/domain."""
        scenarios = []
        for i in range(n):
            d = difficulty if difficulty is not None else random.randint(1, 3)
            scenarios.append(self.generate(difficulty=d, domain=domain))
        return scenarios
