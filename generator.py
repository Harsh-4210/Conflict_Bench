"""
ConflictBench — Scenario Generator
Dynamically generates instruction conflict scenarios for LLM training.
No hardcoded scenarios. Every episode is freshly generated from templates.

Two generators live here, selected by ``GeneratorConfig``:
  - legacy (``GeneratorConfig(legacy=True)``): the original task, byte for byte.
    Each conflict = the two templates of one topic, the restrictive side almost
    always wins, the hierarchy is printed. Solvable by a lookup script; kept only
    for comparison.
  - v2 (the default): balanced winners, scoped distractors, hidden hierarchy,
    standing rules, delegations, dated supersession, conditional instructions,
    chains, and train/val/test splits (FIX_BLUEPRINT.md Phase 3). Its ground truth
    comes from ``resolver.resolve()``, never from template order.

Reproducibility: every random draw (text, sources, IDs, scenario_id) comes from a
private ``random.Random`` instance. The global ``random`` module is never touched.
"""

import datetime
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Sequence, Tuple
from copy import deepcopy


# ---------------------------------------------------------------------------
# Data structures
# (Fields added after Phase 2 all have defaults, so old JSON still loads.)
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    id: str
    text: str
    source: str          # canonical office, e.g. "Legal & Compliance", "CEO Office"
    source_priority: int  # raw tier of the office: 1 = highest authority (Legal), 6 = lowest
    instruction_type: str  # "absolute" | "conditional" | "time_ordered" | "resource"
    action_key: str      # semantic conflict key, e.g. "hiring_status"
    action_value: str    # the directive, e.g. "frozen" | "active"
    condition: Optional[str] = None  # fact key; the instruction is in force only if that fact is True
    action_scope: str = "all"        # "all" or one department/system; conflicts need overlapping scopes
    polarity: str = ""               # "restrictive" | "permissive" | "" (analysis only, never used to resolve)
    author: str = ""                 # person shown in the prompt (v2)
    title: str = ""                  # job title shown in the prompt (v2)
    effective_date: Optional[str] = None  # ISO date; breaks ties between equal authority
    depends_on: Optional[str] = None      # ID of the instruction this one is contingent on


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
    difficulty: int       # legacy: 1/2/3 = 2/4/6 conflicts. v2: see DIFFICULTY_LEVELS
    business_context: str
    instructions: List[Instruction]
    conflicts: List[ConflictPair]
    ground_truth_followed: List[str]   # instruction IDs agent SHOULD follow
    ground_truth_overridden: List[str]  # instruction IDs agent SHOULD override
    prompt: str           # fully formatted LLM prompt
    facts: Dict[str, bool] = field(default_factory=dict)          # status facts used by conditions
    authority_rules: List[Dict[str, str]] = field(default_factory=list)  # {"action_key", "office"}
    delegations: List[Dict[str, str]] = field(default_factory=list)      # {"action_key", "from_office", "to_office"}
    split: str = ""               # "train" | "val" | "test" ("" = legacy)
    generator_version: str = ""   # "v2" for the Phase 3 generator ("" = legacy)


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

# Internal instruction types -> labels shown to the agent in the prompt.
# ConflictPair.conflict_type stores the mapped label, so the ground truth
# always uses the same vocabulary the prompt asks for.
CONFLICT_TYPE_LABELS = {
    "absolute":     "direct",
    "resource":     "resource",
    "time_ordered": "temporal",
    "conditional":  "conditional",
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

# Conflict-type labels the prompt offers to the agent: the mapped label of every
# instruction type that a template can actually produce, in CONFLICT_TYPE_LABELS
# order. (Grows automatically once "conditional" templates exist.)
_USED_INSTRUCTION_TYPES = {
    tmpl["type"] for grp in TEMPLATE_GROUPS.values() for tmpl in grp["templates"]
}
PROMPT_CONFLICT_TYPES = tuple(
    label for itype, label in CONFLICT_TYPE_LABELS.items() if itype in _USED_INSTRUCTION_TYPES
)


# ===========================================================================
# v2 generator library (Phase 3)
# ===========================================================================

SPLITS = ("train", "val", "test")


def split_pool(items: Sequence, split: str) -> list:
    """
    Deterministic, disjoint train/val/test partition of a list: the last
    max(1, n//5) items are test, the max(1, n//5) before them are val, the rest
    train. Order in the source list decides; nothing random.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    items = list(items)
    n = len(items)
    k = max(1, n // 5)
    if n < 3:
        raise ValueError(f"need at least 3 items to split, got {n}")
    if split == "train":
        return items[: n - 2 * k]
    if split == "val":
        return items[n - 2 * k: n - k]
    return items[n - k:]


@dataclass(frozen=True)
class GeneratorConfig:
    """
    Difficulty knobs for ScenarioGenerator. The default is the Phase 3 task.

    legacy=True reproduces the original generator exactly (every other knob is
    then ignored). Each rule has its own on/off flag so its effect can be measured.
    """
    legacy: bool = False
    split: str = "train"                 # which pool of phrasings/contexts/titles to draw from
    show_hierarchy: bool = False         # print the tier list (True) or only names + job titles (False)
    restrictive_win_rate: float = 0.5    # P(the restrictive side wins a conflict) (P3-1)
    # P3-2 distractors
    distractors: bool = True
    distractor_ratio: float = 0.75       # distractor topics per conflict (capped by the 17 topics)
    p_three_distractors: float = 0.5     # P(a distractor topic has 3 instructions instead of 2)
    attached_distractor_prob: float = 0.5  # P(a conflict topic gets an extra non-conflicting instruction)
    # P3-4 rules
    scoped_authority: bool = True        # standing rule: one person has final say on one topic
    delegation: bool = True              # a senior person delegates authority on one topic
    time_supersession: bool = True       # dated instructions; equal authority -> later wins
    conditionals: bool = True            # "If <fact>: ..." instructions + CURRENT STATUS facts
    chains: bool = True                  # "Contingent on [ID] going ahead: ..." instructions
    decoy_notices: bool = True           # rules/delegations on topics where they change nothing
    p_scoped: float = 0.15               # share of conflicts decided by a standing rule
    p_delegation: float = 0.15           # share decided by a delegation notice
    p_time: float = 0.25                 # share between equal authority, decided by date
    p_conditional: float = 0.2           # P(one side of a real conflict is conditional (and true))
    chain_prob: float = 0.3              # P(a conflict gets a contingent follow-up instruction)

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        for name in ("restrictive_win_rate", "attached_distractor_prob", "p_three_distractors", "p_scoped",
                     "p_delegation", "p_time", "p_conditional", "chain_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.distractor_ratio < 0:
            raise ValueError("distractor_ratio must be >= 0")
        if self.p_scoped + self.p_delegation + self.p_time > 1.0 + 1e-9:
            raise ValueError("p_scoped + p_delegation + p_time must be <= 1")

    @classmethod
    def legacy_config(cls) -> "GeneratorConfig":
        return cls(legacy=True)


# v2 difficulty levels: conflicts per scenario and filler range. Distractor topics
# = int(conflicts * distractor_ratio + 0.5). 1-3 keep the legacy conflict counts.
DIFFICULTY_LEVELS = {
    1: {"conflicts": 2, "fillers": (1, 2)},
    2: {"conflicts": 4, "fillers": (1, 2)},
    3: {"conflicts": 6, "fillers": (1, 2)},
    4: {"conflicts": 8, "fillers": (0, 1)},
    5: {"conflicts": 10, "fillers": (0, 1)},
}
MAX_DIFFICULTY = max(DIFFICULTY_LEVELS)


def clamp_difficulty(difficulty: Any, max_level: int = MAX_DIFFICULTY) -> int:
    """Any value -> a valid level in [1, max_level]; unreadable values -> 2."""
    try:
        level = int(difficulty)
    except (TypeError, ValueError):
        return 2
    return max(1, min(max_level, level))


# Offices: function tags decide which topics an office plausibly speaks about.
# "legal", "exec" and "general" offices can speak about every topic.
OFFICE_FUNCTIONS = {
    "Legal & Compliance":       ("legal",),
    "Regulatory Affairs":       ("legal",),
    "CEO Office":               ("exec",),
    "Chief Financial Officer":  ("exec",),
    "Chief Technology Officer": ("exec",),
    "Chief Operating Officer":  ("exec",),
    "VP Engineering":           ("eng",),
    "VP Finance":               ("fin",),
    "VP Operations":            ("ops",),
    "VP Human Resources":       ("hr",),
    "Director of IT":           ("it",),
    "Director of Finance":      ("fin",),
    "Engineering Manager":      ("eng",),
    "Finance Manager":          ("fin",),
    "HR Manager":               ("hr",),
    "IT Manager":               ("it",),
    "Team Lead":                ("eng", "ops", "it"),
    "Department Coordinator":   ("general",),
}

# Job titles shown in hidden-hierarchy mode. Every title names its level plainly
# (Counsel/Legal/Compliance/Regulatory, Chief/CxO, VP/SVP/EVP, Director,
# Manager, Lead/Coordinator), so the AUTHORITY principles in the prompt are
# enough to rank them. split_pool() holds the last title of each office out for
# test and the one before it for val.
OFFICE_TITLES = {
    "Legal & Compliance":       ["General Counsel", "Chief Legal Officer", "Head of Legal",
                                 "Chief Compliance Officer", "Deputy General Counsel"],
    "Regulatory Affairs":       ["Head of Regulatory Affairs", "Chief Regulatory Officer",
                                 "Regulatory Compliance Director", "Director of Regulatory Affairs",
                                 "Senior Regulatory Counsel"],
    "CEO Office":               ["Chief Executive Officer", "CEO", "President and CEO",
                                 "Founder and CEO", "Group Chief Executive"],
    "Chief Financial Officer":  ["Chief Financial Officer", "CFO", "Group CFO",
                                 "Chief Finance Officer", "Group Chief Financial Officer"],
    "Chief Technology Officer": ["Chief Technology Officer", "CTO", "Group CTO",
                                 "Chief Technical Officer", "Group Chief Technology Officer"],
    "Chief Operating Officer":  ["Chief Operating Officer", "COO", "Group COO",
                                 "Chief Operations Officer", "Group Chief Operating Officer"],
    "VP Engineering":           ["VP Engineering", "Vice President of Engineering", "SVP Engineering",
                                 "EVP Engineering", "VP, Software Engineering"],
    "VP Finance":               ["VP Finance", "Vice President of Finance", "SVP Finance",
                                 "VP, Financial Planning", "EVP Finance"],
    "VP Operations":            ["VP Operations", "Vice President of Operations", "SVP Operations",
                                 "VP, Global Operations", "EVP Operations"],
    "VP Human Resources":       ["VP Human Resources", "VP People", "SVP People and Culture",
                                 "Vice President, Talent", "EVP Human Resources"],
    "Director of IT":           ["Director of IT", "IT Director", "Senior Director, IT",
                                 "Director of Infrastructure", "Director, Information Systems"],
    "Director of Finance":      ["Director of Finance", "Senior Director, FP&A", "Director of Financial Planning",
                                 "Director, Accounting", "Senior Finance Director"],
    "Engineering Manager":      ["Engineering Manager", "Senior Engineering Manager", "Software Development Manager",
                                 "Platform Engineering Manager", "Manager, Backend Engineering"],
    "Finance Manager":          ["Finance Manager", "Accounting Manager", "Senior Finance Manager",
                                 "Payroll Manager", "Manager, Financial Operations"],
    "HR Manager":               ["HR Manager", "People Operations Manager", "Talent Acquisition Manager",
                                 "HR Business Partner Manager", "Manager, Employee Relations"],
    "IT Manager":               ["IT Manager", "IT Operations Manager", "Systems Manager",
                                 "Service Desk Manager", "Manager, IT Infrastructure"],
    "Team Lead":                ["Team Lead", "Tech Lead", "Squad Lead", "Shift Lead", "Team Leader"],
    "Department Coordinator":   ["Department Coordinator", "Operations Coordinator", "Program Coordinator",
                                 "Project Coordinator", "Office Coordinator"],
}

FIRST_NAMES = [
    "Aisha", "Ben", "Carlos", "Dana", "Elena", "Farid", "Grace", "Hiro", "Isla", "Jonas",
    "Kavya", "Liam", "Maya", "Noah", "Olga", "Priya", "Quinn", "Rafael", "Sara", "Tom",
    "Uma", "Victor", "Wen", "Yusuf", "Zoe", "Arjun", "Bianca", "Chen", "Diego", "Emma",
]
LAST_NAMES = [
    "Kim", "Patel", "Garcia", "Nair", "Okafor", "Schmidt", "Rossi", "Tanaka", "Silva", "Haddad",
    "Novak", "Brown", "Cohen", "Mensah", "Iyer", "Larsen", "Dubois", "Moreau", "Khan", "Reyes",
    "Sato", "Walsh", "Yilmaz", "Zhang", "Ahmed",
]

DEPARTMENTS = ["Engineering", "Finance", "Operations", "Product", "Marketing", "Sales", "HR", "Support"]
SYSTEMS = ["Jira", "Salesforce", "GitHub", "the data warehouse", "AWS production",
           "the analytics platform", "the billing system", "the HR portal"]

# How a scope is written inside an instruction. {x} = department or system.
SCOPE_PHRASES = {
    "dept": {"all": ["across the company", "company-wide", "in every department"],
             "one": ["in the {x} department", "for the {x} team", "within {x}"]},
    "system": {"all": ["on all systems", "across every system"],
               "one": ["on {x}", "in {x}"]},
}

# Slot values for v2 texts. No slot here names a department, team or system:
# scope is expressed only through {scope}.
V2_FILL = {
    "n": ["2", "3", "4", "5", "6", "8"],
    "days": ["7", "10", "14", "21", "30"],
    "version": ["v2.1", "v3.0", "v1.4.2", "v2.5"],
    "low_amount": ["5,000", "8,000", "10,000", "15,000"],
    "high_amount": ["25,000", "50,000", "75,000", "100,000"],
    "vendor": ["the new SaaS provider", "the cloud services vendor", "the staffing agency"],
    "data_type": ["user activity", "transaction", "customer contact", "behavioural", "log"],
    "location": ["New York", "London", "Singapore", "Bangalore", "San Francisco"],
    "event": ["the industry conference", "the client summit", "the partner meeting"],
    "pct": ["10", "15", "20", "25"],
}

# 17 topics. Each side has 5 phrasings: split_pool() gives 3 to train, 1 to val,
# 1 to test. Every phrasing has exactly one {scope}. Both sides of every topic
# can come from any office in the topic's pool (any tier), so who wins depends on
# authority, never on which side a text is from (P3-1).
V2_TOPICS = {
    "hiring_status": {
        "label": "hiring", "scope_kind": "dept", "type": "absolute", "functions": ("hr", "fin", "eng"),
        "restrictive": ("frozen", [
            "All hiring {scope} is frozen until further notice.",
            "Open headcount requests {scope} are paused; do not post new roles.",
            "Stop all recruiting activity {scope} and withdraw unsent offers.",
            "No new hires may be approved {scope} this quarter.",
            "Hiring {scope} is on hold while headcount plans are reviewed.",
        ]),
        "permissive": ("active", [
            "Go ahead and hire {n} new people {scope} this quarter.",
            "The hiring plan {scope} is approved; start recruiting for {n} open roles.",
            "Recruiters may extend offers to shortlisted candidates {scope}.",
            "Open {n} new positions {scope} and begin interviews this week.",
            "Backfill every vacant role {scope} as soon as candidates are found.",
        ]),
    },
    "deployment_approval": {
        "label": "production deployments", "scope_kind": "system", "type": "absolute", "functions": ("eng", "it"),
        "restrictive": ("blocked", [
            "No production deployments are allowed {scope} until the security review is signed off.",
            "Deployments {scope} are frozen for the next {days} days.",
            "Lock the release pipeline {scope}; nothing ships without a new sign-off.",
            "All deployment windows {scope} are cancelled this week.",
            "Do not push any release {scope} until further notice.",
        ]),
        "permissive": ("approved", [
            "Release {version} is cleared for production deployment {scope} this week.",
            "The team may deploy to production {scope} during the normal window.",
            "Ship the pending updates {scope} as planned.",
            "Deployment of release {version} {scope} is approved for Thursday.",
            "Resume regular production releases {scope} starting today.",
        ]),
    },
    "budget_approval_threshold": {
        "label": "spending approval limits", "scope_kind": "dept", "type": "resource", "functions": ("fin", "ops"),
        "restrictive": ("strict_limit", [
            "Every expense over ${low_amount} {scope} now needs executive sign-off.",
            "Purchase orders above ${low_amount} {scope} require executive approval.",
            "Spending {scope} is capped at ${low_amount} per item until the review ends.",
            "Managers {scope} may not approve anything above ${low_amount} on their own.",
            "Any commitment over ${low_amount} {scope} must go to the executive committee first.",
        ]),
        "permissive": ("standard_limit", [
            "Managers {scope} may approve expenses up to ${high_amount} without further review.",
            "Department heads {scope} keep their usual approval limit of ${high_amount} per purchase.",
            "Spending up to ${high_amount} per item {scope} needs no extra approval.",
            "The standard ${high_amount} approval limit stays in place {scope}.",
            "Teams {scope} can commit up to ${high_amount} under normal delegation rules.",
        ]),
    },
    "remote_work_policy": {
        "label": "remote work", "scope_kind": "dept", "type": "absolute", "functions": ("hr", "eng", "ops"),
        "restrictive": ("mandatory_office", [
            "Everyone {scope} must work from the office five days a week.",
            "Remote work {scope} is suspended; staff return on-site from Monday.",
            "Full in-office attendance is required {scope} during the launch.",
            "Hybrid schedules {scope} are cancelled; office attendance is mandatory.",
            "Staff {scope} may no longer work from home.",
        ]),
        "permissive": ("flexible", [
            "Staff {scope} may work remotely up to {n} days a week.",
            "Hybrid working continues {scope} with no change.",
            "Teams {scope} may stay remote-first through the end of the quarter.",
            "Managers {scope} can approve home working as they see fit.",
            "Flexible schedules {scope} remain in place.",
        ]),
    },
    "overtime_authorization": {
        "label": "overtime", "scope_kind": "dept", "type": "absolute", "functions": ("fin", "eng", "ops", "hr"),
        "restrictive": ("restricted", [
            "No overtime may be approved {scope} until the next budget cycle.",
            "Overtime {scope} is banned this month; deny all requests.",
            "Overtime hours {scope} are capped at zero for this period.",
            "Do not log or pay any overtime {scope} until further notice.",
            "All overtime requests {scope} must be rejected.",
        ]),
        "permissive": ("authorized", [
            "Overtime of up to {n} hours per person is approved {scope} to hit the deadline.",
            "Staff {scope} may work paid overtime through the launch.",
            "Approve overtime requests {scope} as needed this sprint.",
            "Extra paid hours {scope} are authorized for the next {days} days.",
            "Overtime {scope} is allowed to clear the current backlog.",
        ]),
    },
    "vendor_contract_approval": {
        "label": "vendor contracts", "scope_kind": "dept", "type": "absolute", "functions": ("fin", "ops"),
        "restrictive": ("executive_only", [
            "New vendor contracts {scope} need a C-suite signature, whatever their value.",
            "Vendor onboarding {scope} is suspended pending the risk review.",
            "No new supplier agreements may be signed {scope} without legal review.",
            "Managers {scope} may not sign any vendor contract themselves.",
            "Hold all pending vendor agreements {scope} for executive approval.",
        ]),
        "permissive": ("manager_approved", [
            "Managers {scope} may sign vendor contracts under ${high_amount} directly.",
            "Proceed with onboarding {vendor} {scope}; it is within normal limits.",
            "Standard delegation applies {scope}: contracts below ${high_amount} need no extra sign-off.",
            "Teams {scope} can renew existing supplier contracts without escalation.",
            "Directors {scope} may approve new vendors under ${high_amount}.",
        ]),
    },
    "system_access_level": {
        "label": "system access", "scope_kind": "system", "type": "absolute", "functions": ("it", "eng"),
        "restrictive": ("restricted", [
            "Access {scope} is limited to senior staff; revoke all other credentials.",
            "Suspend all external accounts {scope} pending the data protection review.",
            "Cut permissions {scope} to the minimum each role needs.",
            "No new access grants {scope} until the access audit is complete.",
            "Remove admin rights {scope} from everyone outside the core team.",
        ]),
        "permissive": ("expanded", [
            "Give the new project team full read and write access {scope}.",
            "Grant admin credentials {scope} to {n} more engineers for the migration.",
            "Open access {scope} to all contractors on the project.",
            "Provision accounts {scope} for every new joiner this week.",
            "Expand permissions {scope} so analysts can run their own queries.",
        ]),
    },
    "travel_policy": {
        "label": "business travel", "scope_kind": "dept", "type": "absolute", "functions": ("fin", "ops"),
        "restrictive": ("suspended", [
            "All non-essential travel {scope} is suspended until further notice.",
            "No flights or hotels may be booked {scope} this quarter.",
            "Cancel planned trips {scope} and switch to video calls.",
            "Travel spend {scope} must drop by {pct}%; postpone every trip you can.",
            "Business travel {scope} is frozen.",
        ]),
        "permissive": ("approved", [
            "Travel to {location} for {event} is approved {scope}.",
            "Staff {scope} may book trips within the standard per-diem limits.",
            "Client visits {scope} can go ahead as planned.",
            "Book travel for {event} {scope}; the budget is in place.",
            "Normal travel booking resumes {scope}.",
        ]),
    },
    "security_patch_schedule": {
        "label": "security patching", "scope_kind": "system", "type": "time_ordered", "functions": ("it", "eng"),
        "restrictive": ("immediate", [
            "Patch every server {scope} within {n} hours; no exceptions.",
            "Apply the emergency security fix {scope} today.",
            "Install the latest security update {scope} before the end of the day.",
            "Take services {scope} offline if needed to patch them immediately.",
            "Emergency patching {scope} starts now.",
        ]),
        "permissive": ("scheduled", [
            "Security patches {scope} will wait for the next maintenance window.",
            "Apply updates {scope} on the usual quarterly schedule.",
            "Patch servers {scope} during the planned weekend downtime.",
            "Avoid service interruptions {scope}; patch at the scheduled window.",
            "Keep the standard patch calendar {scope}.",
        ]),
    },
    "data_retention": {
        "label": "data retention", "scope_kind": "system", "type": "absolute", "functions": ("it", "eng"),
        "restrictive": ("delete_immediately", [
            "Delete all {data_type} records older than {n} months {scope} immediately.",
            "Purge the flagged {data_type} data {scope} within {n} hours.",
            "Erase {data_type} data {scope} to meet the deletion request.",
            "Remove stored {data_type} records {scope} this week.",
            "Wipe the {data_type} archive {scope} now.",
        ]),
        "permissive": ("retain", [
            "Keep all {data_type} records {scope}; do not purge anything.",
            "Retain {data_type} data {scope} for another {n} months for reporting.",
            "The {data_type} history {scope} must be preserved for model training.",
            "Archive {data_type} data {scope} instead of deleting it.",
            "Hold {data_type} records {scope} until the analytics project finishes.",
        ]),
    },
    "contractor_usage": {
        "label": "contractors", "scope_kind": "dept", "type": "absolute", "functions": ("hr", "fin", "eng"),
        "restrictive": ("banned", [
            "No new contractors may be engaged {scope}.",
            "End all contractor agreements {scope} at the end of the month.",
            "Contractor spend {scope} is frozen.",
            "Do not renew any contractor {scope} this quarter.",
            "Use of agency staff {scope} is suspended.",
        ]),
        "permissive": ("allowed", [
            "Bring in {n} contractors {scope} to cover the backlog.",
            "Contractor renewals {scope} are approved for another quarter.",
            "Teams {scope} may hire agency staff for short projects.",
            "Extend the current contractors {scope} through the launch.",
            "Engage outside specialists {scope} as needed.",
        ]),
    },
    "software_purchases": {
        "label": "software purchases", "scope_kind": "dept", "type": "resource", "functions": ("fin", "it"),
        "restrictive": ("paused", [
            "No new software subscriptions may be bought {scope}.",
            "Pause all software purchases {scope} until the licence audit ends.",
            "Cancel pending software orders {scope}.",
            "Software spend {scope} is frozen this quarter.",
            "Do not renew any tool licences {scope} without review.",
        ]),
        "permissive": ("approved", [
            "Buy the {n} extra licences requested {scope}.",
            "Teams {scope} may purchase the tools they need up to ${high_amount}.",
            "Software renewals {scope} are approved as planned.",
            "Go ahead with the new analytics tool purchase {scope}.",
            "Licence top-ups {scope} can be ordered this week.",
        ]),
    },
    "customer_discounts": {
        "label": "customer discounts", "scope_kind": "dept", "type": "resource", "functions": ("fin", "ops"),
        "restrictive": ("capped", [
            "No discounts may be offered {scope} this quarter.",
            "Discounts {scope} need finance approval before they are offered.",
            "Stop offering promotional pricing {scope}.",
            "Price concessions {scope} are frozen until margins recover.",
            "Withdraw all unsent discount offers {scope}.",
        ]),
        "permissive": ("allowed", [
            "Reps {scope} may offer up to {pct}% off to close deals.",
            "Promotional pricing {scope} is approved for the campaign.",
            "Offer the standard renewal discount {scope}.",
            "Account managers {scope} can approve discounts on their own.",
            "Run the planned customer promotion {scope}.",
        ]),
    },
    "press_communications": {
        "label": "public announcements", "scope_kind": "dept", "type": "absolute", "functions": ("ops", "hr"),
        "restrictive": ("embargo", [
            "No public announcements may be made {scope} until the deal closes.",
            "All press contact {scope} is on hold.",
            "Do not publish blog posts or social updates {scope} this month.",
            "Cancel the planned press release {scope}.",
            "External communications {scope} are frozen.",
        ]),
        "permissive": ("open", [
            "Publish the planned announcement {scope} on Tuesday.",
            "Teams {scope} may post product updates as usual.",
            "Go ahead with the press release {scope}.",
            "Normal social media posting resumes {scope}.",
            "Share the customer success story {scope} this week.",
        ]),
    },
    "bonus_payouts": {
        "label": "bonus payouts", "scope_kind": "dept", "type": "resource", "functions": ("hr", "fin"),
        "restrictive": ("deferred", [
            "All bonus payments {scope} are deferred to next quarter.",
            "Do not pay any bonuses {scope} until results are final.",
            "Bonus payouts {scope} are suspended.",
            "Hold every pending bonus {scope}.",
            "Cancel this cycle's spot bonuses {scope}.",
        ]),
        "permissive": ("scheduled", [
            "Pay the quarterly bonuses {scope} on schedule.",
            "Bonus payouts {scope} are approved for this month's payroll.",
            "Release the pending spot bonuses {scope}.",
            "Go ahead with the annual bonus run {scope}.",
            "Bonuses {scope} will be paid as planned.",
        ]),
    },
    "training_budget": {
        "label": "training spend", "scope_kind": "dept", "type": "resource", "functions": ("hr", "fin"),
        "restrictive": ("cancelled", [
            "External training spend {scope} is cancelled for the year.",
            "No course or certification fees may be paid {scope}.",
            "Pause all training bookings {scope}.",
            "Training budgets {scope} are frozen.",
            "Withdraw pending course registrations {scope}.",
        ]),
        "permissive": ("approved", [
            "Staff {scope} may book approved courses up to ${low_amount} each.",
            "The training budget {scope} is released for this quarter.",
            "Certification fees {scope} will be reimbursed as usual.",
            "Enrol the {n} nominated people {scope} in the leadership course.",
            "Training requests {scope} can be approved by line managers.",
        ]),
    },
    "data_export": {
        "label": "data exports", "scope_kind": "system", "type": "absolute", "functions": ("it", "eng"),
        "restrictive": ("prohibited", [
            "No customer data may be exported {scope}.",
            "Block all bulk data downloads {scope}.",
            "Data exports {scope} are suspended pending the privacy review.",
            "Disable the export feature {scope}.",
            "Stop sharing extracts {scope} with outside partners.",
        ]),
        "permissive": ("permitted", [
            "Analysts may export reports {scope} as needed.",
            "Send the monthly data extract {scope} to the partner as usual.",
            "Enable bulk exports {scope} for the migration.",
            "Teams may download the datasets they need {scope}.",
            "Scheduled data exports {scope} continue as normal.",
        ]),
    },
}

# Neutral sign-offs appended to some instructions (variety only, no meaning).
V2_SIGNOFFS = [
    " Please confirm once done.", " Questions to the operations desk.", " Thanks, everyone.",
    " This was discussed at the leadership meeting.", " Please share with your teams.",
    " Details are in the shared drive.", " Flag any blockers early.", " Thank you for your help.",
    " More details will follow.", " Reply here if anything is unclear.",
]

# Follow-up actions for contingent ("chain") instructions.
V2_FOLLOWUPS = [
    "notify every affected employee by email", "update the internal policy page",
    "brief the regional leads on Friday", "log the change in the governance tracker",
    "schedule a follow-up review in four weeks", "update the planning spreadsheet",
    "tell the customer success team", "add the change to the monthly report",
    "inform the works council", "revise the team calendars",
]

# Status facts for conditional instructions: (phrase used after "If", true line, false line).
V2_FACTS = {
    "audit_open": ("the external audit is still open",
                   "The external audit is still open.", "The external audit has closed."),
    "launch_live": ("the product launch has gone live",
                    "The product launch went live this week.", "The product launch has been postponed."),
    "budget_review_done": ("the budget review is complete",
                           "The budget review is complete.", "The budget review is still in progress."),
    "incident_active": ("the security incident is still active",
                        "The security incident is still active.", "The security incident has been resolved."),
    "deal_pending": ("the acquisition is still pending",
                     "The acquisition is still pending.", "The acquisition has closed."),
    "target_met": ("the hiring target has been met",
                   "The hiring target has been met.", "The hiring target has not been met."),
    "peak_season": ("peak season has started",
                    "Peak season has started.", "Peak season has not started yet."),
    "contract_signed": ("the major client contract is signed",
                        "The major client contract is signed.", "The major client contract is not signed yet."),
}

V2_COMPANIES = [
    "Apex Technologies", "GlobalSoft", "Meridian Corp", "Vertex Capital", "NovaTech",
    "Axiom Financial", "CloudBase Inc.", "DataStream Corp", "StrataTech",
    "PackageFlow Logistics", "BuildRight Construction", "SupplyBridge",
    "Helios Analytics", "Northwind Retail", "Quantia Health",
]

V2_SITUATIONS = {
    "HR": [
        "a mid-size SaaS company going through its annual workforce planning.",
        "a growing startup preparing for a funding round.",
        "a services firm integrating a recently acquired team.",
        "a regional employer reviewing its people policies after a reorganisation.",
        "a software company resetting its headcount plan mid-year.",
    ],
    "Finance": [
        "a company in the middle of its quarterly close.",
        "a firm adjusting its spending controls after a weak quarter.",
        "a business preparing for an external financial review.",
        "a company rebalancing budgets across its divisions.",
        "a lender tightening its internal approval processes.",
    ],
    "IT": [
        "a cloud company running a large infrastructure migration.",
        "a platform business preparing for a security certification.",
        "a software firm recovering from a production incident.",
        "a data company consolidating its internal systems.",
        "a tech company rolling out a new access management tool.",
    ],
    "Operations": [
        "a logistics company heading into its busy season.",
        "a construction firm running several projects at once.",
        "a supplier handling a sudden surge in client demand.",
        "a manufacturer reorganising its vendor base.",
        "a retail chain preparing for a store expansion.",
    ],
}

V2_DATE_START = datetime.date(2026, 1, 5)
V2_DATE_SPAN_DAYS = 80

V2_CONFLICT_TYPES = ("direct", "resource", "temporal", "conditional")


def _tier_list_lines() -> List[str]:
    tiers: Dict[int, List[str]] = {}
    for office, tier in SOURCE_PRIORITY.items():
        tiers.setdefault(tier, []).append(office)
    return [f"{t}. {', '.join(tiers[t])}" for t in sorted(tiers)]


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def format_date(iso: str) -> str:
    """'2026-03-07' -> '7 Mar 2026' (locale independent)."""
    d = datetime.date.fromisoformat(iso)
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def _other_side(side: str) -> str:
    return "permissive" if side == "restrictive" else "restrictive"


class _V2Builder:
    """
    Builds one v2 scenario. It records what it *intends* each instruction's
    outcome to be; the generator then checks that resolver.resolve() derives
    exactly that from the structured fields (a bug in either one fails loudly).
    """

    def __init__(self, rng: random.Random, cfg: GeneratorConfig, difficulty: int):
        self.rng = rng
        self.cfg = cfg
        self.difficulty = difficulty
        self.ids: set = set()
        self.instructions: List[Instruction] = []
        self.persons: Dict[str, Tuple[str, str]] = {}   # office -> (name, title)
        self.used_names: set = set()
        self.facts: Dict[str, bool] = {}
        self.rules: List[Dict[str, str]] = []
        self.delegations: List[Dict[str, str]] = []
        self.notice_lines: List[str] = []
        self.planned: Dict[str, bool] = {}                # id -> followed?
        self.planned_conflicts: Dict[frozenset, str] = {}  # pair -> winner id
        self.mechanisms: Dict[frozenset, str] = {}        # pair -> intended deciding rule
        self.topics_with_conflict: set = set()

    # ---- small helpers ---------------------------------------------------

    def new_id(self) -> str:
        instr_id = f"INS-{self.rng.getrandbits(24):06X}"
        while instr_id in self.ids:
            instr_id = f"INS-{self.rng.getrandbits(24):06X}"
        self.ids.add(instr_id)
        return instr_id

    def person(self, office: str) -> Tuple[str, str]:
        if office not in self.persons:
            name = f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"
            while name in self.used_names:
                name = f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"
            self.used_names.add(name)
            if self.cfg.show_hierarchy:
                title = office
            else:
                title = self.rng.choice(split_pool(OFFICE_TITLES[office], self.cfg.split))
            self.persons[office] = (name, title)
        return self.persons[office]

    def pool(self, key: str) -> List[str]:
        functions = set(V2_TOPICS[key]["functions"]) | {"legal", "exec", "general"}
        return [o for o, fns in OFFICE_FUNCTIONS.items() if functions & set(fns)]

    def scope_values(self, key: str) -> List[str]:
        return DEPARTMENTS if V2_TOPICS[key]["scope_kind"] == "dept" else SYSTEMS

    def one_scope(self, key: str, avoid: Sequence[str] = ()) -> str:
        return self.rng.choice([v for v in self.scope_values(key) if v not in avoid])

    def overlapping_scopes(self, key: str) -> Tuple[str, str]:
        r = self.rng.random()
        x = self.one_scope(key)
        if r < 0.25:
            return "all", "all"
        if r < 0.5:
            return "all", x
        if r < 0.75:
            return x, "all"
        return x, x

    def dates(self, n: int) -> List[Optional[str]]:
        """n distinct ISO dates in ascending order (None each if dates are off)."""
        if not self.cfg.time_supersession:
            return [None] * n
        offsets = sorted(self.rng.sample(range(V2_DATE_SPAN_DAYS), n))
        return [(V2_DATE_START + datetime.timedelta(days=o)).isoformat() for o in offsets]

    def fact(self, want: Optional[bool]) -> Optional[str]:
        """A fact key whose value is `want` (None = random). Prefers facts not yet used."""
        keys = list(V2_FACTS)
        self.rng.shuffle(keys)
        for k in keys:
            if k not in self.facts:
                self.facts[k] = want if want is not None else self.rng.random() < 0.5
                return k
        for k in keys:
            if want is None or self.facts[k] == want:
                return k
        return None

    def scope_text(self, key: str, scope: str) -> str:
        phrases = SCOPE_PHRASES[V2_TOPICS[key]["scope_kind"]]
        if scope == "all":
            return self.rng.choice(phrases["all"])
        return self.rng.choice(phrases["one"]).format(x=scope)

    def fill(self, text: str) -> str:
        for slot, values in V2_FILL.items():
            placeholder = "{" + slot + "}"
            while placeholder in text:
                text = text.replace(placeholder, self.rng.choice(values), 1)
        return text

    def make(
        self,
        key: str,
        side: str,
        office: str,
        scope: str,
        date: Optional[str] = None,
        condition: Optional[str] = None,
        followed: bool = True,
    ) -> Instruction:
        topic = V2_TOPICS[key]
        value, texts = topic[side]
        core = self.rng.choice(split_pool(texts, self.cfg.split))
        core = self.fill(core.replace("{scope}", self.scope_text(key, scope)))
        if condition is not None:
            core = f"If {V2_FACTS[condition][0]}, {core[0].lower()}{core[1:]}"
        if self.rng.random() < 0.3:
            core += self.rng.choice(split_pool(V2_SIGNOFFS, self.cfg.split))
        name, title = self.person(office)
        ins = Instruction(
            id=self.new_id(), text=core, source=office, source_priority=SOURCE_PRIORITY[office],
            instruction_type=topic["type"], action_key=key, action_value=value,
            condition=condition, action_scope=scope, polarity=side,
            author=name, title=title, effective_date=date,
        )
        self.instructions.append(ins)
        self.planned[ins.id] = followed
        return ins

    # ---- conflicts -------------------------------------------------------

    def pick_mechanism(self) -> str:
        cfg = self.cfg
        weights = (
            ("scoped", cfg.p_scoped if cfg.scoped_authority else 0.0),
            ("delegation", cfg.p_delegation if cfg.delegation else 0.0),
            ("time", cfg.p_time if cfg.time_supersession else 0.0),
        )
        r, acc = self.rng.random(), 0.0
        for name, w in weights:
            acc += w
            if r < acc:
                return name
        return "tier"

    def roles(self, key: str, mechanism: str):
        """(winner_office, loser_office, mechanism, notice). Falls back to 'tier' if impossible."""
        rng = self.rng
        pool = self.pool(key)
        tier = SOURCE_PRIORITY.__getitem__
        by_tier: Dict[int, List[str]] = {}
        for o in pool:
            by_tier.setdefault(tier(o), []).append(o)

        if mechanism == "scoped":
            holders = [o for o in pool if 2 <= tier(o) <= 5]
            rng.shuffle(holders)
            for holder in holders:
                losers = [o for o in pool if o != holder and 2 <= tier(o) <= tier(holder)]
                if losers:
                    return holder, rng.choice(losers), "rule", ("rule", {"action_key": key, "office": holder})
        if mechanism == "delegation":
            delegators = [o for o in pool if tier(o) in (2, 3)]
            rng.shuffle(delegators)
            for frm in delegators:
                t = tier(frm)
                delegates = [o for o in pool if tier(o) > t]
                if not delegates:
                    continue
                to = rng.choice(delegates)
                notice = ("delegation", {"action_key": key, "from_office": frm, "to_office": to})
                if rng.random() < 0.65:
                    rivals = [o for o in pool if o not in (frm, to) and t < tier(o) <= tier(to)]
                    if rivals:
                        return to, rng.choice(rivals), "delegation", notice
                seniors = [o for o in pool if tier(o) < t]
                if seniors:
                    return rng.choice(seniors), to, "tier", notice
        if mechanism == "time":
            t = rng.choice(sorted(by_tier))
            offices = by_tier[t]
            if len(offices) >= 2 and rng.random() >= 0.35:
                w, l = rng.sample(offices, 2)
            else:
                w = l = rng.choice(offices)
            return w, l, "time", None
        t_win, t_lose = sorted(rng.sample(sorted(by_tier), 2))
        return rng.choice(by_tier[t_win]), rng.choice(by_tier[t_lose]), "tier", None

    def add_notice(self, notice) -> None:
        kind, data = notice
        label = V2_TOPICS[data["action_key"]]["label"]
        if kind == "rule":
            self.rules.append(data)
            name, title = self.person(data["office"])
            self.notice_lines.append(
                f"On {label}, {name} ({title}) has the final say over every other instruction, "
                f"whatever the sender's seniority."
            )
        else:
            self.delegations.append(data)
            fname, ftitle = self.person(data["from_office"])
            tname, ttitle = self.person(data["to_office"])
            self.notice_lines.append(
                f"{fname} ({ftitle}) has delegated their authority on {label} to {tname} ({ttitle})."
            )

    def conflict_cluster(self, key: str) -> None:
        rng, cfg = self.rng, self.cfg
        self.topics_with_conflict.add(key)
        win_side = "restrictive" if rng.random() < cfg.restrictive_win_rate else "permissive"
        lose_side = _other_side(win_side)
        w_office, l_office, how, notice = self.roles(key, self.pick_mechanism())
        if notice is not None:
            self.add_notice(notice)
        w_scope, l_scope = self.overlapping_scopes(key)
        early, late = self.dates(2)
        if how == "time" or rng.random() < 0.5:
            w_date, l_date = late, early
        else:
            w_date, l_date = early, late
        w_cond = l_cond = None
        if cfg.conditionals and rng.random() < cfg.p_conditional:
            k = self.fact(True)
            if rng.random() < 0.5:
                w_cond = k
            else:
                l_cond = k
        winner = self.make(key, win_side, w_office, w_scope, w_date, w_cond, followed=True)
        loser = self.make(key, lose_side, l_office, l_scope, l_date, l_cond, followed=False)
        pair = frozenset((winner.id, loser.id))
        self.planned_conflicts[pair] = winner.id
        self.mechanisms[pair] = how

        # Attached distractor: same topic, but it cannot clash with either side.
        if cfg.distractors and rng.random() < cfg.attached_distractor_prob:
            options = []
            if l_scope != "all":
                options.append((win_side, l_scope))   # agrees with the winner, misses the loser's scope
            if w_scope != "all":
                options.append((lose_side, w_scope))  # agrees with the loser, misses the winner's scope
            if options:
                side, avoid = rng.choice(options)
                self.make(key, side, rng.choice(self.pool(key)), self.one_scope(key, avoid=(avoid,)),
                          self.dates(1)[0])

        if cfg.chains and rng.random() < cfg.chain_prob:
            self.chain(winner if rng.random() < 0.5 else loser)

    # ---- distractors, fillers, chains -----------------------------------

    def distractor_cluster(self, key: str) -> None:
        rng, cfg = self.rng, self.cfg
        kinds = [("disjoint", 0.45), ("agree", 0.25), ("defused", 0.30 if cfg.conditionals else 0.0)]
        total = sum(w for _, w in kinds)
        r, acc, kind = rng.random() * total, 0.0, "disjoint"
        for name, w in kinds:
            acc += w
            if r < acc:
                kind = name
                break
        pool = self.pool(key)
        side = rng.choice(("restrictive", "permissive"))
        third = rng.random() < cfg.p_three_distractors
        d1, d2 = self.dates(2)
        if rng.random() < 0.5:
            d1, d2 = d2, d1

        def any_scope() -> str:
            return "all" if rng.random() < 0.4 else self.one_scope(key)

        if kind == "defused":
            k = self.fact(False)
            if k is None:
                kind = "disjoint"
            else:
                # The void instruction is usually the more senior one, so a pure
                # seniority lookup "wins" with an instruction that is not in force.
                tiers = sorted({SOURCE_PRIORITY[o] for o in pool})
                t_hi, t_lo = sorted(rng.sample(tiers, 2))
                senior = rng.choice([o for o in pool if SOURCE_PRIORITY[o] == t_hi])
                junior = rng.choice([o for o in pool if SOURCE_PRIORITY[o] == t_lo])
                void_office, live_office = (senior, junior) if rng.random() < 0.7 else (junior, senior)
                s1, s2 = self.overlapping_scopes(key)
                self.make(key, side, void_office, s1, d1, condition=k, followed=False)
                self.make(key, _other_side(side), live_office, s2, d2, followed=True)
                if third:  # agrees with the live one; the void one clashes with nobody
                    self.make(key, _other_side(side), rng.choice(pool), any_scope(), self.dates(1)[0])
                return
        if kind == "disjoint":
            # Opposite directions, but every scope is a different department/system.
            scopes = rng.sample(self.scope_values(key), 3 if third else 2)
            sides = [side, _other_side(side), rng.choice((side, _other_side(side)))]
            for scope, s in zip(scopes, sides):
                self.make(key, s, rng.choice(pool), scope, self.dates(1)[0])
            return
        # agree: same direction, any scopes
        for office in rng.sample(pool, 3 if third else 2):
            self.make(key, side, office, any_scope(), self.dates(1)[0])

    def filler(self, key: str) -> None:
        rng, cfg = self.rng, self.cfg
        side = rng.choice(("restrictive", "permissive"))
        scope = "all" if rng.random() < 0.5 else self.one_scope(key)
        cond = None
        if cfg.conditionals and rng.random() < 0.25:
            cond = self.fact(None)
        followed = cond is None or self.facts[cond]
        self.make(key, side, rng.choice(self.pool(key)), scope, self.dates(1)[0], cond, followed=followed)

    def chain(self, parent: Instruction) -> None:
        action = self.rng.choice(split_pool(V2_FOLLOWUPS, self.cfg.split))
        office = self.rng.choice(self.pool(parent.action_key))
        name, title = self.person(office)
        ins = Instruction(
            id=self.new_id(),
            text=f"Contingent on [{parent.id}] going ahead: {action[0].upper()}{action[1:]}.",
            source=office, source_priority=SOURCE_PRIORITY[office], instruction_type="absolute",
            action_key="follow_up", action_value="carry_out", action_scope="all",
            author=name, title=title, effective_date=self.dates(1)[0], depends_on=parent.id,
        )
        self.instructions.append(ins)
        self.planned[ins.id] = self.planned[parent.id]

    def decoy_notices(self) -> None:
        """Rules/delegations on topics without a real conflict: they must change nothing."""
        cfg, rng = self.cfg, self.rng
        kinds = [k for k, on in (("rule", cfg.scoped_authority), ("delegation", cfg.delegation)) if on]
        if not (cfg.decoy_notices and kinds):
            return
        free = [k for k in V2_TOPICS if k not in self.topics_with_conflict]
        rng.shuffle(free)
        n_conflicts = len(self.topics_with_conflict)
        n = min(len(free), rng.randint(1, max(1, n_conflicts // 2)))
        used_offices = sorted({ins.source for ins in self.instructions})
        for key in free[:n]:
            kind = rng.choice(kinds)
            # Prefer people who already sent instructions on other topics: a notice
            # about the wrong topic must not change their authority.
            if kind == "rule":
                candidates = [o for o in used_offices if SOURCE_PRIORITY[o] >= 2] or self.pool(key)
                self.add_notice(("rule", {"action_key": key, "office": rng.choice(candidates)}))
            else:
                frm = rng.choice([o for o in OFFICE_FUNCTIONS if SOURCE_PRIORITY[o] in (2, 3)])
                t = SOURCE_PRIORITY[frm]
                candidates = [o for o in used_offices if SOURCE_PRIORITY[o] > t] or \
                             [o for o in OFFICE_FUNCTIONS if SOURCE_PRIORITY[o] > t]
                self.add_notice(("delegation", {"action_key": key, "from_office": frm,
                                                "to_office": rng.choice(candidates)}))

    def status_lines(self) -> List[str]:
        if self.cfg.conditionals and self.rng.random() < 0.5:
            self.fact(None)  # a status fact no instruction depends on
        keys = sorted(self.facts)
        self.rng.shuffle(keys)
        return [V2_FACTS[k][1] if self.facts[k] else V2_FACTS[k][2] for k in keys]


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """
    Generates fully dynamic instruction conflict scenarios.
    No two generated scenarios are identical.

    Each generator owns a private RNG (``self.rng``), so generators never interfere
    with each other or with the global ``random`` module. ``generate(..., seed=s)``
    gives a fully reproducible scenario regardless of the generator's own state.

    ``config`` picks the task: None = ``GeneratorConfig()`` (the Phase 3 task);
    ``GeneratorConfig(legacy=True)`` = the original generator, unchanged.
    """

    def __init__(self, seed: Optional[int] = None, config: Optional[GeneratorConfig] = None):
        self.rng = random.Random(seed)
        self.config = config if config is not None else GeneratorConfig()

    def _fill(self, text: str, rng: Optional[random.Random] = None) -> str:
        """Fill template slots with randomly chosen values."""
        rng = rng or self.rng
        for key, values in FILL_PARAMS.items():
            placeholder = "{" + key + "}"
            while placeholder in text:
                text = text.replace(placeholder, rng.choice(values), 1)
        return text

    def _new_id(self, existing_ids: set, rng: random.Random) -> str:
        """Draw a fresh instruction ID (INS- + 6 hex chars) from the RNG."""
        instr_id = f"INS-{rng.getrandbits(24):06X}"
        while instr_id in existing_ids:
            instr_id = f"INS-{rng.getrandbits(24):06X}"
        return instr_id

    def _make_instruction(
        self,
        group_key: str,
        template_entry: dict,
        existing_ids: set,
        rng: Optional[random.Random] = None,
        source: Optional[str] = None,
    ) -> Instruction:
        """Construct a single Instruction from a template entry."""
        rng = rng or self.rng
        instr_id = self._new_id(existing_ids, rng)
        existing_ids.add(instr_id)

        if source is None:
            source = rng.choice(template_entry["sources"])
        raw_text = rng.choice(template_entry["texts"])
        text = self._fill(raw_text, rng)

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
        self,
        group_key: str,
        existing_ids: set,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Instruction, Instruction, ConflictPair]:
        """
        Generate one conflict pair from a template group.
        Always picks two templates with DIFFERENT action_values and
        DIFFERENT source priorities (no ties — every conflict has a clear winner).
        Returns (winner_instruction, loser_instruction, conflict_pair).
        """
        rng = rng or self.rng
        group = TEMPLATE_GROUPS[group_key]
        templates = group["templates"]

        # Pick two templates from different action_value buckets
        t_a = templates[0]
        t_b = templates[1]

        instr_a = self._make_instruction(group_key, t_a, existing_ids, rng)

        # Choose the second source only from sources in a different authority tier,
        # so a tie is impossible by construction (no retry loop needed).
        tier_a = instr_a.source_priority
        sources_b = [s for s in t_b["sources"] if SOURCE_PRIORITY[s] != tier_a]
        assert sources_b, (
            f"{group_key}: no source in the second template differs in tier from "
            f"{instr_a.source!r} (tier {tier_a}); a conflict would be a tie"
        )
        instr_b = self._make_instruction(
            group_key, t_b, existing_ids, rng, source=rng.choice(sources_b)
        )

        # Resolve: strictly lower priority number = higher authority = wins
        if instr_a.source_priority < instr_b.source_priority:
            winner, loser = instr_a, instr_b
        else:
            winner, loser = instr_b, instr_a
        assert winner.source_priority < loser.source_priority, (
            f"{group_key}: winner tier {winner.source_priority} is not strictly "
            f"higher authority than loser tier {loser.source_priority}"
        )

        explanation = (
            f"{winner.source} (authority level {winner.source_priority}) overrides "
            f"{loser.source} (authority level {loser.source_priority}) on {group_key.replace('_', ' ')}."
        )

        conflict = ConflictPair(
            instruction_a_id=instr_a.id,
            instruction_b_id=instr_b.id,
            conflict_type=CONFLICT_TYPE_LABELS[instr_a.instruction_type],
            resolution_id=winner.id,
            explanation=explanation,
        )

        return winner, loser, conflict

    def _generate_filler_instructions(
        self,
        n: int,
        existing_ids: set,
        existing_keys: set,
        rng: Optional[random.Random] = None,
    ) -> List[Instruction]:
        """
        Generate n non-conflicting filler instructions to pad the scenario.
        These are always 'follow' instructions with no opposing directive.
        """
        rng = rng or self.rng
        fillers = []
        available_groups = [k for k in TEMPLATE_GROUPS if k not in existing_keys]
        rng.shuffle(available_groups)

        for i in range(min(n, len(available_groups))):
            group_key = available_groups[i]
            group = TEMPLATE_GROUPS[group_key]
            # Pick either template but only one per group (no conflict)
            t = rng.choice(group["templates"])
            instr = self._make_instruction(group_key, t, existing_ids, rng)
            existing_ids.add(instr.id)
            fillers.append(instr)

        return fillers

    def _format_prompt(
        self,
        context: str,
        instructions: List[Instruction],
        rng: Optional[random.Random] = None,
        shuffle: bool = True,
    ) -> str:
        """
        Format the full LLM prompt for the scenario.

        The caller's list is never modified: presentation order is randomised on a
        copy (unless shuffle=False, when ``instructions`` is used in the given order).
        """
        rng = rng or self.rng
        ordered = list(instructions)
        if shuffle:
            rng.shuffle(ordered)  # randomize presentation order
        conflict_types = "|".join(PROMPT_CONFLICT_TYPES)

        instruction_lines = []
        for ins in ordered:
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
{{"identified_conflicts":[{{"instruction_a":"<ID>","instruction_b":"<ID>","conflict_type":"{conflict_types}","resolution":"<winning ID>","reasoning":"<one sentence>"}}],"execution_plan":["<ID>",...],"overridden_instructions":["<ID>",...]}}"""
        return prompt

    def generate(
        self,
        difficulty: int = 2,
        domain: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Scenario:
        """
        Generate one complete scenario.

        Args:
            difficulty: legacy: 1/2/3 = 2/4/6 conflicts. v2: 1..MAX_DIFFICULTY
                (see DIFFICULTY_LEVELS; 1-3 keep 2/4/6 conflicts). Values outside
                the valid range are clamped.
            domain: "HR" | "Finance" | "IT" | "Operations" | None (random)
            seed: optional per-call seed. When given, this call uses only
                ``random.Random(seed)`` (the generator's own RNG is neither read
                nor advanced), so ``generate(d, seed=s)`` is identical across
                generator instances. Use it for fixed evaluation sets.

        Returns:
            A fully populated Scenario object.
        """
        rng = random.Random(seed) if seed is not None else self.rng
        if not self.config.legacy:
            return self._generate_v2(difficulty, domain, rng)

        difficulty = clamp_difficulty(difficulty, 3)
        num_conflicts = {1: 2, 2: 4, 3: 6}[difficulty]

        if domain is None:
            domain = rng.choice(list(BUSINESS_CONTEXTS.keys()))

        # Select business context
        context_template = rng.choice(BUSINESS_CONTEXTS[domain])
        context = self._fill(context_template, rng)

        # Select template groups for conflicts (no repetition)
        all_groups = list(TEMPLATE_GROUPS.keys())
        rng.shuffle(all_groups)
        conflict_groups = all_groups[:num_conflicts]

        existing_ids: set = set()
        all_instructions: List[Instruction] = []
        all_conflicts: List[ConflictPair] = []
        ground_truth_followed: List[str] = []
        ground_truth_overridden: List[str] = []

        for group_key in conflict_groups:
            winner, loser, conflict_pair = self._generate_conflict_pair(group_key, existing_ids, rng)
            all_instructions.append(winner)
            all_instructions.append(loser)
            all_conflicts.append(conflict_pair)
            ground_truth_followed.append(winner.id)
            ground_truth_overridden.append(loser.id)

        # Add non-conflicting filler instructions (makes scenario realistic)
        used_keys = set(conflict_groups)
        filler_count = rng.randint(2, 4)
        fillers = self._generate_filler_instructions(filler_count, existing_ids, used_keys, rng)
        all_instructions.extend(fillers)
        ground_truth_followed.extend([f.id for f in fillers])

        # Randomise presentation order once; Scenario.instructions keeps the order
        # shown in the prompt.
        all_instructions = list(all_instructions)
        rng.shuffle(all_instructions)
        prompt = self._format_prompt(context, all_instructions, rng, shuffle=False)

        return Scenario(
            scenario_id=str(uuid.UUID(int=rng.getrandbits(128))),
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
            d = difficulty if difficulty is not None else self.rng.randint(1, 3)
            scenarios.append(self.generate(difficulty=d, domain=domain))
        return scenarios

    # ------------------------------------------------------------------
    # v2 (Phase 3) generation
    # ------------------------------------------------------------------

    def _generate_v2(self, difficulty: Any, domain: Optional[str], rng: random.Random) -> Scenario:
        from resolver import resolve  # resolver imports this module; import lazily

        cfg = self.config
        difficulty = clamp_difficulty(difficulty)
        level = DIFFICULTY_LEVELS[difficulty]

        if domain is None:
            domain = rng.choice(list(V2_SITUATIONS))
        company = rng.choice(split_pool(V2_COMPANIES, cfg.split))
        situation = rng.choice(split_pool(V2_SITUATIONS[domain], cfg.split))
        context = f"{company} is {situation} Several leaders have sent instructions this month."

        n_conflicts = level["conflicts"]
        n_distractors = int(n_conflicts * cfg.distractor_ratio + 0.5) if cfg.distractors else 0
        topics = list(V2_TOPICS)
        rng.shuffle(topics)
        conflict_topics = topics[:n_conflicts]
        rest = topics[n_conflicts:]
        distractor_topics = rest[:n_distractors]
        rest = rest[n_distractors:]
        filler_topics = rest[: min(len(rest), rng.randint(*level["fillers"]))]

        b = _V2Builder(rng, cfg, difficulty)
        for key in conflict_topics:
            b.conflict_cluster(key)
        for key in distractor_topics:
            b.distractor_cluster(key)
        for key in filler_topics:
            b.filler(key)
        b.decoy_notices()
        status = b.status_lines()
        notices = list(b.notice_lines)
        rng.shuffle(notices)

        instructions = list(b.instructions)
        rng.shuffle(instructions)
        sc = Scenario(
            scenario_id=str(uuid.UUID(int=rng.getrandbits(128))),
            domain=domain,
            difficulty=difficulty,
            business_context=context,
            instructions=instructions,
            conflicts=[],
            ground_truth_followed=[],
            ground_truth_overridden=[],
            prompt="",
            facts=dict(sorted(b.facts.items())),
            authority_rules=list(b.rules),
            delegations=list(b.delegations),
            split=cfg.split,
            generator_version="v2",
        )
        followed, overridden, conflicts = resolve(sc)
        self._check_against_plan(b, followed, overridden, conflicts)
        sc.ground_truth_followed = followed
        sc.ground_truth_overridden = overridden
        sc.conflicts = conflicts
        sc.prompt = self._format_prompt_v2(context, status, notices, instructions)
        return sc

    @staticmethod
    def _check_against_plan(b: "_V2Builder", followed, overridden, conflicts) -> None:
        """The builder's intent and the resolver must agree exactly; anything else is a bug."""
        planned_f = sorted(i for i, f in b.planned.items() if f)
        planned_o = sorted(i for i, f in b.planned.items() if not f)
        got = {frozenset((c.instruction_a_id, c.instruction_b_id)): c.resolution_id for c in conflicts}
        if followed != planned_f or overridden != planned_o or got != b.planned_conflicts:
            raise AssertionError(
                "v2 generator and resolver disagree:\n"
                f"  planned followed {planned_f}\n  resolved followed {followed}\n"
                f"  planned conflicts {b.planned_conflicts}\n  resolved conflicts {got}"
            )
        seen: Dict[str, int] = {}
        for c in conflicts:
            for i in (c.instruction_a_id, c.instruction_b_id):
                seen[i] = seen.get(i, 0) + 1
        if any(n > 1 for n in seen.values()):
            raise AssertionError(f"an instruction is in more than one conflict: {seen}")

    def _format_prompt_v2(
        self,
        context: str,
        status: List[str],
        notices: List[str],
        instructions: List[Instruction],
    ) -> str:
        """The v2 prompt. Instructions are shown in the given order (already shuffled)."""
        cfg = self.config
        lines = [
            "You are a business operations coordinator. Several people have sent instructions. "
            "Decide which ones to carry out.",
            "",
            "AUTHORITY:",
        ]
        if cfg.show_hierarchy:
            lines.append("Tiers (lower number = higher authority):")
            lines += _tier_list_lines()
            lines.append("- An instruction from a higher tier beats one from a lower tier.")
        else:
            lines.append("- Legal, compliance and regulatory roles outrank everyone else, including the CEO.")
            lines.append(
                "- Otherwise seniority decides: C-suite (CEO, President, Chief ... Officer) > vice presidents "
                "(VP, SVP, EVP) > directors > managers > team leads and coordinators. "
                "Titles at the same level have equal authority."
            )
        if cfg.scoped_authority or cfg.delegation:
            lines.append(
                "- STANDING NOTICES can give one person the final say on one topic, or pass a senior "
                "person's authority on one topic to someone else. A notice applies only to the topic it names."
            )
        if cfg.time_supersession:
            lines.append(
                "- Between instructions of equal authority, the later date wins. "
                "A later date never beats higher authority."
            )
        lines += [
            "",
            "CONFLICTS:",
            "- Two instructions conflict only if they give different directions on the same topic and "
            "their scopes overlap. Company-wide (or all systems) overlaps everything; two different "
            "departments or two different systems do not overlap. Instructions that agree never conflict.",
            "- The instruction that loses a conflict is overridden.",
        ]
        if cfg.conditionals:
            lines.append(
                '- An instruction that starts with "If" applies only when its condition is true in '
                "CURRENT STATUS. If the condition is false, the instruction is void: override it and do "
                "not list it in any conflict."
            )
        if cfg.chains:
            lines.append(
                "- An instruction contingent on another one is cancelled (overridden) when that other "
                "instruction is overridden."
            )
        lines += ["", f"CONTEXT: {context}"]
        if status:
            lines += ["", "CURRENT STATUS:"] + [f"- {s}" for s in status]
        if notices:
            lines += ["", "STANDING NOTICES:"] + [f"- {n}" for n in notices]
        lines += ["", "INSTRUCTIONS:"]
        blocks = []
        for ins in instructions:
            dated = f", {format_date(ins.effective_date)}" if ins.effective_date else ""
            blocks.append(f"[{ins.id}] From {ins.author} ({ins.title}){dated}:\n  {ins.text}")
        lines.append("\n\n".join(blocks))
        conflict_types = "|".join(V2_CONFLICT_TYPES)
        lines += [
            "",
            'Output ONLY a JSON object. No preamble, no explanation outside the JSON. Keep "reasoning" to '
            "one short sentence. execution_plan = every instruction to carry out; overridden_instructions "
            "= every other instruction.",
            '{"identified_conflicts":[{"instruction_a":"<ID>","instruction_b":"<ID>","conflict_type":"'
            + conflict_types
            + '","resolution":"<winning ID>","reasoning":"<one sentence>"}],'
            '"execution_plan":["<ID>",...],"overridden_instructions":["<ID>",...]}',
        ]
        return "\n".join(lines)
