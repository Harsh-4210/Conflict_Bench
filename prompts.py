"""
Shared prompt construction for ConflictBench.

Single source of truth for the system prompt and the chat messages. The demo
app, the inference script and (from Phase 5) the training scripts must all
build their model input through ``build_messages`` so that training and
evaluation see identical text.

The system prompt deliberately contains NO authority hierarchy: each scenario's
own prompt decides whether the tier list is shown (GeneratorConfig.show_hierarchy).
Printing it here would leak the hierarchy into hidden-hierarchy scenarios.
"""

from typing import Dict, List

SYSTEM_PROMPT = """You are an expert business operations coordinator.
Your task: given business instructions from several people, find every real conflict, resolve each one using only the authority rules, notices and status given in the task, and produce an execution plan.

Rules:
- List every conflicting pair, and only pairs that really conflict
- execution_plan must contain every instruction that should be carried out
- overridden_instructions must contain every instruction that should not be carried out
- No instruction may appear in both lists

Output ONLY valid JSON matching this exact schema. No preamble, no explanation outside the JSON:
{"identified_conflicts":[{"instruction_a":"<ID>","instruction_b":"<ID>","conflict_type":"direct|resource|temporal|conditional","resolution":"<winning ID>","reasoning":"<one sentence>"}],"execution_plan":["<ID>",...],"overridden_instructions":["<ID>",...]}"""

# The pre-Phase-3 system prompt. It prints the authority hierarchy, so it must
# not be used with hidden-hierarchy scenarios. Kept only because the legacy
# training scripts still inline this exact text (tests check it does not drift).
LEGACY_SYSTEM_PROMPT = """You are an expert business operations coordinator.
Your task: given a set of business instructions from various stakeholders, identify ALL conflicts and produce a resolution plan.

Authority hierarchy (lower number = higher authority — ALWAYS wins):
1. Legal & Compliance, Regulatory Affairs
2. CEO Office, CFO, CTO, COO
3. VP Engineering, VP Finance, VP Operations, VP Human Resources
4. Director of IT, Director of Finance
5. Engineering Manager, Finance Manager, HR Manager, IT Manager
6. Team Lead, Department Coordinator

Rules:
- When two instructions conflict, the HIGHER authority (lower tier number) always wins
- List ALL conflicting pairs — do not miss any
- Your execution_plan must contain ONLY the winning instructions + non-conflicting ones
- overridden_instructions must contain ONLY the losing instructions

Output ONLY valid JSON matching this exact schema. No preamble, no explanation outside the JSON:
{"identified_conflicts":[{"instruction_a":"<ID>","instruction_b":"<ID>","conflict_type":"direct|resource|temporal|absolute","resolution":"<winning ID>","reasoning":"<one sentence>"}],"execution_plan":["<ID>",...],"overridden_instructions":["<ID>",...]}"""


def build_messages(scenario) -> List[Dict[str, str]]:
    """Chat messages for one scenario: the system prompt plus the scenario prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario.prompt},
    ]
