"""
Shared prompt construction for ConflictBench.

Single source of truth for the system prompt and the chat messages. The demo
app, the inference script and (from Phase 5) the training scripts must all
build their model input through ``build_messages`` so that training and
evaluation see identical text.
"""

from typing import Dict, List

SYSTEM_PROMPT = """You are an expert business operations coordinator.
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
