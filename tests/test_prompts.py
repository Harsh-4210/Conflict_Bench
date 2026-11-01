"""Tests for the shared prompt module."""

import re
from pathlib import Path

import pytest

from generator import ScenarioGenerator
import prompts

ROOT = Path(__file__).resolve().parent.parent


def test_build_messages_matches_scenario_prompt_exactly():
    for seed in range(5):
        for diff in (1, 2, 3):
            sc = ScenarioGenerator(seed=seed).generate(difficulty=diff)
            msgs = prompts.build_messages(sc)
            assert [m["role"] for m in msgs] == ["system", "user"]
            assert msgs[0]["content"] == prompts.SYSTEM_PROMPT
            assert msgs[1]["content"] == sc.prompt


def test_system_prompt_is_json_only_no_thought_block():
    assert "<thought>" not in prompts.SYSTEM_PROMPT
    assert "Output ONLY valid JSON" in prompts.SYSTEM_PROMPT


def test_system_prompt_matches_training_script_if_still_inline():
    """
    The legacy training script still inlines the pre-Phase-3 system prompt (with the
    tier list). It must not drift from prompts.LEGACY_SYSTEM_PROMPT until Phase 5
    switches training to prompts.build_messages.
    """
    src = (ROOT / "train_grpo.py").read_text(encoding="utf-8")
    m = re.search(r'SYSTEM_PROMPT = """(.*?)"""', src, re.S)
    if m is None:
        pytest.skip("train_grpo.py no longer defines SYSTEM_PROMPT inline")
    assert m.group(1) == prompts.LEGACY_SYSTEM_PROMPT


def test_system_prompt_does_not_leak_the_hierarchy():
    """Hidden-hierarchy scenarios (P3-3) would be pointless if the system prompt printed the tiers."""
    text = prompts.SYSTEM_PROMPT
    assert not re.search(r"^\s*[1-6]\. ", text, re.M)
    for office in ("Legal & Compliance", "Regulatory Affairs", "VP Engineering", "Team Lead", "hierarchy"):
        assert office not in text
    sc = ScenarioGenerator(seed=1).generate(2)  # default: hidden hierarchy
    joined = "\n".join(m["content"] for m in prompts.build_messages(sc))
    assert not re.search(r"^\s*[1-6]\. ", joined, re.M)


def test_system_prompt_offers_the_v2_conflict_types():
    from generator import V2_CONFLICT_TYPES
    assert '"conflict_type":"' + "|".join(V2_CONFLICT_TYPES) + '"' in prompts.SYSTEM_PROMPT


def test_inference_uses_shared_prompt_builder():
    import inference
    assert inference.build_messages is prompts.build_messages
    assert inference.DEFAULT_ADAPTER == "Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora"
