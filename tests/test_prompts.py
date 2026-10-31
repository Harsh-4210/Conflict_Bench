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
    """Until the training scripts switch to prompts.py (Phase 5), the text must not drift."""
    src = (ROOT / "train_grpo.py").read_text(encoding="utf-8")
    m = re.search(r'SYSTEM_PROMPT = """(.*?)"""', src, re.S)
    if m is None:
        pytest.skip("train_grpo.py no longer defines SYSTEM_PROMPT inline")
    assert m.group(1) == prompts.SYSTEM_PROMPT


def test_inference_uses_shared_prompt_builder():
    import inference
    assert inference.build_messages is prompts.build_messages
    assert inference.DEFAULT_ADAPTER == "Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora"
