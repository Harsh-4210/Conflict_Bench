"""Tests for multistep/tools.py: the tool-call parser never crashes and says what is wrong."""

import json

import pytest

from multistep.tools import ActionError, ToolCall, parse_action


def test_valid_calls_from_dicts():
    assert parse_action({"tool": "apply", "id": "INS-1"}) == ToolCall("apply", id="INS-1")
    assert parse_action({"tool": "revert", "id": " INS-1 "}) == ToolCall("revert", id="INS-1")
    assert parse_action({"tool": "reject", "id": "INS-1", "reason": "outranked"}).reason == "outranked"
    assert parse_action({"tool": "reject", "id": "INS-1"}).reason == ""
    assert parse_action({"tool": "escalate", "ids": ["A", "B"]}).ids == ("A", "B")
    assert parse_action({"tool": "wait"}).tool == "wait"
    assert parse_action({"tool": "finish"}).tool == "finish"


def test_valid_calls_from_text():
    assert parse_action('{"tool": "wait"}').tool == "wait"
    fenced = '```json\n{"tool": "apply", "id": "INS-9"}\n```'
    assert parse_action(fenced).id == "INS-9"
    chatty = 'I will apply it now. {"tool": "apply", "id": "INS-9"} Thanks!'
    assert parse_action(chatty).id == "INS-9"
    with_brace_first = 'Plan {not json} then {"tool": "finish"}'
    assert parse_action(with_brace_first).tool == "finish"
    assert parse_action(json.dumps({"tool": "wait"}).encode()).tool == "wait"


@pytest.mark.parametrize("bad", [
    None, 5, 1.5, True, [], [1, 2], {}, "", "   ", "{", "}{", "not json at all", b"\xff\xfe",
    {"tool": None}, {"tool": 5}, {"tool": ["apply"]}, {"tool": {"a": 1}}, {"tool": "delete"},
    {"tool": "APPLY", "id": "x"},
    {"tool": "apply"}, {"tool": "apply", "id": ""}, {"tool": "apply", "id": 5},
    {"tool": "apply", "id": ["INS-1"]}, {"tool": "apply", "id": {"a": 1}}, {"tool": "revert", "id": None},
    {"tool": "reject", "id": "A", "reason": 5}, {"tool": "reject", "id": "A", "reason": ["x"]},
    {"tool": "escalate"}, {"tool": "escalate", "ids": []}, {"tool": "escalate", "ids": "A"},
    {"tool": "escalate", "ids": [1, 2]}, {"tool": "escalate", "ids": [["A"]]},
    {"tool": "escalate", "ids": ["A", "A"]}, {"tool": "escalate", "ids": ["A"] * 3},
    {"tool": "escalate", "ids": [str(i) for i in range(51)]},
    "x" * 30_000,
])
def test_bad_input_raises_action_error_only(bad):
    with pytest.raises(ActionError) as exc:
        parse_action(bad)
    assert str(exc.value)  # an actual message the agent can read


def test_long_reason_is_clipped_not_rejected():
    call = parse_action({"tool": "reject", "id": "A", "reason": "x" * 5000})
    assert len(call.reason) == 500
