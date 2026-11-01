"""
ConflictBench multi-step: the tool-call language.

An action is one JSON object with a ``tool`` key:

    {"tool": "apply",    "id": "INS-1A2B3C"}
    {"tool": "reject",   "id": "INS-1A2B3C", "reason": "outranked by the CFO"}
    {"tool": "revert",   "id": "INS-1A2B3C"}
    {"tool": "escalate", "ids": ["INS-1A2B3C", "INS-4D5E6F"]}
    {"tool": "wait"}
    {"tool": "finish"}

``parse_action`` turns a dict or a string (bare JSON, a fenced block, or JSON inside
other text) into a ``ToolCall`` or raises ``ActionError`` with a message the agent can
read. It never raises anything else, whatever the input.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

TOOLS: List[Dict[str, Any]] = [
    {"tool": "apply", "args": {"id": "instruction ID"},
     "does": "Carry out an instruction that has arrived. Writes its value to the company state."},
    {"tool": "reject", "args": {"id": "instruction ID", "reason": "optional short text"},
     "does": "Record that you will not carry out an instruction that has arrived. Does not change the state."},
    {"tool": "revert", "args": {"id": "instruction ID"},
     "does": "Undo an instruction you applied earlier. The state goes back to what it would be without it."},
    {"tool": "escalate", "args": {"ids": "list of instruction IDs"},
     "does": "Flag arrived instructions for a human and leave them out of the state for now."},
    {"tool": "wait", "args": {}, "does": "Read the next messages. Uses a step."},
    {"tool": "finish", "args": {}, "does": "End the episode. The company state at that moment is scored."},
]
TOOL_NAMES = tuple(t["tool"] for t in TOOLS)

MAX_ACTION_CHARS = 20_000
MAX_REASON_CHARS = 500
MAX_ESCALATE_IDS = 50


class ActionError(ValueError):
    """The action could not be understood. The message is shown to the agent."""


@dataclass(frozen=True)
class ToolCall:
    tool: str
    id: Optional[str] = None
    ids: Optional[tuple] = None
    reason: str = ""


def _extract_json(text: str) -> Any:
    """First JSON value found in ``text``: the whole string, else the first parsable '{...}'."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    pos = stripped.find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(stripped, pos)
            return obj
        except ValueError:
            pos = stripped.find("{", pos + 1)
    raise ActionError("Action is not valid JSON. Send one JSON object such as {\"tool\": \"wait\"}.")


def _need_id(obj: Dict[str, Any], key: str = "id") -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ActionError(f"'{key}' must be a non-empty string instruction ID.")
    return value.strip()


def parse_action(action: Any) -> ToolCall:
    """Validate one action. Raises ActionError (and nothing else) if it is not a well-formed tool call."""
    try:
        if isinstance(action, (bytes, bytearray)):
            action = action.decode("utf-8", "replace")
        if isinstance(action, str):
            if len(action) > MAX_ACTION_CHARS:
                raise ActionError(f"Action is longer than {MAX_ACTION_CHARS} characters.")
            action = _extract_json(action)
        if not isinstance(action, dict):
            raise ActionError("Action must be a JSON object with a 'tool' key.")
        tool = action.get("tool")
        if not isinstance(tool, str) or tool not in TOOL_NAMES:
            raise ActionError(f"Unknown tool {tool!r}. Available tools: {', '.join(TOOL_NAMES)}.")
        if tool in ("apply", "revert"):
            return ToolCall(tool, id=_need_id(action))
        if tool == "reject":
            reason = action.get("reason", "")
            if reason is None:
                reason = ""
            if not isinstance(reason, str):
                raise ActionError("'reason' must be a string.")
            return ToolCall(tool, id=_need_id(action), reason=reason[:MAX_REASON_CHARS])
        if tool == "escalate":
            ids = action.get("ids")
            if not isinstance(ids, list) or not ids:
                raise ActionError("'ids' must be a non-empty list of instruction IDs.")
            if len(ids) > MAX_ESCALATE_IDS:
                raise ActionError(f"'ids' may hold at most {MAX_ESCALATE_IDS} IDs.")
            if not all(isinstance(i, str) and i.strip() for i in ids):
                raise ActionError("Every entry in 'ids' must be a non-empty string.")
            cleaned = tuple(i.strip() for i in ids)
            if len(set(cleaned)) != len(cleaned):
                raise ActionError("'ids' lists the same ID twice.")
            return ToolCall(tool, ids=cleaned)
        return ToolCall(tool)  # wait, finish
    except ActionError:
        raise
    except Exception as exc:  # pragma: no cover - belt and braces: never crash the server
        raise ActionError(f"Could not read the action ({type(exc).__name__}).") from exc
