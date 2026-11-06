"""Explicit evaluation identity, with conservative support for historical summaries."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def evaluation_provenance(
    task: str, task_version: str, scorer_files: Mapping[str, Path],
    protocol: Mapping[str, Any], scenario_ids: Iterable[str], *,
    scored_scenario_ids: Optional[Iterable[str]] = None,
) -> dict:
    """Fingerprint scorer sources, public experiment settings and ordered selection.

    Never pass API credentials or complete backend configuration as protocol settings.
    Task identity is separate from the actual dataset file hash.
    """
    sources = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
               for name, path in sorted(scorer_files.items())}
    selected = list(scenario_ids)
    scored = list(scored_scenario_ids) if scored_scenario_ids is not None else selected
    return {
        "evaluation_schema_version": 2,
        "task": task,
        "task_version": task_version,
        "scorer_files": sources,
        "scorer_sha256": _digest(sources),
        "protocol": dict(protocol),
        "protocol_sha256": _digest(dict(protocol)),
        "selection_sha256": _digest(selected),
        "scored_selection_sha256": _digest(scored),
    }


def real_data_sha256(summary: Mapping[str, Any]) -> str:
    """Read old multistep-prefixed keys without presenting the prefix as a file hash."""
    value = summary.get("data_file_sha256") or summary.get("data_sha256") or "unknown"
    return value.removeprefix("multistep-")


def comparison_key(summary: Mapping[str, Any]) -> Tuple[str, str, str, str, str, str, str]:
    """Missing provenance is unknown, never silently equivalent to a newer protocol."""
    task = summary.get("task")
    if not task:
        task = "multistep" if str(summary.get("data_sha256", "")).startswith("multistep-") else "single_step"
    return (
        real_data_sha256(summary), task, summary.get("task_version") or "unversioned",
        summary.get("scorer_sha256") or "unversioned",
        summary.get("protocol_sha256") or "unversioned",
        summary.get("selection_sha256") or "unversioned",
        summary.get("scored_selection_sha256") or "unversioned",
    )
