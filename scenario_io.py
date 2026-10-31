"""
ConflictBench — Scenario serialisation
One JSON format for scenarios, shared by the training reward, the eval harness
and the fixed test sets in data/. Works field by field from the dataclasses, so
new fields added to Instruction/ConflictPair/Scenario (with defaults) round-trip
without changes here; unknown keys in old files are ignored.
"""

import dataclasses
import json
from pathlib import Path
from typing import Iterable, List, Union

from generator import ConflictPair, Instruction, Scenario


def _known(cls, data: dict) -> dict:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def scenario_to_dict(scenario: Scenario) -> dict:
    return dataclasses.asdict(scenario)


def scenario_from_dict(data: dict) -> Scenario:
    fields = _known(Scenario, data)
    fields["instructions"] = [Instruction(**_known(Instruction, d)) for d in data["instructions"]]
    fields["conflicts"] = [ConflictPair(**_known(ConflictPair, d)) for d in data["conflicts"]]
    return Scenario(**fields)


def dumps_scenario(scenario: Scenario) -> str:
    return json.dumps(scenario_to_dict(scenario), separators=(",", ":"))


def loads_scenario(text: str) -> Scenario:
    return scenario_from_dict(json.loads(text))


def save_scenarios(path: Union[str, Path], scenarios: Iterable[Scenario]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sc in scenarios:
            f.write(dumps_scenario(sc) + "\n")


def load_scenarios(path: Union[str, Path]) -> List[Scenario]:
    with Path(path).open(encoding="utf-8") as f:
        return [loads_scenario(line) for line in f if line.strip()]
