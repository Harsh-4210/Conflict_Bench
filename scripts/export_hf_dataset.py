"""
ConflictBench — Export a fixed scenario file as a Hugging Face dataset folder
Builds a folder that can be uploaded as-is, with the answers kept apart:

  <out>/public/test.jsonl         prompts only (no answers): upload to a PUBLIC dataset repo
  <out>/private/test_full.jsonl   full scenario_io records (answers): upload to a PRIVATE repo,
                                  read only by the scorer (leaderboard_space/app.py)
  <out>/README.md                 dataset card (YAML front matter + task, format, scoring, caveats)
  <out>/manifest.json             SHA-256 and size of every other file

The output is deterministic: the same input gives byte-identical files (no timestamps, sorted
keys, "\\n" line ends). Nothing is uploaded and no network is used. See docs/BENCHMARK.md.

Usage:
  python scripts/export_hf_dataset.py --data data/test_v1.jsonl --out dist/conflictbench-test-v1
"""

import argparse
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator import Scenario
from prompts import build_messages
from scenario_io import dumps_scenario, load_scenarios
from verifier import RUBRIC_WEIGHTS, SCHEMA_ERROR_FACTOR

# The ONLY keys a public record may have. Everything else in a Scenario (instructions' structured
# fields, conflicts, ground truth, facts, authority rules, delegations) is an answer or a hint.
PUBLIC_FIELDS = ("scenario_id", "difficulty", "domain", "split", "generator_version", "messages")

DATASET_NAME = "conflictbench-test-v1"
ANSWER_SCHEMA = (
    '{"identified_conflicts":[{"instruction_a":"<ID>","instruction_b":"<ID>",'
    '"conflict_type":"direct|resource|temporal|conditional","resolution":"<winning ID>",'
    '"reasoning":"<one sentence>"}],"execution_plan":["<ID>",...],"overridden_instructions":["<ID>",...]}'
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def public_record(scenario: Scenario) -> dict:
    """One public row: metadata plus exactly the chat messages a model is given. No answer fields."""
    return {
        "scenario_id": scenario.scenario_id,
        "difficulty": scenario.difficulty,
        "domain": scenario.domain,
        "split": scenario.split,
        "generator_version": scenario.generator_version,
        "messages": build_messages(scenario),
    }


def dumps_public(scenario: Scenario) -> str:
    return json.dumps(public_record(scenario), ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_source_hash(data_path: Path) -> Optional[str]:
    """
    If '<stem>.sha256' sits next to the input (sha256sum format), the input must match it.
    Returns the expected hash when a check was made, None when there is no hash file.
    """
    hash_file = data_path.with_suffix(".sha256")
    if not hash_file.is_file():
        return None
    tokens = hash_file.read_text(encoding="utf-8").split()
    expected = tokens[0].lower() if tokens else ""
    actual = sha256_file(data_path)
    if expected != actual:
        raise SystemExit(f"SHA-256 mismatch for {data_path}: {hash_file.name} says {expected or '(empty)'}, "
                         f"the file hashes to {actual}. Refusing to export.")
    return expected


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def _size_category(n: int) -> str:
    if n < 1000:
        return "n<1K"
    if n < 10000:
        return "1K<n<10K"
    return "10K<n<100K"


def build_card(scenarios: List[Scenario], source_name: str, source_sha256: str, hash_checked: bool) -> str:
    n = len(scenarios)
    by_diff = collections.Counter(s.difficulty for s in scenarios)
    by_domain = collections.Counter(s.domain for s in scenarios)
    splits = sorted({s.split for s in scenarios})
    versions = sorted({s.generator_version for s in scenarios})
    weights = RUBRIC_WEIGHTS
    formula = " + ".join(f"{w:.2f} x {name}" for name, w in weights.items())
    diff_rows = "\n".join(f"| {d} | {c} |" for d, c in sorted(by_diff.items()))
    domain_rows = "\n".join(f"| {d} | {c} |" for d, c in sorted(by_domain.items()))
    hash_note = ("It matched the hash file stored next to it when this folder was built."
                 if hash_checked else "No hash file was stored next to it, so this hash was computed but not checked.")

    return f"""---
license: mit
language:
- en
task_categories:
- text-generation
size_categories:
- {_size_category(n)}
pretty_name: ConflictBench test v1
tags:
- benchmark
- instruction-following
- reinforcement-learning
- synthetic
configs:
- config_name: default
  data_files:
  - split: test
    path: public/test.jsonl
---

# ConflictBench test v1

{n} synthetic business scenarios in which several people send conflicting instructions. A model has to decide which instructions to carry out and which to override, using only the authority rules, standing notices, dates and status facts given in the prompt. Scoring is rule-based (no LLM judge).

The answers are **not** in this repository. They are held in a separate private dataset and used only by the scorer (the ConflictBench leaderboard Space). Code, generator and verifier: https://github.com/Harsh-4210/Conflict_Bench

## Contents

| Field | Meaning |
|---|---|
| `scenario_id` | Unique ID of the scenario |
| `difficulty` | 1, 2 or 3 (more instructions and conflicts as it rises) |
| `domain` | {", ".join(sorted(by_domain))} |
| `split` | `{", ".join(splits) or "(none)"}` |
| `generator_version` | `{", ".join(v or "(legacy)" for v in versions)}` |
| `messages` | The chat messages to send to the model: a system message and a user message |

Difficulty counts:

| Difficulty | Scenarios |
|---|---:|
{diff_rows}

Domain counts:

| Domain | Scenarios |
|---|---:|
{domain_rows}

## Task and answer format

Send `messages` to the model unchanged, with greedy or low-temperature decoding. The model must reply with one JSON object:

```json
{ANSWER_SCHEMA}
```

- `identified_conflicts`: every pair of instructions that really conflict, and the winner of each pair.
- `execution_plan`: every instruction to carry out.
- `overridden_instructions`: every other instruction.

The scorer finds the last JSON object with all three keys in the reply, so reasoning text before it does not matter. A reply with no such object scores 0.

## Scoring

Each reply gets a composite score in [0, 1]:

```
composite = ({formula}) x format_gate
format_gate = 0 if no answer object is found, {SCHEMA_ERROR_FACTOR} if it has schema errors, else 1
```

- `correct_final_state`: mean of the F1 of the plan against the instructions that should be carried out and the F1 of the overridden list against those that should be overridden. IDs that are not in the scenario count as wrong in both lists, and an ID in both lists is penalised.
- `conflict_identification`: F1 of the listed conflict pairs against the real ones, times the share of correctly found pairs with the correct winner.
- `no_contradictions`: share of real conflicts where the plan holds exactly one of the two instructions and the stated winner is that one.

The implementation is `verifier.score` in the GitHub repository; if this card and the code disagree, the code is right. A leaderboard score is the mean over scenarios with a seeded 95% bootstrap confidence interval (1000 resamples). Per-difficulty and per-rubric means are reported too.

## How to submit

1. Run your model on every row of `public/test.jsonl` and write one line per scenario: `{{"scenario_id": "...", "completion": "<the raw model output>"}}`.
2. Open the ConflictBench leaderboard Space, upload the file and give your model a name. The Space scores it against the private answers.
3. A file must contain exactly one line for every scenario, with no duplicates and no unknown IDs; otherwise it is rejected with a message.

## Source and integrity

Built from `{source_name}` (SHA-256 `{source_sha256}`). {hash_note} `manifest.json` lists the SHA-256 of every other file in the folder this card was built with.

## Caveats

- The scenarios are generated from templates, not collected from real companies. Names, firms and rules are invented.
- The task can be solved by a hand-written rule solver (`resolver.py` in the repository scores 1.0 on the source file; see `results/` there). A good score here shows the model reads authority rules well, not that the task is hard in principle.
- The answers are only secret if the source file is. The source file is also stored in the GitHub repository with its answers. Until a fresh private set is generated and kept out of the repository, scores can be contaminated and should not be treated as proof of generalisation.
- Predictions are self-reported. The Space cannot check how a completion was produced.
- Only the one fixed prompt in `messages` is tested. Results may change with other prompts.
"""


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def export(data_path: Path, out_dir: Path) -> Dict[str, object]:
    """Write the dataset folder and return the manifest dict."""
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    expected = check_source_hash(data_path)
    source_sha = sha256_file(data_path)
    scenarios = load_scenarios(data_path)
    if not scenarios:
        raise SystemExit(f"{data_path} holds no scenarios")
    ids = [s.scenario_id for s in scenarios]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate scenario_id values in the input")

    # Only the files this script owns are cleared, never the whole --out folder.
    for sub in ("public", "private"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
    for name in ("README.md", "manifest.json"):
        (out_dir / name).unlink(missing_ok=True)

    _write(out_dir / "public" / "test.jsonl", "".join(dumps_public(s) + "\n" for s in scenarios))
    _write(out_dir / "private" / "test_full.jsonl", "".join(dumps_scenario(s) + "\n" for s in scenarios))
    _write(out_dir / "README.md",
           build_card(scenarios, data_path.name, source_sha, hash_checked=expected is not None))

    files = {}
    for rel in ("README.md", "private/test_full.jsonl", "public/test.jsonl"):
        p = out_dir / rel
        files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    manifest = {
        "dataset": DATASET_NAME,
        "n_scenarios": len(scenarios),
        "source": {"name": data_path.name, "sha256": source_sha, "hash_file_checked": expected is not None},
        "files": files,
    }
    _write(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a ConflictBench scenario file as an HF dataset folder")
    p.add_argument("--data", required=True, help="scenario JSONL file, e.g. data/test_v1.jsonl")
    p.add_argument("--out", required=True, help="output folder, e.g. dist/conflictbench-test-v1")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = build_parser().parse_args(argv)
    manifest = export(Path(args.data), Path(args.out))
    print(f"wrote {manifest['n_scenarios']} scenarios to {args.out}")
    for rel, info in manifest["files"].items():
        print(f"  {rel}  {info['bytes']} bytes  sha256 {info['sha256'][:12]}")
    print("Nothing was uploaded. See docs/BENCHMARK.md for the publish commands.")
    return manifest


if __name__ == "__main__":
    main()
