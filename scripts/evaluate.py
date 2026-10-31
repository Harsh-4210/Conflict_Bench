"""
ConflictBench — One evaluation harness
Scores any backend (baseline strategy, local model, OpenAI-compatible API) on a
fixed scenario file with the shared prompt and the shared verifier, and writes
every completion plus a summary to results/<run_name>/.

Usage:
  python scripts/evaluate.py --strategy oracle_lookup --generate 600 --seed 1234
  python scripts/evaluate.py --model Qwen/Qwen2.5-3B-Instruct --data data/test_v1.jsonl
  python scripts/evaluate.py --model BASE --adapter ADAPTER --samples 4 --data FILE
  CONFLICTBENCH_API_KEY=... python scripts/evaluate.py --api https://api.groq.com/openai/v1 \\
      --api-model llama-3.3-70b-versatile --data FILE
"""

import argparse
import datetime
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator import Scenario, ScenarioGenerator
from scenario_io import load_scenarios, save_scenarios
from verifier import score

RUBRICS = ("correct_final_state", "no_contradictions", "conflict_identification",
           "efficiency", "format_compliance")
N_RESAMPLES = 1000
CI_SEED = 0
EXACT = 0.999  # correct_final_state at or above this counts as "solved" for pass@k


# ---------------------------------------------------------------------------
# Statistics and provenance helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values: Sequence[float], n_resamples: int = N_RESAMPLES,
                 seed: int = CI_SEED, level: float = 0.95) -> Tuple[float, float]:
    """Seeded percentile bootstrap CI of the mean. Same inputs -> same interval."""
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    alpha = (1.0 - level) / 2.0
    lo = means[int(alpha * n_resamples)]
    hi = means[min(n_resamples - 1, int((1.0 - alpha) * n_resamples))]
    return lo, hi


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _display_path(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo, so summaries do not leak local directories."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def git_info() -> Dict[str, object]:
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_dataset(n: int, seed: int, out_dir: Path) -> Path:
    """Build n scenarios spread evenly over difficulties 1-3 and save them under <out_dir>/data/.

    The filename carries the first 8 hex chars of the content hash, so a changed
    generator never silently overwrites an older set that results still point to.
    """
    gen = ScenarioGenerator(seed=seed)
    scenarios: List[Scenario] = []
    for d, count in zip((1, 2, 3), (n // 3 + (1 if i < n % 3 else 0) for i in range(3))):
        scenarios.extend(gen.generate(difficulty=d) for _ in range(count))
    data_dir = Path(out_dir) / "data"
    tmp = data_dir / f"_tmp_n{n}_seed{seed}.jsonl"
    save_scenarios(tmp, scenarios)
    final = data_dir / f"gen_n{n}_seed{seed}_{sha256_file(tmp)[:8]}.jsonl"
    if final.exists():
        tmp.unlink()
    else:
        tmp.replace(final)
    return final


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def summarise(rows: List[dict], skipped: int, n_samples: int) -> dict:
    """Aggregate per-completion rows into the summary statistics."""
    by_scenario: Dict[str, List[dict]] = {}
    for r in rows:
        by_scenario.setdefault(r["scenario_id"], []).append(r)
    per_scen = []  # (difficulty, mean of each rubric/composite over the scenario's samples)
    for sid, rs in by_scenario.items():
        entry = {k: _mean([r["score"][k] for r in rs]) for k in (*RUBRICS, "composite")}
        entry["difficulty"] = rs[0]["difficulty"]
        entry["solved_any"] = any(r["score"]["correct_final_state"] >= EXACT for r in rs)
        per_scen.append(entry)

    composites = [e["composite"] for e in per_scen]
    lo, hi = bootstrap_ci(composites)
    summary = {
        "n": len(per_scen),
        "n_completions": len(rows),
        "skipped_too_long": skipped,
        "composite": {"mean": _mean(composites), "ci95_low": lo, "ci95_high": hi,
                      "n_resamples": N_RESAMPLES, "ci_seed": CI_SEED},
        "per_difficulty": {
            str(d): {"mean": _mean([e["composite"] for e in per_scen if e["difficulty"] == d]),
                     "n": sum(1 for e in per_scen if e["difficulty"] == d)}
            for d in sorted({e["difficulty"] for e in per_scen})
        },
        "per_rubric": {k: _mean([e[k] for e in per_scen]) for k in RUBRICS},
        "parse_failure_rate": (sum(1 for r in rows if not r["score"]["format_ok"]) / len(rows))
        if rows else 0.0,
    }
    if n_samples > 1:
        summary["pass_at_k"] = {
            "k": n_samples,
            "definition": f"fraction of scenarios with at least one sample with correct_final_state >= {EXACT}",
            "value": _mean([1.0 if e["solved_any"] else 0.0 for e in per_scen]),
        }
        best = [max(r["score"]["composite"] for r in by_scenario[sid]) for sid in by_scenario]
        summary["best_of_k_composite"] = _mean(best)
    return summary


def run_evaluation(backend, scenarios: List[Scenario], run_name: str, out_dir: Path,
                   data_path: Path, progress: bool = False) -> dict:
    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    skipped = 0
    for i, sc in enumerate(scenarios):
        completions = backend.complete(sc)
        if completions is None:
            skipped += 1
            continue
        for k, text in enumerate(completions):
            rows.append({"scenario_id": sc.scenario_id, "difficulty": sc.difficulty,
                         "sample": k, "completion": text, "score": score(text, sc).to_dict()})
        if progress and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(scenarios)}", file=sys.stderr, flush=True)

    with (run_dir / "completions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = backend.describe()
    summary = {
        "run_name": run_name,
        **meta,
        "data_path": _display_path(Path(data_path)),
        "data_sha256": sha256_file(Path(data_path)),
        "n_scenarios_in_data": len(scenarios),
        **git_info(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **summarise(rows, skipped, meta.get("samples", 1)),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ConflictBench evaluation harness")
    src = p.add_argument_group("data (one of)")
    src.add_argument("--data", help="scenario JSONL file")
    src.add_argument("--generate", type=int, metavar="N", help="build N scenarios on the fly (even over difficulties)")
    src.add_argument("--seed", type=int, default=1234, help="seed for --generate (default 1234)")
    p.add_argument("--limit", type=int, help="score only the first N scenarios")
    p.add_argument("--run-name", help="results sub-directory (default: derived from backend and data)")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    be = p.add_argument_group("backend (one of)")
    be.add_argument("--strategy", help="baseline strategy name from scripts/baselines.STRATEGIES")
    be.add_argument("--model", help="HF model id or local path")
    be.add_argument("--adapter", help="PEFT adapter path or HF id (with --model)")
    be.add_argument("--api", metavar="BASE_URL", help="OpenAI-compatible base URL; key from CONFLICTBENCH_API_KEY")
    be.add_argument("--api-model", help="model name sent to the API")
    g = p.add_argument_group("generation")
    g.add_argument("--samples", type=int, default=1, help="samples per prompt; >1 uses temperature sampling")
    g.add_argument("--temperature", type=float, help="sampling temperature (model: default 0.7 when --samples>1; api: default 0)")
    g.add_argument("--max-new-tokens", type=int, default=1024)
    g.add_argument("--max-prompt-tokens", type=int, help="skip prompts longer than this (default: context window minus max-new-tokens)")
    g.add_argument("--strategy-seed", type=int, default=0, help="rng seed for randomised strategies")
    return p


def make_backend(args):
    chosen = [bool(args.strategy), bool(args.model), bool(args.api)]
    if sum(chosen) != 1:
        raise SystemExit("choose exactly one of --strategy, --model, --api")
    from scripts import backends

    if args.strategy:
        return backends.StrategyBackend(args.strategy, seed=args.strategy_seed)
    if args.model:
        return backends.ModelBackend(
            args.model, adapter=args.adapter, samples=args.samples,
            temperature=0.7 if args.temperature is None else args.temperature,
            max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens)
    if not args.api_model:
        raise SystemExit("--api needs --api-model")
    return backends.ApiBackend(
        args.api, args.api_model, samples=args.samples,
        temperature=0.0 if args.temperature is None else args.temperature,
        max_tokens=args.max_new_tokens)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in text).strip("-")


def default_run_name(args, data_path: Path) -> str:
    who = args.strategy or (Path(args.model).name + (f"+{Path(args.adapter).name}" if args.adapter else "")
                            if args.model else args.api_model)
    return _slug(f"{who}__{data_path.stem}")


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    if bool(args.data) == bool(args.generate):
        raise SystemExit("give exactly one of --data or --generate")
    data_path = Path(args.data) if args.data else generate_dataset(args.generate, args.seed, out_dir)
    scenarios = load_scenarios(data_path)
    if args.limit:
        scenarios = scenarios[: args.limit]
    backend = make_backend(args)
    run_name = args.run_name or default_run_name(args, data_path)
    summary = run_evaluation(backend, scenarios, run_name, out_dir, data_path, progress=True)
    c = summary["composite"]
    print(f"{run_name}: composite {c['mean']:.4f} (95% CI {c['ci95_low']:.4f}-{c['ci95_high']:.4f}) "
          f"n={summary['n']} parse_fail={summary['parse_failure_rate']:.3f} "
          f"skipped_too_long={summary['skipped_too_long']}")
    return summary


if __name__ == "__main__":
    main()
