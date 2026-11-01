"""
ConflictBench - the one GRPO training pipeline.

    python -m training.train_grpo --config configs/grpo_smoke.yaml
    python -m training.train_grpo --config configs/grpo_t4.yaml
    python -m training.train_grpo --config configs/grpo_a10g.yaml --max-steps 10

Everything (model, LoRA, data sizes, lengths, lr, beta, ...) comes from the YAML file. Sections:

  1. dataset   - scenarios from the TRAIN split (training) and VAL split (in-training eval), filtered by the
                 TOKEN length of the full chat-templated prompt. The test split is never touched here.
  2. reward    - ``verifier.score(...).composite`` exactly, plus one logging-only function per rubric.
  3. trainer   - Unsloth + TRL GRPOTrainer, resume, best checkpoint by val reward, run artefacts.

Heavy imports (unsloth, torch, datasets, trl) happen only inside ``train()``, so sections 1 and 2 are
importable and unit-testable on a machine without a GPU.

Reward design (see ``build_reward_functions``): TRL sums reward functions as ``sum_i weight_i * r_i``
(no division by the weights, then normalises inside each prompt's group). We pass
[composite, <one function per rubric>] with ``reward_weights = [1, 0, 0, 0]``. The total reward is therefore
exactly ``composite`` (which already includes the format gate, so summing the weighted rubrics would NOT equal it),
while TRL still logs ``rewards/<name>/mean`` for every rubric so you can see which part is learning.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import training  # noqa: F401  (puts the repo root on sys.path)
import verifier
from generator import GeneratorConfig, ScenarioGenerator
from prompts import build_messages
from scenario_io import dumps_scenario, loads_scenario
from training.config import ConfigError, TrainConfig, allocate_counts, load_config

log = logging.getLogger("conflictbench.train")

REPO_ROOT = Path(__file__).resolve().parent.parent

# ===========================================================================
# 1. Dataset
# ===========================================================================

# Splits this module is allowed to draw from. "test" is deliberately absent: it is evaluated once, at the end,
# by scripts/evaluate.py (P4-1), never during training.
TRAINING_SPLITS = ("train", "val")
_SPLIT_SEED_OFFSET = {"train": 0, "val": 1_000_003}
MAX_ATTEMPT_FACTOR = 3          # generate at most 3 x target candidates per difficulty
MIN_KEPT_FRACTION = 0.5         # fail if a difficulty keeps less than this share of its target


class DatasetError(RuntimeError):
    """The length filter removed too much of a requested difficulty."""


def make_generator(cfg: TrainConfig, split: str):
    """The ONE place that builds a scenario generator for training.

    Each split draws from its own disjoint pools of phrasings, companies and job titles
    (``GeneratorConfig(split=...)``), so the val reward measures held-out templates.
    """
    if split not in TRAINING_SPLITS:
        raise ValueError(f"training may only use splits {TRAINING_SPLITS}, got {split!r} "
                         "(the test split is evaluated once at the end by scripts/evaluate.py)")
    return ScenarioGenerator(seed=cfg.seed + _SPLIT_SEED_OFFSET[split], config=GeneratorConfig(split=split))


def prompt_token_count(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> int:
    """Token length of the full chat-templated prompt (system + user + generation prompt)."""
    text = tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


@dataclass
class DifficultyStats:
    target: int
    generated: int = 0
    dropped_too_long: int = 0
    dropped_duplicate: int = 0
    kept: int = 0
    max_tokens_seen: int = 0

    def to_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


@dataclass
class BuiltSplit:
    records: List[Dict[str, Any]]
    stats: Dict[int, DifficultyStats]


@dataclass
class BuiltDatasets:
    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]
    stats: Dict[str, Dict[int, DifficultyStats]]

    def stats_dict(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        return {s: {str(d): st.to_dict() for d, st in per.items()} for s, per in self.stats.items()}


def build_split(
    cfg: TrainConfig,
    split: str,
    size: int,
    mix: Mapping[int, float],
    tokenizer: Any,
    exclude_prompts: Optional[set] = None,
) -> BuiltSplit:
    """Generate ``size`` records for one split, honouring the difficulty mix and the token budget."""
    gen = make_generator(cfg, split)
    targets = allocate_counts(size, mix)
    exclude_prompts = exclude_prompts or set()
    records: List[Dict[str, Any]] = []
    stats: Dict[int, DifficultyStats] = {}
    problems: List[str] = []

    for difficulty, target in targets.items():
        st = stats[difficulty] = DifficultyStats(target=target)
        budget = MAX_ATTEMPT_FACTOR * target
        while st.kept < target and st.generated < budget:
            sc = gen.generate(difficulty=difficulty)
            st.generated += 1
            if sc.prompt in exclude_prompts:
                st.dropped_duplicate += 1
                continue
            messages = build_messages(sc)
            n_tokens = prompt_token_count(tokenizer, messages)
            st.max_tokens_seen = max(st.max_tokens_seen, n_tokens)
            if n_tokens > cfg.max_prompt_tokens:
                st.dropped_too_long += 1
                continue
            records.append({
                "prompt": messages,
                "scenario_json": dumps_scenario(sc),
                "difficulty": int(getattr(sc, "difficulty", difficulty)),
            })
            st.kept += 1
        log.info("[%s] difficulty %d: target=%d kept=%d generated=%d dropped_too_long=%d dropped_duplicate=%d "
                 "max_prompt_tokens_seen=%d (limit %d)", split, difficulty, st.target, st.kept, st.generated,
                 st.dropped_too_long, st.dropped_duplicate, st.max_tokens_seen, cfg.max_prompt_tokens)
        if st.kept < st.target:
            log.warning("[%s] difficulty %d kept only %d of %d requested", split, difficulty, st.kept, st.target)
        if st.kept < MIN_KEPT_FRACTION * st.target:
            problems.append(
                f"{split} difficulty {difficulty}: kept {st.kept}/{st.target} "
                f"({st.dropped_too_long} longer than max_prompt_tokens={cfg.max_prompt_tokens}, "
                f"largest seen {st.max_tokens_seen} tokens)")
    if problems:
        raise DatasetError(
            "The prompt length filter removed more than half of a requested difficulty. Raise max_prompt_tokens "
            "(and max_seq_length), or drop that difficulty from the mix:\n  - " + "\n  - ".join(problems))

    random.Random(cfg.seed + _SPLIT_SEED_OFFSET[split]).shuffle(records)
    return BuiltSplit(records=records, stats=stats)


def build_datasets(cfg: TrainConfig, tokenizer: Any) -> BuiltDatasets:
    """Train records from the train split, val records from the val split (never the test split)."""
    train = build_split(cfg, "train", cfg.train_size, cfg.train_difficulty_mix, tokenizer)
    train_prompts = {r["prompt"][-1]["content"] for r in train.records}
    val = build_split(cfg, "val", cfg.val_size, cfg.val_difficulty_mix, tokenizer, exclude_prompts=train_prompts)
    return BuiltDatasets(train=train.records, val=val.records, stats={"train": train.stats, "val": val.stats})


# ===========================================================================
# 2. Reward
# ===========================================================================

MAX_CONSECUTIVE_REWARD_FAILURES = 10
_CACHE_LIMIT = 4096


class RewardFailure(RuntimeError):
    """The reward code failed too many times in a row; training is stopped instead of learning from zeros."""


def completion_text(completion: Any) -> str:
    """TRL passes a string for plain prompts and a list of ``{"role", "content"}`` messages for chat prompts."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return completion_text(completion.get("content"))
    if isinstance(completion, (list, tuple)):
        parts = []
        for item in completion:
            if isinstance(item, Mapping) and "text" in item and "content" not in item:
                parts.append(str(item["text"]))      # multimodal-style content part
            else:
                parts.append(completion_text(item))
        return "".join(parts)
    return str(completion)


class RewardScorer:
    """Scores completions with ``verifier.score`` and keeps failure statistics.

    One instance is shared by the composite reward function and the per-rubric functions, so every
    completion is scored once and a verifier crash is counted once.
    """

    def __init__(self, max_consecutive_failures: int = MAX_CONSECUTIVE_REWARD_FAILURES):
        self.max_consecutive_failures = max_consecutive_failures
        self.failures = 0
        self.consecutive_failures = 0
        self.scored = 0
        self._cache: Dict[Tuple[str, str], Optional[Any]] = {}

    def score(self, completion: Any, scenario_json: Optional[str]):
        """Return a ``ScoreBreakdown``, or ``None`` if scoring failed (already logged and counted)."""
        text = completion_text(completion)
        key = (scenario_json or "", text)
        if key in self._cache:
            return self._cache[key]
        try:
            if not scenario_json:
                raise ValueError("record has no scenario_json")
            breakdown = verifier.score(text, loads_scenario(scenario_json))
        except Exception as exc:  # noqa: BLE001 - we count, log, and re-raise after N in a row
            self.failures += 1
            self.consecutive_failures += 1
            log.warning("reward failure #%d (%d in a row): %s: %s | completion[:200]=%r",
                        self.failures, self.consecutive_failures, type(exc).__name__, exc, text[:200])
            if self.consecutive_failures >= self.max_consecutive_failures:
                raise RewardFailure(
                    f"{self.consecutive_failures} reward computations failed in a row "
                    f"({self.failures} in total); last error: {type(exc).__name__}: {exc}") from exc
            breakdown = None
        else:
            self.consecutive_failures = 0
            self.scored += 1
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = breakdown
        return breakdown


def _report_metric(kwargs: Mapping[str, Any], scorer: RewardScorer) -> None:
    """Best effort: surface the failure counter through TRL's ``log_metric`` hook when this TRL has one."""
    log_metric = kwargs.get("log_metric")
    if callable(log_metric):
        try:
            log_metric("reward_errors_total", float(scorer.failures))
        except Exception:  # noqa: BLE001 - logging must never break training
            log.debug("log_metric hook rejected reward_errors_total", exc_info=True)


def build_reward_functions(scorer: Optional[RewardScorer] = None):
    """Return ``(reward_funcs, reward_weights, scorer)`` for ``GRPOTrainer``.

    reward_funcs    = [reward_composite, reward_<rubric> for each key of verifier.RUBRIC_WEIGHTS]
    reward_weights  = [1.0, 0.0, 0.0, 0.0]

    Only ``reward_composite`` drives learning and it equals ``verifier.score(...).composite`` exactly (no
    re-weighting, no difficulty bonus). A failed computation yields 0.0 for that completion (and is counted and
    logged); ``MAX_CONSECUTIVE_REWARD_FAILURES`` failures in a row raise ``RewardFailure``.
    """
    scorer = scorer or RewardScorer()

    def reward_composite(prompts=None, completions=None, scenario_json=None, **kwargs) -> List[float]:
        out: List[float] = []
        for completion, sj in zip(completions or [], scenario_json or []):
            bd = scorer.score(completion, sj)
            out.append(0.0 if bd is None else float(bd.composite))
        _report_metric(kwargs, scorer)
        return out

    def _rubric(name: str) -> Callable[..., List[float]]:
        def fn(prompts=None, completions=None, scenario_json=None, **kwargs) -> List[float]:
            out: List[float] = []
            for completion, sj in zip(completions or [], scenario_json or []):
                bd = scorer.score(completion, sj)
                out.append(0.0 if bd is None else float(getattr(bd, name)))
            return out
        fn.__name__ = f"reward_{name}"
        fn.__qualname__ = fn.__name__
        return fn

    funcs: List[Callable[..., List[float]]] = [reward_composite] + [_rubric(n) for n in verifier.RUBRIC_WEIGHTS]
    weights = [1.0] + [0.0] * len(verifier.RUBRIC_WEIGHTS)
    return funcs, weights, scorer


# ===========================================================================
# 3. Trainer
# ===========================================================================

# TRL's eval metric for the mean total reward is logged as "eval_reward" (verified in TRL v1.13.0 source:
# metrics are stored under "reward" and prefixed "eval_" when evaluating). Used to pick the best checkpoint.
BEST_METRIC = "eval_reward"
# GRPOConfig fields we refuse to silently drop if the installed TRL does not know them.
_CRITICAL_GRPO_KEYS = {
    "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "num_generations",
    "max_completion_length", "reward_weights", "eval_strategy", "save_strategy", "metric_for_best_model",
    "load_best_model_at_end", "learning_rate", "beta", "temperature", "output_dir",
}


def build_grpo_kwargs(cfg: TrainConfig, bf16: bool, reward_weights: Sequence[float]) -> Dict[str, Any]:
    """Keyword arguments for ``GRPOConfig`` (pure function so it can be tested without TRL)."""
    return {
        "output_dir": cfg.output_dir,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        # batching: GRPO needs (batch x accumulation) divisible by num_generations, eval batch = one group
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_generations": cfg.num_generations,
        # duration
        "num_train_epochs": cfg.epochs,
        "max_steps": cfg.max_steps,
        # optimiser
        "learning_rate": cfg.learning_rate,
        "warmup_steps": cfg.warmup_steps,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "weight_decay": cfg.weight_decay,
        # sampling / objective
        "max_completion_length": cfg.max_completion_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "beta": cfg.beta,
        "reward_weights": list(reward_weights),
        # older TRL releases truncate prompts to this many TOKENS; newer ones removed the option (we filter instead)
        "max_prompt_length": cfg.max_prompt_tokens,
        # logging / checkpoints / best-by-val-reward
        "logging_steps": cfg.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": cfg.eval_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": BEST_METRIC,
        "greater_is_better": True,
        # precision, reporting, misc
        "bf16": bf16,
        "fp16": not bf16,
        "report_to": cfg.report_to,
        "remove_unused_columns": False,      # reward functions need the scenario_json column
        "dataloader_num_workers": 0,
        "use_vllm": False,
    }


def filter_supported_kwargs(config_cls: type, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop keyword arguments the installed TRL/Unsloth ``GRPOConfig`` does not have (warn), except critical ones."""
    names = {f.name for f in dataclasses.fields(config_cls)}
    missing = sorted(k for k in kwargs if k not in names)
    critical = [k for k in missing if k in _CRITICAL_GRPO_KEYS]
    if critical:
        raise RuntimeError(f"The installed TRL's GRPOConfig has no {critical}; the pinned TRL version in "
                           "requirements-train.txt is not what is installed.")
    if missing:
        log.warning("GRPOConfig of the installed TRL does not accept %s; skipping them", missing)
    return {k: v for k, v in kwargs.items() if k in names}


def find_resume_checkpoint(output_dir: str) -> Optional[str]:
    """Latest ``checkpoint-N`` directory (with a trainer_state.json) under ``output_dir``, else None."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    best: Optional[Tuple[int, Path]] = None
    for d in root.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if m and d.is_dir() and (d / "trainer_state.json").is_file():
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, d)
    return str(best[1]) if best else None


def git_commit_hash(cwd: Path = REPO_ROOT) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("CONFLICTBENCH_REF", "unknown")


def pip_freeze_text() -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    except (OSError, subprocess.SubprocessError):
        pass
    from importlib import metadata  # uv venvs have no pip
    lines = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]},
                   key=str.lower)
    return "\n".join(lines) + "\n"


def write_run_artifacts(cfg: TrainConfig, run_dir: Path) -> None:
    """Everything needed to reproduce the run, written before training starts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(git_commit_hash() + "\n", encoding="utf-8")
    (run_dir / "pip_freeze.txt").write_text(pip_freeze_text(), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({
        "python": sys.version, "platform": platform.platform(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2), encoding="utf-8")


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def push_to_hub(folder: Path, repo: str, token: str, commit_message: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model", commit_message=commit_message)


def train(cfg: TrainConfig, *, resume: bool = True, progress_callback: Optional[Callable[[str], None]] = None
          ) -> Dict[str, Any]:
    """Run one GRPO training job. Needs a GPU plus the packages in requirements-train.txt."""
    def emit(msg: str) -> None:
        log.info(msg)
        if progress_callback:
            progress_callback(msg)

    run_dir = Path(cfg.results_dir) / cfg.run_name
    write_run_artifacts(cfg, run_dir)
    emit(f"run '{cfg.run_name}': artefacts in {run_dir}, checkpoints in {cfg.output_dir}")

    # --- heavy imports. unsloth FIRST: it patches transformers/trl/peft on import. -----------------------------
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit("unsloth is not installed. Install the pinned training stack: "
                         "pip install -r requirements-train.txt") from exc
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    random.seed(cfg.seed)
    if not torch.cuda.is_available():
        emit("WARNING: no CUDA GPU detected; training will not work or will be extremely slow")

    emit(f"loading {cfg.model} (4-bit={cfg.load_in_4bit}, max_seq_length={cfg.max_seq_length})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model, max_seq_length=cfg.max_seq_length, load_in_4bit=cfg.load_in_4bit, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r, target_modules=list(cfg.lora_target_modules), lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, bias="none", use_gradient_checkpointing="unsloth", random_state=cfg.seed)
    tokenizer.padding_side = "left"          # GRPOTrainer requires left padding for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}           # some TRL/Unsloth combinations read this attribute

    emit("building datasets (train split for training, val split for evaluation)")
    built = build_datasets(cfg, tokenizer)
    (run_dir / "dataset_stats.json").write_text(json.dumps(built.stats_dict(), indent=2), encoding="utf-8")
    train_ds, val_ds = Dataset.from_list(built.train), Dataset.from_list(built.val)
    emit(f"datasets ready: {len(train_ds)} train / {len(val_ds)} val")

    funcs, weights, scorer = build_reward_functions()
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    emit(f"precision: {'bf16' if bf16 else 'fp16'}; prompts per optimiser step: {cfg.prompts_per_step}")
    args = GRPOConfig(**filter_supported_kwargs(GRPOConfig, build_grpo_kwargs(cfg, bf16, weights)))

    class _ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if logs:
                parts = [f"{k}={v:.4g}" for k, v in logs.items() if isinstance(v, (int, float))]
                emit(f"step {state.global_step}: " + " | ".join(parts))

    trainer = GRPOTrainer(
        model=model, args=args, reward_funcs=funcs, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, callbacks=[_ProgressCallback()])

    checkpoint = find_resume_checkpoint(cfg.output_dir) if resume else None
    if checkpoint:
        emit(f"resuming from {checkpoint}")
    emit("starting GRPO training")
    trainer.train(resume_from_checkpoint=True if checkpoint else None)

    # load_best_model_at_end has reloaded the checkpoint with the best val reward by now.
    emit("final validation pass on the best checkpoint")
    final_val = _json_safe(trainer.evaluate())
    final_dir = Path(cfg.output_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log_history = _json_safe(trainer.state.log_history)
    summary = {
        "run_name": cfg.run_name,
        "git_commit": git_commit_hash(),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric_name": BEST_METRIC,
        "best_metric": _json_safe(trainer.state.best_metric),
        "global_step": trainer.state.global_step,
        "final_val_metrics": final_val,
        "reward_errors_total": scorer.failures,
        "rewards_scored": scorer.scored,
        "adapter_dir": str(final_dir),
        "dataset": built.stats_dict(),
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "log_history.json").write_text(json.dumps(log_history, indent=2), encoding="utf-8")
    emit(f"saved adapter to {final_dir}; summary in {run_dir / 'final_metrics.json'}")

    token = os.environ.get("HF_TOKEN")
    if cfg.hub_repo and token:
        emit(f"pushing adapter to {cfg.hub_repo}")
        try:
            push_to_hub(final_dir, cfg.hub_repo, token, f"ConflictBench GRPO {cfg.run_name}")
        except Exception as exc:  # noqa: BLE001
            emit(f"hub push failed ({type(exc).__name__}: {exc}); the adapter is still saved locally")
    elif cfg.hub_repo:
        emit("hub_repo is set but HF_TOKEN is not in the environment; not pushing")

    emit("next: run scripts/evaluate.py on the TEST split, once, with --adapter " + str(final_dir))
    return {"run_dir": str(run_dir), "final_dir": str(final_dir), "log_history": log_history, "summary": summary}


# ===========================================================================
# CLI
# ===========================================================================

def _parse_override(text: str) -> Tuple[str, Any]:
    import yaml
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m training.train_grpo", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="YAML config, e.g. configs/grpo_smoke.yaml")
    p.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=None,
                   help="override max_steps (e.g. 10 for a dry run)")
    p.add_argument("--run-name", dest="run_name", default=None, help="override run_name (results/<run_name>/)")
    p.add_argument("--output-dir", dest="output_dir", default=None, help="override output_dir (checkpoints)")
    p.add_argument("--set", dest="overrides", action="append", default=[], type=_parse_override,
                   metavar="KEY=VALUE", help="override any config key (repeatable)")
    p.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints in output_dir")
    p.add_argument("--dry-run", action="store_true",
                   help="validate the config, build the datasets with the real tokenizer, print stats, then exit "
                        "(needs `transformers` only, no GPU)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    overrides: Dict[str, Any] = dict(args.overrides)
    for key in ("max_steps", "run_name", "output_dir"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    log.info("config %s ok: model=%s, %d train / %d val scenarios, %d prompts per step",
             args.config, cfg.model, cfg.train_size, cfg.val_size, cfg.prompts_per_step)

    if args.dry_run:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        try:
            built = build_datasets(cfg, tokenizer)
        except DatasetError as exc:
            log.error("%s", exc)
            return 3
        print(json.dumps(built.stats_dict(), indent=2))
        return 0

    train(cfg, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
