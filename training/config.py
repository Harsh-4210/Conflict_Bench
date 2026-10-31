"""
ConflictBench - training configuration.

One dataclass, loaded from a YAML file (``configs/*.yaml``) and validated up
front so a bad value fails in the first second instead of an hour into a GPU
run. This module is pure Python (+ PyYAML): it must stay importable without
torch / unsloth / trl / datasets.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml


class ConfigError(ValueError):
    """Raised when a training config is missing, malformed or inconsistent."""


def _default_targets() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    # --- identity / output -------------------------------------------------
    run_name: str = "grpo"                    # results go to <results_dir>/<run_name>/
    output_dir: str = "conflictbench-grpo-output/grpo"   # checkpoints (git-ignored path)
    results_dir: str = "results"
    seed: int = 42

    # --- model -------------------------------------------------------------
    model: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=_default_targets)

    # --- data --------------------------------------------------------------
    train_size: int = 300                      # scenarios drawn from the TRAIN split
    val_size: int = 30                         # scenarios drawn from the VAL split (in-training eval)
    train_difficulty_mix: Dict[int, float] = field(default_factory=lambda: {1: 0.6, 2: 0.4})
    val_difficulty_mix: Dict[int, float] = field(default_factory=lambda: {1: 0.4, 2: 0.4, 3: 0.2})
    max_prompt_tokens: int = 1280              # TOKENS of the full chat-templated prompt
    max_completion_tokens: int = 512

    # --- optimisation ------------------------------------------------------
    learning_rate: float = 5e-6
    warmup_steps: int = 10
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    beta: float = 0.04                         # KL penalty
    num_generations: int = 4                   # completions per prompt (GRPO group size)
    per_device_train_batch_size: int = 4       # must be a multiple of num_generations (or batch*accum is)
    gradient_accumulation_steps: int = 2
    temperature: float = 0.85
    top_p: float = 0.92
    epochs: float = 1.0                        # ignored when max_steps > 0
    max_steps: int = -1                        # > 0 overrides epochs (smoke runs)

    # --- logging / checkpoints --------------------------------------------
    logging_steps: int = 5
    eval_steps: int = 25
    save_steps: int = 25                       # multiple of eval_steps (best-checkpoint requirement)
    save_total_limit: int = 3
    report_to: str = "none"                    # "none" | "wandb" | "tensorboard" ...
    hub_repo: Optional[str] = None             # push only when set AND HF_TOKEN is in the environment

    # ------------------------------------------------------------------ API
    @property
    def per_device_eval_batch_size(self) -> int:
        """GRPO needs the eval batch to be a multiple of num_generations; we use exactly one group."""
        return self.num_generations

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def prompts_per_step(self) -> int:
        return self.effective_batch_size // self.num_generations

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def validate(self) -> "TrainConfig":
        problems = validation_problems(self)
        if problems:
            raise ConfigError("Invalid training config:\n  - " + "\n  - ".join(problems))
        return self


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

_HUB_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _mix_problems(name: str, mix: Any) -> List[str]:
    if not isinstance(mix, dict) or not mix:
        return [f"{name} must be a non-empty mapping difficulty -> weight"]
    out: List[str] = []
    for d, w in mix.items():
        if not _is_int(d) or d < 1:
            out.append(f"{name}: difficulty keys must be integers >= 1, got {d!r}")
        if not _is_num(w) or w < 0:
            out.append(f"{name}: weight for difficulty {d!r} must be a number >= 0, got {w!r}")
    if not out and sum(mix.values()) <= 0:
        out.append(f"{name}: weights must not all be zero")
    return out


def validation_problems(c: TrainConfig) -> List[str]:
    p: List[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            p.append(msg)

    need(isinstance(c.run_name, str) and re.fullmatch(r"[\w.-]+", c.run_name or "") is not None,
         "run_name must be a non-empty string of letters, digits, '_', '-', '.'")
    need(isinstance(c.output_dir, str) and bool(c.output_dir), "output_dir must be a non-empty string")
    need(isinstance(c.results_dir, str) and bool(c.results_dir), "results_dir must be a non-empty string")
    need(_is_int(c.seed), "seed must be an integer")

    need(isinstance(c.model, str) and bool(c.model), "model must be a non-empty string")
    need(_is_int(c.max_seq_length) and c.max_seq_length >= 256, "max_seq_length must be an integer >= 256")
    need(isinstance(c.load_in_4bit, bool), "load_in_4bit must be true or false")
    need(_is_int(c.lora_r) and c.lora_r > 0, "lora_r must be a positive integer")
    need(_is_int(c.lora_alpha) and c.lora_alpha > 0, "lora_alpha must be a positive integer")
    need(_is_num(c.lora_dropout) and 0 <= c.lora_dropout < 1, "lora_dropout must be in [0, 1)")
    need(isinstance(c.lora_target_modules, list) and bool(c.lora_target_modules)
         and all(isinstance(m, str) and m for m in c.lora_target_modules),
         "lora_target_modules must be a non-empty list of module names")

    need(_is_int(c.train_size) and c.train_size >= 1, "train_size must be a positive integer")
    need(_is_int(c.val_size) and c.val_size >= 1, "val_size must be a positive integer")
    p += _mix_problems("train_difficulty_mix", c.train_difficulty_mix)
    p += _mix_problems("val_difficulty_mix", c.val_difficulty_mix)
    need(_is_int(c.max_prompt_tokens) and c.max_prompt_tokens > 0, "max_prompt_tokens must be a positive integer")
    need(_is_int(c.max_completion_tokens) and c.max_completion_tokens > 0,
         "max_completion_tokens must be a positive integer")
    if (_is_int(c.max_prompt_tokens) and _is_int(c.max_completion_tokens) and _is_int(c.max_seq_length)
            and c.max_prompt_tokens + c.max_completion_tokens > c.max_seq_length):
        p.append(f"max_prompt_tokens ({c.max_prompt_tokens}) + max_completion_tokens ({c.max_completion_tokens}) "
                 f"exceeds max_seq_length ({c.max_seq_length})")

    need(_is_num(c.learning_rate) and c.learning_rate > 0, "learning_rate must be > 0")
    need(_is_int(c.warmup_steps) and c.warmup_steps >= 0, "warmup_steps must be an integer >= 0")
    need(isinstance(c.lr_scheduler_type, str) and bool(c.lr_scheduler_type), "lr_scheduler_type must be a string")
    need(_is_num(c.weight_decay) and c.weight_decay >= 0, "weight_decay must be >= 0")
    need(_is_num(c.beta) and c.beta >= 0, "beta must be >= 0")
    need(_is_int(c.num_generations) and c.num_generations >= 2,
         "num_generations must be an integer >= 2 (GRPO compares completions within a group)")
    need(_is_int(c.per_device_train_batch_size) and c.per_device_train_batch_size >= 1,
         "per_device_train_batch_size must be a positive integer")
    need(_is_int(c.gradient_accumulation_steps) and c.gradient_accumulation_steps >= 1,
         "gradient_accumulation_steps must be a positive integer")
    if (_is_int(c.num_generations) and c.num_generations >= 2 and _is_int(c.per_device_train_batch_size)
            and _is_int(c.gradient_accumulation_steps) and c.per_device_train_batch_size >= 1
            and c.gradient_accumulation_steps >= 1
            and (c.per_device_train_batch_size * c.gradient_accumulation_steps) % c.num_generations != 0):
        p.append(f"per_device_train_batch_size ({c.per_device_train_batch_size}) x gradient_accumulation_steps "
                 f"({c.gradient_accumulation_steps}) = {c.per_device_train_batch_size * c.gradient_accumulation_steps} "
                 f"must be divisible by num_generations ({c.num_generations}) - GRPO requirement")
    need(_is_num(c.temperature) and c.temperature > 0, "temperature must be > 0")
    need(_is_num(c.top_p) and 0 < c.top_p <= 1, "top_p must be in (0, 1]")
    need(_is_num(c.epochs), "epochs must be a number")
    need(_is_int(c.max_steps) and (c.max_steps == -1 or c.max_steps > 0), "max_steps must be -1 (use epochs) or > 0")
    if _is_int(c.max_steps) and c.max_steps == -1:
        need(_is_num(c.epochs) and c.epochs > 0, "epochs must be > 0 when max_steps is -1")

    need(_is_int(c.logging_steps) and c.logging_steps >= 1, "logging_steps must be a positive integer")
    need(_is_int(c.eval_steps) and c.eval_steps >= 1, "eval_steps must be a positive integer")
    need(_is_int(c.save_steps) and c.save_steps >= 1, "save_steps must be a positive integer")
    if _is_int(c.save_steps) and _is_int(c.eval_steps) and c.eval_steps >= 1 and c.save_steps % c.eval_steps != 0:
        p.append(f"save_steps ({c.save_steps}) must be a multiple of eval_steps ({c.eval_steps}) "
                 "so that every checkpoint has a validation reward (needed to pick the best one)")
    need(_is_int(c.save_total_limit) and c.save_total_limit >= 1, "save_total_limit must be a positive integer")
    need(isinstance(c.report_to, str) and bool(c.report_to), 'report_to must be a string such as "none" or "wandb"')
    need(c.hub_repo is None or (isinstance(c.hub_repo, str) and _HUB_RE.match(c.hub_repo) is not None),
         'hub_repo must be null or "user-or-org/repo-name"')
    return p


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _coerce(name: str, value: Any, annotation: str) -> Any:
    """PyYAML reads ``3e-6`` as a *string* (YAML 1.1); turn numeric strings into numbers."""
    if isinstance(value, str):
        if annotation == "float":
            try:
                return float(value)
            except ValueError:
                raise ConfigError(f"{name}: expected a number, got {value!r}") from None
        if annotation == "int":
            try:
                return int(value)
            except ValueError:
                raise ConfigError(f"{name}: expected an integer, got {value!r}") from None
    if annotation == "float" and _is_int(value):
        return float(value)
    return value


def _normalise_mix(name: str, mix: Any) -> Any:
    if not isinstance(mix, Mapping):
        return mix  # validation reports it
    out: Dict[Any, Any] = {}
    for k, v in mix.items():
        try:
            key = int(k)
        except (TypeError, ValueError):
            key = k  # validation reports it
        out[key] = v
    return out


def config_from_dict(data: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> TrainConfig:
    merged: Dict[str, Any] = dict(data or {})
    if overrides:
        merged.update(overrides)
    by_name = {f.name: f for f in dataclasses.fields(TrainConfig)}
    unknown = sorted(set(merged) - set(by_name))
    if unknown:
        raise ConfigError(f"Unknown config key(s): {', '.join(unknown)}. Valid keys: {', '.join(sorted(by_name))}")
    kwargs: Dict[str, Any] = {}
    for key, value in merged.items():
        ann = str(by_name[key].type)
        if key.endswith("_difficulty_mix"):
            value = _normalise_mix(key, value)
        else:
            value = _coerce(key, value, ann)
        kwargs[key] = value
    return TrainConfig(**kwargs).validate()


def load_config(path: Union[str, Path], overrides: Optional[Mapping[str, Any]] = None) -> TrainConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping of keys to values")
    return config_from_dict(data, overrides)


def allocate_counts(size: int, mix: Mapping[int, float]) -> Dict[int, int]:
    """Split ``size`` across difficulties by weight (largest-remainder), summing to exactly ``size``."""
    total = float(sum(mix.values()))
    raw = {int(d): size * (w / total) for d, w in mix.items()}
    counts = {d: int(v) for d, v in raw.items()}
    short = size - sum(counts.values())
    for d in sorted(raw, key=lambda k: (raw[k] - counts[k], -k), reverse=True)[:short]:
        counts[d] += 1
    return {d: n for d, n in sorted(counts.items()) if n > 0}
