"""
Check, BEFORE a GPU run, that the prompts of a training config fit its token budget.

Reads the same YAML as the trainer, builds scenarios with the same generator, applies the model's real chat
template (system prompt + user prompt + generation prompt) and counts TOKENS - the exact filter the trainer applies.

    python scripts/diagnose_tokens.py --config configs/grpo_t4.yaml [--samples 40]

Needs `transformers` (tokenizer download only); no GPU.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts import build_messages  # noqa: E402
from training.config import TrainConfig, load_config  # noqa: E402
from training.train_grpo import make_generator, prompt_token_count  # noqa: E402


def measure(cfg: TrainConfig, tokenizer: Any, samples: int = 40) -> Dict[int, List[int]]:
    """Prompt token counts per difficulty (union of the train and val mixes), from the train-split generator."""
    difficulties = sorted(set(cfg.train_difficulty_mix) | set(cfg.val_difficulty_mix))
    gen = make_generator(cfg, "train")
    return {d: [prompt_token_count(tokenizer, build_messages(gen.generate(difficulty=d))) for _ in range(samples)]
            for d in difficulties}


def report(cfg: TrainConfig, lengths: Dict[int, List[int]]) -> List[str]:
    """Human-readable lines; entries starting with 'PROBLEM' mean the config will lose data."""
    lines = [f"config run '{cfg.run_name}': max_prompt_tokens={cfg.max_prompt_tokens}, "
             f"max_completion_tokens={cfg.max_completion_tokens}, max_seq_length={cfg.max_seq_length}"]
    for d, ls in lengths.items():
        fits = sum(1 for n in ls if n <= cfg.max_prompt_tokens)
        avg = sum(ls) / len(ls)
        lines.append(f"difficulty {d}: avg={avg:.0f} max={max(ls)} tokens | fit in max_prompt_tokens: {fits}/{len(ls)}")
        if fits < 0.5 * len(ls):
            lines.append(f"PROBLEM: more than half of difficulty {d} is longer than max_prompt_tokens; "
                         "the trainer will refuse to start. Raise max_prompt_tokens/max_seq_length or drop the level.")
        elif fits < len(ls):
            lines.append(f"note: {len(ls) - fits} of {len(ls)} difficulty-{d} samples are dropped by the length filter")
        if max(ls) + cfg.max_completion_tokens > cfg.max_seq_length:
            lines.append(f"note: longest difficulty-{d} prompt + max_completion_tokens exceeds max_seq_length "
                         "(those prompts are dropped by the filter)")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--samples", type=int, default=40)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    from transformers import AutoTokenizer
    print(f"loading tokenizer of {cfg.model}")
    tok = AutoTokenizer.from_pretrained(cfg.model)
    lines = report(cfg, measure(cfg, tok, args.samples))
    print("\n".join(lines))
    return 1 if any(line.startswith("PROBLEM") for line in lines) else 0


if __name__ == "__main__":
    raise SystemExit(main())
