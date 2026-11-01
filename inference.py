"""
Run the trained ConflictBench LoRA adapter on one generated scenario and print
the verifier score.

    python inference.py --seed 0 --difficulty 2

The adapter repo/path comes from the CONFLICTBENCH_ADAPTER environment variable
(default: the published Hugging Face adapter). The prompt is built by
prompts.build_messages, the same function training and the demo app use, and the
model answers with JSON only (greedy decoding).

Heavy imports (torch, transformers, peft) happen inside main(), so importing this
module is cheap.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generator import ScenarioGenerator
from prompts import build_messages
from verifier import score as compute_score

DEFAULT_ADAPTER = "Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora"
MAX_NEW_TOKENS = 512
MAX_PROMPT_TOKENS = 3072


def load_model(adapter_id: str):
    """Load the LoRA adapter (and the base model it points to) plus its tokenizer."""
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    cuda = torch.cuda.is_available()
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_id,
        torch_dtype=torch.float16 if cuda else torch.float32,
        device_map="auto" if cuda else None,
    )
    model.eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_id)
    except Exception:  # adapter repo without tokenizer files: use the base model's
        tokenizer = AutoTokenizer.from_pretrained(model.peft_config["default"].base_model_name_or_path)
    return model, tokenizer


def resolve_conflicts(model, tokenizer, scenario, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Return the model's raw completion (JSON text) for a scenario. Greedy decoding."""
    import torch

    encoded = tokenizer.apply_chat_template(
        build_messages(scenario),
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Newer transformers return a BatchEncoding, older ones a tensor.
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    n_prompt = int(input_ids.shape[-1])
    if n_prompt > MAX_PROMPT_TOKENS:
        raise ValueError(
            f"Prompt is {n_prompt} tokens (limit {MAX_PROMPT_TOKENS}); refusing to truncate it."
        )

    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0, help="scenario generator seed")
    parser.add_argument("--difficulty", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--adapter", default=os.getenv("CONFLICTBENCH_ADAPTER", DEFAULT_ADAPTER),
                        help="LoRA adapter repo id or local path (env: CONFLICTBENCH_ADAPTER)")
    args = parser.parse_args(argv)

    scenario = ScenarioGenerator(seed=args.seed).generate(difficulty=args.difficulty)
    print(f"Scenario {scenario.scenario_id} (domain={scenario.domain}, difficulty={scenario.difficulty}, "
          f"{len(scenario.conflicts)} conflicts)")

    print(f"Loading adapter {args.adapter} ...")
    model, tokenizer = load_model(args.adapter)

    print("Generating ...")
    response = resolve_conflicts(model, tokenizer, scenario)
    print("\n--- Model output ---")
    print(response)

    breakdown = compute_score(response, scenario).to_dict()
    print("\n--- Verifier score ---")
    print(json.dumps(breakdown, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
