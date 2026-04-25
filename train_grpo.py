"""
ConflictBench — GRPO Training Script
Model: Qwen2.5-3B-Instruct (4-bit quantized)
Trainer: Unsloth + TRL GRPOTrainer
Hardware target: NVIDIA RTX 4060 8GB (Windows-compatible)

Run: python training/train_grpo.py
"""

import os
import sys
import json
import random
import logging
from typing import List, Dict, Any
from pathlib import Path

import torch
from datasets import Dataset

# Unsloth must be imported before transformers on Windows
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, str(Path(__file__).parent))
from generator import ScenarioGenerator
from verifier import score as compute_score
from conflict_bench import ConflictBenchEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config — tune these per your hardware
# ---------------------------------------------------------------------------

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"  # 4-bit, ~2.5GB VRAM
MAX_SEQ_LENGTH = 4096        # increased — prompts are long, need headroom
OUTPUT_DIR = "./conflictbench-grpo-output"
HF_REPO_ID = None            # set to "your-username/conflictbench-model" to push to HF

TRAIN_SCENARIOS = 600        # slightly reduced — faster iteration
EVAL_SCENARIOS = 80
NUM_EPOCHS = 2               # 2 epochs is enough with proper token budget
BATCH_SIZE = 1               # keep 1 for Colab T4/4060
GRADIENT_ACCUMULATION = 4    # effective batch = 4
NUM_GENERATIONS = 4          # GRPO samples 4 responses per prompt, ranks them
MAX_NEW_TOKENS = 512         # JSON response for 2-conflict scenario fits in 512 easily
LEARNING_RATE = 5e-6
WARMUP_RATIO = 0.05
SEED = 42
FORCE_DIFFICULTY = 1         # ← KEY FIX: train on difficulty 1 only (2 conflicts)
                             #   Fewer instructions = shorter prompts = responses fit in budget
                             #   Difficulty 1 prompt ≈ 900 tokens, leaving 512 for response


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(n_train: int, n_eval: int) -> tuple[Dataset, Dataset]:
    """
    Build train + eval datasets.
    Each example is a dict with 'prompt' and the raw scenario JSON (for reward fn).
    """
    gen = ScenarioGenerator(seed=SEED)
    log.info(f"Generating {n_train} training + {n_eval} eval scenarios...")

    def make_records(n: int, difficulty_mix: dict) -> List[Dict[str, Any]]:
        records = []
        MAX_PROMPT_CHARS = 3000  # safety guard — prompts above this eat the response budget
        attempts = 0
        while len(records) < n and attempts < n * 3:
            attempts += 1
            # Use FORCE_DIFFICULTY if set, else sample from mix
            if FORCE_DIFFICULTY is not None:
                diff = FORCE_DIFFICULTY
            else:
                diff = random.choices(
                    list(difficulty_mix.keys()),
                    weights=list(difficulty_mix.values()),
                )[0]
            scenario = gen.generate(difficulty=diff)
            # Skip prompts that are too long — they starve the response token budget
            if len(scenario.prompt) > MAX_PROMPT_CHARS:
                continue
            records.append({
                "prompt": scenario.prompt,
                "scenario_json": json.dumps({
                    "ground_truth_followed": scenario.ground_truth_followed,
                    "ground_truth_overridden": scenario.ground_truth_overridden,
                    "conflicts": [
                        {
                            "instruction_a_id": c.instruction_a_id,
                            "instruction_b_id": c.instruction_b_id,
                            "conflict_type": c.conflict_type,
                            "resolution_id": c.resolution_id,
                            "explanation": c.explanation,
                        }
                        for c in scenario.conflicts
                    ],
                    "instructions": [
                        {
                            "id": ins.id,
                            "action_key": ins.action_key,
                            "action_value": ins.action_value,
                            "source_priority": ins.source_priority,
                        }
                        for ins in scenario.instructions
                    ],
                }),
            })
        if len(records) < n:
            log.warning(f"Only generated {len(records)}/{n} records after {attempts} attempts (prompt length filter active).")
        return records

    # Training: difficulty 1 only when FORCE_DIFFICULTY is set, else mixed
    train_mix = {1: 1.0} if FORCE_DIFFICULTY else {1: 0.5, 2: 0.4, 3: 0.1}
    eval_mix  = {1: 0.5, 2: 0.3, 3: 0.2}  # eval always has mix for coverage

    train_records = make_records(n_train, train_mix)
    eval_records  = make_records(n_eval,  eval_mix)

    log.info("Dataset generation complete.")
    return Dataset.from_list(train_records), Dataset.from_list(eval_records)


# ---------------------------------------------------------------------------
# Reward function — called by GRPOTrainer
# ---------------------------------------------------------------------------

def build_reward_function():
    """
    Returns a reward function compatible with TRL GRPOTrainer.
    Signature: fn(prompts, completions, **kwargs) -> list[float]

    We reconstruct a lightweight Scenario object from the stored JSON
    to run the deterministic verifier.
    """
    # Import here to avoid circular issues at module level
    from generator import Scenario, Instruction, ConflictPair
    from verifier import score as _score

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        scenario_jsons = kwargs.get("scenario_json", [None] * len(completions))

        for completion, scenario_json in zip(completions, scenario_jsons):
            if scenario_json is None:
                rewards.append(0.0)
                continue

            try:
                data = json.loads(scenario_json)

                # Reconstruct minimal Scenario for verifier
                instructions = [
                    Instruction(
                        id=ins["id"],
                        text="",  # not needed for verification
                        source="",
                        source_priority=ins["source_priority"],
                        instruction_type="absolute",
                        action_key=ins["action_key"],
                        action_value=ins["action_value"],
                    )
                    for ins in data["instructions"]
                ]

                conflicts = [
                    ConflictPair(
                        instruction_a_id=c["instruction_a_id"],
                        instruction_b_id=c["instruction_b_id"],
                        conflict_type=c["conflict_type"],
                        resolution_id=c["resolution_id"],
                        explanation=c["explanation"],
                    )
                    for c in data["conflicts"]
                ]

                scenario = Scenario(
                    scenario_id="",
                    domain="",
                    difficulty=1,
                    business_context="",
                    instructions=instructions,
                    conflicts=conflicts,
                    ground_truth_followed=data["ground_truth_followed"],
                    ground_truth_overridden=data["ground_truth_overridden"],
                    prompt="",
                )

                breakdown = _score(completion, scenario)
                rewards.append(breakdown.composite)

            except Exception as e:
                log.warning(f"Reward computation error: {e}")
                rewards.append(0.0)

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main():
    log.info("=== ConflictBench GRPO Training ===")
    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log.info(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

    # ---- Load model ----
    log.info("Loading model with Unsloth 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    # Apply LoRA for GRPO fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # ---- Build datasets ----
    train_dataset, eval_dataset = build_dataset(TRAIN_SCENARIOS, EVAL_SCENARIOS)

    # ---- Configure GRPO trainer ----
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        num_generations=NUM_GENERATIONS,
        max_new_tokens=MAX_NEW_TOKENS,           # 512 — JSON response fits comfortably
        max_prompt_length=3200,                  # ~900 tokens for diff-1 prompt + padding
        temperature=0.9,
        top_p=0.95,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",  # set to "wandb" if you want W&B tracking
        seed=SEED,
        dataloader_num_workers=0,  # Windows compatibility
        # GRPO-specific
        use_vllm=False,          # keep False for 4060 (vllm needs more VRAM)
        beta=0.04,               # KL divergence penalty
    )

    reward_fn = build_reward_function()

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_fn],
    )

    # ---- Train ----
    log.info("Starting GRPO training...")
    trainer.train()

    # ---- Save ----
    log.info(f"Saving model to {OUTPUT_DIR}/final")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    # ---- Optional: push to HF Hub ----
    if HF_REPO_ID:
        log.info(f"Pushing to HuggingFace Hub: {HF_REPO_ID}")
        model.push_to_hub(HF_REPO_ID)
        tokenizer.push_to_hub(HF_REPO_ID)

    log.info("Training complete.")


# ---------------------------------------------------------------------------
# Evaluation utility — run after training to generate before/after plots
# ---------------------------------------------------------------------------

def evaluate_model(model_path: str, n_scenarios: int = 50):
    """
    Evaluate a trained model on fresh scenarios.
    Prints per-rubric average scores. Useful for before/after comparison.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Evaluating model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    gen = ScenarioGenerator(seed=999)
    totals = {
        "composite": 0.0,
        "correct_final_state": 0.0,
        "no_contradictions": 0.0,
        "conflict_identification": 0.0,
        "efficiency": 0.0,
        "format_compliance": 0.0,
    }

    for i in range(n_scenarios):
        diff = random.choice([1, 2, 3])
        scenario = gen.generate(difficulty=diff)

        inputs = tokenizer(scenario.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        breakdown = compute_score(response, scenario)
        for k, v in breakdown.to_dict().items():
            totals[k] += v

    print("\n=== Evaluation Results ===")
    for k, v in totals.items():
        print(f"  {k}: {v / n_scenarios:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to trained model for evaluation (skip training)")
    args = parser.parse_args()

    if args.eval:
        evaluate_model(args.eval)
    else:
        main()
