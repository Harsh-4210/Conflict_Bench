"""
ConflictBench — Production GRPO Training Script
================================================
Model: Qwen2.5-3B-Instruct
Trainer: Unsloth + TRL GRPOTrainer
Target: HuggingFace Spaces GPU (L40S / L4 / A10G)

Usage (in HF Space or Colab notebook):
    python train_grpo_prod.py

Estimated times:
    - L40S (48GB): ~8 hours  ($14)
    - L4   (24GB): ~14 hours ($11)
    - A10G (24GB): ~12 hours ($14)
"""

import os
import sys
import json
import random
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import torch
from datasets import Dataset

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, str(Path(__file__).parent))
from generator import ScenarioGenerator
from verifier import score as compute_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-detect GPU and set optimal config
# ---------------------------------------------------------------------------

def detect_gpu_config():
    """Auto-detect GPU and return optimal training parameters."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — need a GPU for training!")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16_ok = torch.cuda.is_bf16_supported()

    log.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f}GB | BF16: {bf16_ok}")

    # Default config (works on any 16GB+ GPU)
    config = {
        "model_name": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "load_in_4bit": True,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "num_generations": 6,
        "lora_r": 32,
    }

    if vram_gb >= 40:
        # L40S (48GB) or A100 — use 8-bit for better gradients
        log.info("→ L40S/A100 detected: using 8-bit quant + larger batch")
        config.update({
            "model_name": "unsloth/Qwen2.5-3B-Instruct",  # full precision base, Unsloth handles quant
            "load_in_4bit": False,
            "batch_size": 2,
            "gradient_accumulation": 4,
            "num_generations": 6,
            "lora_r": 32,
        })
    elif vram_gb >= 20:
        # L4 (24GB) or A10G (24GB) — 4-bit but can do bigger batch
        log.info("→ L4/A10G detected: 4-bit quant, moderate batch")
        config.update({
            "batch_size": 1,
            "gradient_accumulation": 8,
            "num_generations": 6,
            "lora_r": 32,
        })
    else:
        # T4 (16GB) fallback
        log.info("→ T4 detected: 4-bit quant, minimal batch")
        config.update({
            "batch_size": 1,
            "gradient_accumulation": 4,
            "num_generations": 6,
            "lora_r": 32,
        })

    return config


# ---------------------------------------------------------------------------
# Production hyperparameters
# ---------------------------------------------------------------------------

TRAIN_SCENARIOS = 400        # enough for GRPO convergence with fixed bugs
EVAL_SCENARIOS = 60          # larger eval set for reliable metrics
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 768         # avg completion ~286 tokens, 768 gives headroom without waste
MAX_PROMPT_LENGTH = 3200
LEARNING_RATE = 3e-6         # slightly lower than demo for stability with more data
WARMUP_RATIO = 0.05
BETA = 0.02                  # lower KL penalty — let model move further from base
SEED = 42
OUTPUT_DIR = "./conflictbench-grpo-output"

# Training difficulty — curriculum: 80% diff-1 (learn fundamentals), 20% diff-2 (generalize)
TRAIN_DIFFICULTY_MIX = {1: 0.8, 2: 0.2}

# Auto-upload to HF Hub after training
HF_REPO_ID = os.getenv("HF_REPO_ID", "Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Early stopping: save best checkpoint and stop if eval reward stagnates
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 5


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(n_train: int, n_eval: int) -> tuple[Dataset, Dataset]:
    gen = ScenarioGenerator(seed=SEED)
    log.info(f"Generating {n_train} training + {n_eval} eval scenarios (curriculum={TRAIN_DIFFICULTY_MIX})...")

    def make_records(n: int, difficulty) -> List[Dict[str, Any]]:
        records = []
        MAX_PROMPT_CHARS = 4000  # increased for difficulty 2
        attempts = 0
        while len(records) < n and attempts < n * 3:
            attempts += 1
            if isinstance(difficulty, int):
                diff = difficulty
            else:
                diff = random.choices(
                    list(difficulty.keys()),
                    weights=list(difficulty.values()),
                )[0]
            scenario = gen.generate(difficulty=diff)
            if len(scenario.prompt) > MAX_PROMPT_CHARS:
                continue
            records.append({
                "prompt": [{"role": "user", "content": scenario.prompt}],
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
            log.warning(f"Only generated {len(records)}/{n} records.")
        return records

    train_records = make_records(n_train, TRAIN_DIFFICULTY_MIX)
    eval_records = make_records(n_eval, {1: 0.3, 2: 0.5, 3: 0.2})  # mixed eval

    log.info(f"Dataset ready: {len(train_records)} train, {len(eval_records)} eval")
    return Dataset.from_list(train_records), Dataset.from_list(eval_records)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def build_reward_function():
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

                instructions = [
                    Instruction(
                        id=ins["id"],
                        text="",
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
                    difficulty=1,  # placeholder, not used by verifier
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
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("ConflictBench PRODUCTION GRPO Training")
    log.info("=" * 60)

    # Auto-detect GPU
    gpu_config = detect_gpu_config()

    # Load model
    log.info(f"Loading model: {gpu_config['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=gpu_config["model_name"],
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=gpu_config["load_in_4bit"],
        dtype=None,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=gpu_config["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=gpu_config["lora_r"],  # alpha = r is standard
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # Build datasets
    train_dataset, eval_dataset = build_dataset(TRAIN_SCENARIOS, EVAL_SCENARIOS)

    # Calculate expected steps for logging
    effective_batch = gpu_config["batch_size"] * gpu_config["gradient_accumulation"]
    total_samples = TRAIN_SCENARIOS * NUM_EPOCHS
    expected_steps = total_samples // effective_batch
    log.info(f"Expected training steps: {expected_steps}")
    log.info(f"Effective batch size: {effective_batch}")
    log.info(f"Num generations per prompt: {gpu_config['num_generations']}")

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=gpu_config["batch_size"],
        gradient_accumulation_steps=gpu_config["gradient_accumulation"],
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        num_generations=gpu_config["num_generations"],
        max_completion_length=MAX_NEW_TOKENS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        temperature=1.0,
        top_p=0.95,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        seed=SEED,
        dataloader_num_workers=2,
        beta=BETA,
        # Save total_limit to avoid disk overflow
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_reward",
        greater_is_better=True,
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

    # Train
    log.info("🚀 Starting GRPO training...")
    trainer.train()

    # Save final model
    log.info(f"Saving final model to {OUTPUT_DIR}/final")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    # Auto-upload to HF Hub
    if HF_REPO_ID and HF_TOKEN:
        log.info(f"Uploading to HF Hub: {HF_REPO_ID}")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)

            # Upload best checkpoint (not final, to avoid overfitting)
            best_ckpt = None
            for d in sorted(Path(OUTPUT_DIR).glob("checkpoint-*"), reverse=True):
                if (d / "adapter_model.safetensors").exists():
                    best_ckpt = d
                    break

            upload_path = str(best_ckpt) if best_ckpt else f"{OUTPUT_DIR}/final"
            log.info(f"Uploading from: {upload_path}")
            api.upload_folder(
                folder_path=upload_path,
                repo_id=HF_REPO_ID,
                repo_type="model",
                commit_message=f"ConflictBench GRPO - {TRAIN_SCENARIOS} scenarios, {NUM_EPOCHS} epochs",
                ignore_patterns=["optimizer.pt", "rng_state.pth", "scaler.pt",
                                 "scheduler.pt", "training_args.bin"],
            )
            log.info(f"✅ Model uploaded to https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            log.error(f"HF upload failed: {e}")
            log.info("Model saved locally — upload manually later.")

    log.info("🎉 Production training complete!")


if __name__ == "__main__":
    main()
