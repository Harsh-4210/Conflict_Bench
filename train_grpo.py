"""
ConflictBench — GRPO Training Script  RUN 2
═══════════════════════════════════════════════════════════════════
Changes from Run 1:
  ✅ 600 training scenarios  (was 120)
  ✅ Difficulty mix 60% diff-1 + 40% diff-2  (was 100% diff-1)
  ✅ 3 epochs  (was 2)
  ✅ LR lowered to 3e-6  (was 5e-6 — more stable second run)
  ✅ Gradient accumulation 8  (was 4 — bigger effective batch = 8)
  ✅ NUM_GENERATIONS 8  (was 4 — more GRPO variance per prompt)
  ✅ MAX_PROMPT_CHARS 2800  (tighter guard vs 3000)
  ✅ Save checkpoint every 50 steps  (was 100)
  ✅ Eval every 50 steps  (was 50 — keep same)
  ✅ bf16 auto-detect for A10G  (A10G supports bf16 natively)
  ✅ System prompt injection to guide JSON format
  ✅ Weights fixed in RUBRIC_WEIGHTS to match UI display

Hardware target: HuggingFace Jobs — A10G (24GB)
Estimated time: 6–9 hours
Expected reward improvement: 0.52 → 0.62–0.68

Run on HF Jobs:
  hf jobs uv run --flavor a10g-small -- python train_grpo_v2.py

Or Colab A100:
  !python train_grpo_v2.py
"""

import os
import sys
import json
import random
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset

# Unsloth MUST come before transformers
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, str(Path(__file__).parent))
from generator import ScenarioGenerator
from verifier import score as compute_score, parse_agent_output
from conflict_bench import ConflictBenchEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

RUN_TAG           = f"run2_{datetime.now().strftime('%Y%m%d_%H%M')}"
MODEL_NAME        = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH    = 4096
OUTPUT_DIR        = f"./conflictbench-grpo-{RUN_TAG}"
HF_REPO_ID        = "Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora"  # set None to skip push

# Dataset
TRAIN_SCENARIOS   = 600      # ↑ was 120
EVAL_SCENARIOS    = 30       # ↑ was 20
SEED              = 42

# Difficulty mix: 60% easy (diff 1) + 40% medium (diff 2)
# Diff 1 = 2 conflicts (model is already decent here)
# Diff 2 = 4 conflicts (new challenge — teaches richer conflict resolution)
DIFFICULTY_TRAIN_WEIGHTS = {1: 0.60, 2: 0.40}
DIFFICULTY_EVAL_WEIGHTS  = {1: 0.40, 2: 0.40, 3: 0.20}

# Training
NUM_EPOCHS        = 3        # ↑ was 2
BATCH_SIZE        = 1
GRADIENT_ACCUM    = 8        # ↑ was 4 → effective batch = 8
NUM_GENERATIONS   = 8        # ↑ was 4 → more GRPO variance per prompt
MAX_NEW_TOKENS    = 768      # slightly more room for diff-2 multi-conflict JSON
MAX_PROMPT_CHARS  = 2800     # tighter than run1's 3000
LEARNING_RATE     = 3e-6     # ↓ was 5e-6 — more stable
WARMUP_STEPS      = 20       # flat warmup steps (not ratio — more predictable)
BETA              = 0.04     # KL penalty — keep same
TEMPERATURE       = 0.85     # slightly lower for more focused sampling
TOP_P             = 0.92

# ── System prompt ─────────────────────────────────────────────────────────────
# Injected as a system message to give the model stronger format guidance.
# Run 1 had no system message — this alone should improve format_compliance.

SYSTEM_PROMPT = """You are an expert business operations coordinator.
Your task: given a set of business instructions from various stakeholders, identify ALL conflicts and produce a resolution plan.

Authority hierarchy (lower number = higher authority — ALWAYS wins):
1. Legal & Compliance, Regulatory Affairs
2. CEO Office, CFO, CTO, COO
3. VP Engineering, VP Finance, VP Operations, VP Human Resources
4. Director of IT, Director of Finance
5. Engineering Manager, Finance Manager, HR Manager, IT Manager
6. Team Lead, Department Coordinator

Rules:
- When two instructions conflict, the HIGHER authority (lower tier number) always wins
- List ALL conflicting pairs — do not miss any
- Your execution_plan must contain ONLY the winning instructions + non-conflicting ones
- overridden_instructions must contain ONLY the losing instructions

Output ONLY valid JSON matching this exact schema. No preamble, no explanation outside the JSON:
{"identified_conflicts":[{"instruction_a":"<ID>","instruction_b":"<ID>","conflict_type":"direct|resource|temporal|absolute","resolution":"<winning ID>","reasoning":"<one sentence>"}],"execution_plan":["<ID>",...],"overridden_instructions":["<ID>",...]}"""


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset() -> Tuple[Dataset, Dataset]:
    gen = ScenarioGenerator(seed=SEED)
    log.info(f"Generating {TRAIN_SCENARIOS} train + {EVAL_SCENARIOS} eval scenarios...")

    def make_records(n: int, diff_weights: Dict[int, float]) -> List[Dict[str, Any]]:
        records = []
        attempts = 0
        skipped  = 0

        while len(records) < n:
            attempts += 1
            if attempts > n * 5:
                log.warning(f"Stopping early: {len(records)}/{n} records after {attempts} attempts ({skipped} skipped for length)")
                break

            diff = random.choices(
                list(diff_weights.keys()),
                weights=list(diff_weights.values()),
            )[0]

            scenario = gen.generate(difficulty=diff)

            # Hard prompt length gate — prompts above this eat the response budget
            if len(scenario.prompt) > MAX_PROMPT_CHARS:
                skipped += 1
                continue

            # Build messages with system prompt for better format guidance
            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": scenario.prompt},
            ]

            records.append({
                "prompt": messages,
                "scenario_json": json.dumps({
                    "ground_truth_followed":  scenario.ground_truth_followed,
                    "ground_truth_overridden": scenario.ground_truth_overridden,
                    "difficulty":             scenario.difficulty,
                    "conflicts": [
                        {
                            "instruction_a_id": c.instruction_a_id,
                            "instruction_b_id": c.instruction_b_id,
                            "conflict_type":    c.conflict_type,
                            "resolution_id":    c.resolution_id,
                            "explanation":      c.explanation,
                        }
                        for c in scenario.conflicts
                    ],
                    "instructions": [
                        {
                            "id":               ins.id,
                            "action_key":       ins.action_key,
                            "action_value":     ins.action_value,
                            "source_priority":  ins.source_priority,
                        }
                        for ins in scenario.instructions
                    ],
                }),
            })

        diff_counts = {}
        for r in records:
            d = json.loads(r["scenario_json"])["difficulty"]
            diff_counts[d] = diff_counts.get(d, 0) + 1
        log.info(f"  Generated {len(records)} records. Difficulty distribution: {diff_counts}")
        return records

    train_records = make_records(TRAIN_SCENARIOS, DIFFICULTY_TRAIN_WEIGHTS)
    eval_records  = make_records(EVAL_SCENARIOS,  DIFFICULTY_EVAL_WEIGHTS)

    return Dataset.from_list(train_records), Dataset.from_list(eval_records)


# ── Reward function ───────────────────────────────────────────────────────────

def build_reward_fn():
    """
    Returns GRPO-compatible reward function.
    Signature: fn(prompts, completions, **kwargs) -> list[float]

    Run 2 improvement: per-difficulty reward shaping.
    Difficulty 2 answers get a small bonus multiplier to encourage
    the model to try harder on harder problems.
    """
    from generator import Scenario, Instruction, ConflictPair
    from verifier import score as _score

    # Weights matching the UI display (fixed from run 1 mismatch)
    RUBRIC_WEIGHTS_V2 = {
        "correct_final_state":     0.35,
        "no_contradictions":       0.25,  # ↑ was 0.15
        "conflict_identification": 0.20,  # ↓ was 0.30
        "efficiency":              0.10,  # ↑ was 0.08
        "format_compliance":       0.10,  # ↓ was 0.12
    }

    def reward_fn(
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
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
                    difficulty=data.get("difficulty", 1),
                    business_context="",
                    instructions=instructions,
                    conflicts=conflicts,
                    ground_truth_followed=data["ground_truth_followed"],
                    ground_truth_overridden=data["ground_truth_overridden"],
                    prompt="",
                )

                breakdown = _score(completion, scenario)

                # Re-compute composite with v2 weights
                composite_v2 = (
                    RUBRIC_WEIGHTS_V2["correct_final_state"]     * breakdown.correct_final_state +
                    RUBRIC_WEIGHTS_V2["no_contradictions"]        * breakdown.no_contradictions +
                    RUBRIC_WEIGHTS_V2["conflict_identification"]  * breakdown.conflict_identification +
                    RUBRIC_WEIGHTS_V2["efficiency"]               * breakdown.efficiency +
                    RUBRIC_WEIGHTS_V2["format_compliance"]        * breakdown.format_compliance
                )

                # Small difficulty bonus: diff-2 scenarios score up to +5% extra
                # This encourages the model to engage with harder problems
                diff_bonus = 0.05 if data.get("difficulty", 1) == 2 else 0.0
                final_reward = min(1.0, composite_v2 + diff_bonus * composite_v2)

                rewards.append(round(final_reward, 4))

            except Exception as e:
                log.debug(f"Reward error: {e}")
                rewards.append(0.0)

        return rewards

    return reward_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info(f"  ConflictBench GRPO Training — {RUN_TAG}")
    log.info("=" * 60)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  GPU: {gpu}  |  VRAM: {vram:.1f} GB")
    else:
        log.warning("  No CUDA GPU detected — training will be very slow")

    log.info(f"  Scenarios: {TRAIN_SCENARIOS} train / {EVAL_SCENARIOS} eval")
    log.info(f"  Difficulty: {DIFFICULTY_TRAIN_WEIGHTS}")
    log.info(f"  Epochs: {NUM_EPOCHS}  |  LR: {LEARNING_RATE}")
    log.info(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUM}  |  Generations: {NUM_GENERATIONS}")
    log.info("")

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading model (Unsloth 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    log.info("Model loaded and LoRA adapters applied.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_dataset, eval_dataset = build_dataset()
    log.info(f"Datasets ready: {len(train_dataset)} train / {len(eval_dataset)} eval")

    # ── GRPO Config ───────────────────────────────────────────────────────────
    use_bf16 = torch.cuda.is_bf16_supported()
    log.info(f"Precision: {'bf16' if use_bf16 else 'fp16'}")

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,

        # Batch / accumulation
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM,

        # Training duration
        num_train_epochs=NUM_EPOCHS,

        # Optimiser
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,

        # GRPO sampling
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        max_prompt_length=2800,
        temperature=TEMPERATURE,
        top_p=TOP_P,

        # KL penalty
        beta=BETA,

        # Logging & checkpointing
        logging_steps=10,
        save_steps=50,           # ↓ was 100 — more granular recovery
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=5,      # keep last 5 checkpoints

        # Precision
        fp16=not use_bf16,
        bf16=use_bf16,

        # Reporting
        report_to="wandb",
        run_name=RUN_TAG,

        # Misc
        seed=SEED,
        dataloader_num_workers=0,
        use_vllm=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    reward_fn = build_reward_fn()

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_fn],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training...")
    log.info(f"Total steps ≈ {(TRAIN_SCENARIOS * NUM_EPOCHS) // (BATCH_SIZE * GRADIENT_ACCUM * NUM_GENERATIONS)}")

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_path = f"{OUTPUT_DIR}/final"
    log.info(f"Saving final model to: {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # ── Push to HF Hub ────────────────────────────────────────────────────────
    if HF_REPO_ID:
        log.info(f"Pushing to HF Hub: {HF_REPO_ID}")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=final_path,
                repo_id=HF_REPO_ID,
                repo_type="model",
                commit_message=f"ConflictBench GRPO Run 2 — {RUN_TAG}",
            )
            log.info(f"Pushed to: https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            log.warning(f"HF Hub push failed: {e}. Checkpoint saved locally at {final_path}")

    log.info("")
    log.info("=" * 60)
    log.info("  Training complete!")
    log.info(f"  Checkpoint: {final_path}")
    log.info(f"  Model hub:  https://huggingface.co/{HF_REPO_ID}")
    log.info("=" * 60)


# ── Evaluation util ───────────────────────────────────────────────────────────

def evaluate_model(model_path: str, n: int = 60):
    """Quick before/after evaluation across all 3 difficulties."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Evaluating: {model_path}  ({n} scenarios)")
    model_eval = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_path)
    gen = ScenarioGenerator(seed=999)

    totals = {d: {"composite": 0, "n": 0} for d in [1, 2, 3]}

    for _ in range(n):
        diff = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        sc = gen.generate(difficulty=diff)
        inputs = tok(sc.prompt, return_tensors="pt").to(model_eval.device)
        with torch.no_grad():
            out = model_eval.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        bd = compute_score(response, sc)
        totals[diff]["composite"] += bd.composite
        totals[diff]["n"] += 1

    print("\n=== Evaluation Results ===")
    for d in [1, 2, 3]:
        n_d = totals[d]["n"]
        if n_d > 0:
            avg = totals[d]["composite"] / n_d
            print(f"  Difficulty {d}: avg composite = {avg:.4f}  (n={n_d})")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=str, default=None,
                   help="Path to trained model checkpoint for evaluation")
    p.add_argument("--n-eval", type=int, default=60)
    args = p.parse_args()

    if args.eval:
        evaluate_model(args.eval, args.n_eval)
    else:
        main()