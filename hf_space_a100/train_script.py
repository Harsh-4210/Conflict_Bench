"""
ConflictBench — GRPO Training Script for HF Space (A100 Target)
Tailored for NVIDIA A100 (48GB VRAM)
"""

import os
import sys
from unsloth import FastLanguageModel # Must be before transformers
import json
import random
import logging
import time
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

import torch
from datasets import Dataset
from transformers import TrainerCallback

# Setup Paths
REPO_URL = "https://github.com/Harsh-4210/Conflict_Bench.git"
REPO_DIR = Path("/tmp/conflictbench_repo")
OUTPUT_DIR = Path("./grpo-out")
PLOTS_DIR = OUTPUT_DIR / "plots"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")

def clone_repo():
    """Clone the ConflictBench repo if not already present."""
    if REPO_DIR.exists():
        log.info("Repo already cloned, pulling latest...")
        try:
            subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)
        except:
            log.warning("Pull failed, deleting and re-cloning...")
            shutil.rmtree(REPO_DIR)
            subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    else:
        log.info(f"Cloning {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    log.info("✅ Repo ready")

# ── Config (Run 2 Parameters) ────────────────────────────────────────────────

TRAIN_SCENARIOS  = 300    # ↓ was 600
EVAL_SCENARIOS   = 10     # ↓ was 30
SEED             = 42
 
DIFFICULTY_TRAIN_WEIGHTS = {1: 0.60, 2: 0.40}
DIFFICULTY_EVAL_WEIGHTS  = {1: 0.40, 2: 0.40, 3: 0.20}
 
NUM_EPOCHS       = 2      # ↓ was 3
BATCH_SIZE       = 1
GRADIENT_ACCUM   = 4      # ↓ was 8
NUM_GENERATIONS  = 4      # ↓ was 8  ← BIGGEST SPEED WIN
MAX_NEW_TOKENS   = 512    # ↓ was 768
MAX_PROMPT_CHARS = 2800
LEARNING_RATE    = 3e-6
WARMUP_STEPS     = 20
BETA             = 0.04
TEMPERATURE      = 0.85
TOP_P            = 0.92
 
HF_REPO_ID = os.getenv("HF_REPO_ID", None)
HF_TOKEN   = os.getenv("HF_TOKEN",   None)

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

# Global list to collect training metrics for plotting
_training_log = []

# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset():
    from generator import ScenarioGenerator
    gen = ScenarioGenerator(seed=SEED)

    def make_records(n: int, diff_weights: Dict[int, float]) -> List[Dict[str, Any]]:
        records = []
        attempts = 0
        while len(records) < n and attempts < n * 5:
            attempts += 1
            diff = random.choices(list(diff_weights.keys()), weights=list(diff_weights.values()))[0]
            scenario = gen.generate(difficulty=diff)
            
            if len(scenario.prompt) > MAX_PROMPT_CHARS:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": scenario.prompt},
            ]

            records.append({
                "prompt": messages,
                "scenario_json": json.dumps({
                    "ground_truth_followed": scenario.ground_truth_followed,
                    "ground_truth_overridden": scenario.ground_truth_overridden,
                    "difficulty": scenario.difficulty,
                    "conflicts": [
                        {
                            "instruction_a_id": c.instruction_a_id,
                            "instruction_b_id": c.instruction_b_id,
                            "conflict_type": c.conflict_type,
                            "resolution_id": c.resolution_id,
                            "explanation": c.explanation,
                        } for c in scenario.conflicts
                    ],
                    "instructions": [
                        {
                            "id": ins.id,
                            "action_key": ins.action_key,
                            "action_value": ins.action_value,
                            "source_priority": ins.source_priority,
                        } for ins in scenario.instructions
                    ],
                }),
            })
        return records

    train_records = make_records(TRAIN_SCENARIOS, DIFFICULTY_TRAIN_WEIGHTS)
    eval_records = make_records(EVAL_SCENARIOS, DIFFICULTY_EVAL_WEIGHTS)
    return Dataset.from_list(train_records), Dataset.from_list(eval_records)

# ── Reward function ───────────────────────────────────────────────────────────

def build_reward_fn():
    from generator import Scenario, Instruction, ConflictPair
    from verifier import score as _score

    RUBRIC_WEIGHTS_V2 = {
        "correct_final_state": 0.35,
        "no_contradictions": 0.25,
        "conflict_identification": 0.20,
        "efficiency": 0.10,
        "format_compliance": 0.10,
    }

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        scenario_jsons = kwargs.get("scenario_json", [None] * len(completions))

        for completion, scenario_json in zip(completions, scenario_jsons):
            if scenario_json is None:
                rewards.append(0.0)
                continue
            try:
                data = json.loads(scenario_json)
                instructions = [Instruction(id=ins["id"], text="", source="", source_priority=ins["source_priority"],
                                            instruction_type="absolute", action_key=ins["action_key"],
                                            action_value=ins["action_value"]) for ins in data["instructions"]]
                conflicts = [ConflictPair(instruction_a_id=c["instruction_a_id"], instruction_b_id=c["instruction_b_id"],
                                          conflict_type=c["conflict_type"], resolution_id=c["resolution_id"],
                                          explanation=c["explanation"]) for c in data["conflicts"]]
                scenario = Scenario(scenario_id="", domain="", difficulty=data.get("difficulty", 1), 
                                    business_context="", instructions=instructions, conflicts=conflicts,
                                    ground_truth_followed=data["ground_truth_followed"],
                                    ground_truth_overridden=data["ground_truth_overridden"], prompt="")

                breakdown = _score(completion, scenario)
                composite_v2 = (
                    RUBRIC_WEIGHTS_V2["correct_final_state"] * breakdown.correct_final_state +
                    RUBRIC_WEIGHTS_V2["no_contradictions"] * breakdown.no_contradictions +
                    RUBRIC_WEIGHTS_V2["conflict_identification"] * breakdown.conflict_identification +
                    RUBRIC_WEIGHTS_V2["efficiency"] * breakdown.efficiency +
                    RUBRIC_WEIGHTS_V2["format_compliance"] * breakdown.format_compliance
                )
                
                diff_bonus = 0.05 if data.get("difficulty", 1) == 2 else 0.0
                final_reward = min(1.0, composite_v2 + diff_bonus * composite_v2)
                rewards.append(round(final_reward, 4))
            except Exception as e:
                rewards.append(0.0)
        return rewards

    return reward_fn

# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logs = _training_log
    if not logs: return []

    steps = [l["step"] for l in logs]
    plot_paths = []

    # Reward
    fig, ax = plt.subplots(figsize=(10, 5))
    train_r = [(l["step"], l["reward"]) for l in logs if "reward" in l]
    eval_r = [(l["step"], l["eval_reward"]) for l in logs if "eval_reward" in l]
    if train_r: ax.plot(*zip(*train_r), label="Train Reward")
    if eval_r: ax.plot(*zip(*eval_r), label="Eval Reward")
    ax.set_title("Reward Curve"); ax.legend(); ax.grid(True)
    p = PLOTS_DIR / "reward_curve.png"; fig.savefig(p); plt.close(fig); plot_paths.append(str(p))

    # Loss
    fig, ax = plt.subplots(figsize=(10, 5))
    loss = [(l["step"], l["loss"]) for l in logs if "loss" in l]
    if loss: ax.plot(*zip(*loss), color="red")
    ax.set_title("Training Loss"); ax.grid(True)
    p = PLOTS_DIR / "loss_curve.png"; fig.savefig(p); plt.close(fig); plot_paths.append(str(p))

    # KL
    fig, ax = plt.subplots(figsize=(10, 5))
    kl = [(l["step"], l["kl"]) for l in logs if "kl" in l]
    if kl: ax.plot(*zip(*kl), color="purple")
    ax.set_title("KL Divergence"); ax.grid(True)
    p = PLOTS_DIR / "kl_divergence.png"; fig.savefig(p); plt.close(fig); plot_paths.append(str(p))

    # Combined
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.flat, ["reward", "loss", "kl", "eval_reward"]):
        pts = [(l["step"], l[key]) for l in logs if key in l]
        if pts: ax.plot(*zip(*pts))
        ax.set_title(key); ax.grid(True)
    fig.tight_layout()
    p = PLOTS_DIR / "training_dashboard.png"; fig.savefig(p); plt.close(fig); plot_paths.append(str(p))

    return plot_paths

# ── Main Training Loop ────────────────────────────────────────────────────────

def run_training(progress_callback=None):
    def emit(msg):
        if progress_callback: progress_callback(msg)
        log.info(msg)

    _training_log.clear()
    clone_repo()

    from trl import GRPOTrainer, GRPOConfig

    emit("🚀 Loading model (Unsloth 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        max_seq_length=4096,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )

    # Cast LoRA params to avoid Half/Float mismatch
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for name, param in model.named_parameters():
        if "lora_" in name: param.data = param.data.to(compute_dtype)
    # Patch for compatibility with recent TRL GRPOTrainer
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    emit(f"✅ LoRA applied (dtype: {compute_dtype})")

    emit("📊 Building datasets...")
    train_dataset, eval_dataset = build_dataset()
    emit(f"✅ Datasets ready: {len(train_dataset)} train / {len(eval_dataset)} eval")

    class GradioLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                entry = {"step": state.global_step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
                _training_log.append(entry)
                
                # Format learning rate with scientific notation, others with 4 decimals
                parts = []
                for k, v in entry.items():
                    if k == "step": continue
                    if "learning_rate" in k:
                        parts.append(f"{k}={v:.2e}")
                    else:
                        parts.append(f"{k}={v:.4f}")
                
                msg = f"Step {state.global_step}: " + " | ".join(parts)
                emit(msg)

    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=NUM_GENERATIONS, # Unsloth forces batch_size == num_generations
        per_device_eval_batch_size=NUM_GENERATIONS,
        gradient_accumulation_steps=1, # Keep effective batch size at 8 (8*1=8)
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        max_prompt_length=MAX_PROMPT_CHARS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        beta=BETA,
        logging_steps=10,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=SEED,
    )

    trainer = GRPOTrainer(
        model=model, args=grpo_config, tokenizer=tokenizer,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        reward_funcs=[build_reward_fn()],
    )
    trainer.add_callback(GradioLogCallback())

    emit("🔥 Starting GRPO training...")
    trainer.train()

    emit("💾 Saving final model...")
    final_path = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    if HF_REPO_ID and HF_TOKEN:
        emit(f"☁️ Uploading to {HF_REPO_ID}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
            api.upload_folder(folder_path=str(final_path), repo_id=HF_REPO_ID, repo_type="model")
            emit(f"✅ Uploaded to https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            emit(f"❌ Upload failed: {e}")

    emit("📈 Generating plots...")
    return generate_plots()

if __name__ == "__main__":
    run_training()
