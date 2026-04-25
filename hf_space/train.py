"""
ConflictBench — GRPO Training + Plot Generation for HF Space
Clones repo, trains model, generates presentation plots, uploads adapter.
"""

import os, sys, json, random, logging, time, subprocess, shutil
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Setup: clone repo and add to path
# ---------------------------------------------------------------------------

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
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)
    else:
        log.info(f"Cloning {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    sys.path.insert(0, str(REPO_DIR))
    log.info("✅ Repo ready")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_SCENARIOS = 400
EVAL_SCENARIOS = 60
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 768
MAX_PROMPT_LENGTH = 3200
LEARNING_RATE = 3e-6
WARMUP_RATIO = 0.05
BETA = 0.02
SEED = 42
TRAIN_DIFFICULTY_MIX = {1: 0.8, 2: 0.2}
HF_REPO_ID = os.getenv("HF_REPO_ID", "Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora")
HF_TOKEN = os.getenv("HF_TOKEN", None)
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 5


def detect_gpu_config():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f}GB")

    config = {
        "model_name": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "load_in_4bit": True,
        "batch_size": 1, "gradient_accumulation": 8,
        "num_generations": 4, "lora_r": 32,
    }
    if vram_gb >= 40:
        config.update({"model_name": "unsloth/Qwen2.5-3B-Instruct",
                        "load_in_4bit": False, "batch_size": 2, "gradient_accumulation": 4})
    elif vram_gb >= 20:
        config.update({"batch_size": 1, "gradient_accumulation": 8})
    else:
        config.update({"batch_size": 1, "gradient_accumulation": 4})
    return config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(n_train, n_eval):
    from generator import ScenarioGenerator
    from datasets import Dataset

    gen = ScenarioGenerator(seed=SEED)
    log.info(f"Generating {n_train} train + {n_eval} eval scenarios...")

    def make_records(n, difficulty):
        records, attempts = [], 0
        while len(records) < n and attempts < n * 3:
            attempts += 1
            diff = (difficulty if isinstance(difficulty, int)
                    else random.choices(list(difficulty.keys()), weights=list(difficulty.values()))[0])
            scenario = gen.generate(difficulty=diff)
            if len(scenario.prompt) > 4000:
                continue
            records.append({
                "prompt": [{"role": "user", "content": scenario.prompt}],
                "scenario_json": json.dumps({
                    "ground_truth_followed": scenario.ground_truth_followed,
                    "ground_truth_overridden": scenario.ground_truth_overridden,
                    "conflicts": [{"instruction_a_id": c.instruction_a_id, "instruction_b_id": c.instruction_b_id,
                                   "conflict_type": c.conflict_type, "resolution_id": c.resolution_id,
                                   "explanation": c.explanation} for c in scenario.conflicts],
                    "instructions": [{"id": ins.id, "action_key": ins.action_key,
                                      "action_value": ins.action_value, "source_priority": ins.source_priority}
                                     for ins in scenario.instructions],
                }),
            })
        return records

    train_records = make_records(n_train, TRAIN_DIFFICULTY_MIX)
    eval_records = make_records(n_eval, {1: 0.3, 2: 0.5, 3: 0.2})
    log.info(f"Dataset ready: {len(train_records)} train, {len(eval_records)} eval")
    return Dataset.from_list(train_records), Dataset.from_list(eval_records)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def build_reward_function():
    from generator import Scenario, Instruction, ConflictPair
    from verifier import score as _score

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        scenario_jsons = kwargs.get("scenario_json", [None] * len(completions))
        for completion, sj in zip(completions, scenario_jsons):
            if sj is None:
                rewards.append(0.0)
                continue
            try:
                data = json.loads(sj)
                instructions = [Instruction(id=i["id"], text="", source="", source_priority=i["source_priority"],
                                            instruction_type="absolute", action_key=i["action_key"],
                                            action_value=i["action_value"]) for i in data["instructions"]]
                conflicts = [ConflictPair(instruction_a_id=c["instruction_a_id"], instruction_b_id=c["instruction_b_id"],
                                          conflict_type=c["conflict_type"], resolution_id=c["resolution_id"],
                                          explanation=c["explanation"]) for c in data["conflicts"]]
                scenario = Scenario(scenario_id="", domain="", difficulty=1, business_context="",
                                    instructions=instructions, conflicts=conflicts,
                                    ground_truth_followed=data["ground_truth_followed"],
                                    ground_truth_overridden=data["ground_truth_overridden"], prompt="")
                rewards.append(_score(completion, scenario).composite)
            except Exception as e:
                log.warning(f"Reward error: {e}")
                rewards.append(0.0)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

# Global list to collect training metrics for plotting
_training_log = []


class MetricsCallback:
    """TRL-compatible callback that collects metrics each log step."""
    def __init__(self):
        self.logs = _training_log

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
            self.logs.append(entry)


def generate_plots():
    """Generate all presentation-quality training plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logs = _training_log
    if not logs:
        log.warning("No training logs to plot")
        return []

    steps = [l["step"] for l in logs]
    plot_paths = []

    # --- 1. Reward Curve ---
    rewards = [l.get("reward", l.get("eval_reward")) for l in logs]
    if any(r is not None for r in rewards):
        fig, ax = plt.subplots(figsize=(10, 5))
        train_r = [(s, r) for s, r in zip(steps, [l.get("reward") for l in logs]) if r is not None]
        eval_r = [(s, r) for s, r in zip(steps, [l.get("eval_reward") for l in logs]) if r is not None]
        if train_r:
            ax.plot(*zip(*train_r), "o-", label="Train Reward", color="#6366f1", markersize=3)
        if eval_r:
            ax.plot(*zip(*eval_r), "s-", label="Eval Reward", color="#f97316", markersize=4)
        ax.set_xlabel("Training Step"); ax.set_ylabel("Reward"); ax.set_title("ConflictBench — GRPO Reward Curve")
        ax.legend(); ax.grid(alpha=0.3)
        p = PLOTS_DIR / "reward_curve.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        plot_paths.append(str(p))

    # --- 2. Loss Curve ---
    losses = [l.get("loss") for l in logs]
    if any(v is not None for v in losses):
        fig, ax = plt.subplots(figsize=(10, 5))
        pts = [(s, v) for s, v in zip(steps, losses) if v is not None]
        if pts:
            ax.plot(*zip(*pts), "-", color="#ef4444", linewidth=1.5)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
        ax.grid(alpha=0.3)
        p = PLOTS_DIR / "loss_curve.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        plot_paths.append(str(p))

    # --- 3. KL Divergence ---
    kl = [l.get("kl") for l in logs]
    if any(v is not None for v in kl):
        fig, ax = plt.subplots(figsize=(10, 5))
        pts = [(s, v) for s, v in zip(steps, kl) if v is not None]
        if pts:
            ax.plot(*zip(*pts), "-", color="#8b5cf6", linewidth=1.5)
        ax.set_xlabel("Step"); ax.set_ylabel("KL Divergence"); ax.set_title("KL Divergence from Base Policy")
        ax.grid(alpha=0.3)
        p = PLOTS_DIR / "kl_divergence.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        plot_paths.append(str(p))

    # --- 4. Combined Dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ConflictBench GRPO — Training Dashboard", fontsize=14, fontweight="bold")
    metrics = [("reward", "Reward", "#6366f1"), ("loss", "Loss", "#ef4444"),
               ("kl", "KL Div", "#8b5cf6"), ("eval_reward", "Eval Reward", "#f97316")]
    for ax, (key, title, color) in zip(axes.flat, metrics):
        pts = [(l["step"], l[key]) for l in logs if key in l and l[key] is not None]
        if pts:
            ax.plot(*zip(*pts), "-o", color=color, markersize=2)
        ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = PLOTS_DIR / "training_dashboard.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    plot_paths.append(str(p))

    log.info(f"✅ Generated {len(plot_paths)} plots in {PLOTS_DIR}")
    return plot_paths


# ---------------------------------------------------------------------------
# Main training entry point (called from app.py)
# ---------------------------------------------------------------------------

def run_training(progress_callback=None):
    """Run full training pipeline. Yields log lines for streaming."""

    def emit(msg):
        log.info(msg)
        if progress_callback:
            progress_callback(msg)

    _training_log.clear()
    clone_repo()

    import torch
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    from transformers import TrainerCallback

    gpu_config = detect_gpu_config()
    emit(f"🖥️ GPU config: {json.dumps(gpu_config, indent=2)}")

    # Load model
    emit(f"📦 Loading model: {gpu_config['model_name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=gpu_config["model_name"],
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=gpu_config["load_in_4bit"],
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model, r=gpu_config["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=gpu_config["lora_r"], lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )
    emit("✅ Model loaded with LoRA")

    # Dataset
    emit("📊 Building datasets...")
    train_dataset, eval_dataset = build_dataset(TRAIN_SCENARIOS, EVAL_SCENARIOS)
    emit(f"✅ Datasets ready: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # Callback for logging to Gradio
    class GradioLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                entry = {"step": state.global_step}
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        entry[k] = v
                _training_log.append(entry)
                msg = f"Step {state.global_step}: " + " | ".join(f"{k}={v:.4f}" for k, v in entry.items() if k != "step")
                emit(msg)

    _report_to = "none"

    # GRPO config
    str_output = str(OUTPUT_DIR)
    grpo_config = GRPOConfig(
        output_dir=str_output,
        per_device_train_batch_size=gpu_config["batch_size"],
        per_device_eval_batch_size=gpu_config["num_generations"],
        gradient_accumulation_steps=gpu_config["gradient_accumulation"],
        num_train_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO, num_generations=gpu_config["num_generations"],
        max_completion_length=MAX_NEW_TOKENS, max_prompt_length=MAX_PROMPT_LENGTH,
        temperature=1.0, top_p=0.95,
        logging_steps=LOGGING_STEPS, save_steps=SAVE_STEPS,
        eval_strategy="steps", eval_steps=EVAL_STEPS,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        report_to=_report_to,
        seed=SEED, dataloader_num_workers=2, beta=BETA,
        save_total_limit=5,
    )

    reward_fn = build_reward_function()
    trainer = GRPOTrainer(
        model=model, args=grpo_config, tokenizer=tokenizer,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        reward_funcs=[reward_fn],
    )
    trainer.add_callback(GradioLogCallback())

    emit("🚀 Starting GRPO training...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    emit(f"⏱️ Training completed in {elapsed/3600:.1f} hours")

    # Save
    final_dir = str(OUTPUT_DIR / "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    emit(f"💾 Model saved to {final_dir}")

    # Generate plots
    emit("📈 Generating presentation plots...")
    plot_paths = generate_plots()

    # Upload to HF Hub
    if HF_REPO_ID and HF_TOKEN:
        emit(f"☁️ Uploading to {HF_REPO_ID}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
            best_ckpt = None
            for d in sorted(Path(str_output).glob("checkpoint-*"), reverse=True):
                if (d / "adapter_model.safetensors").exists():
                    best_ckpt = d; break
            upload_path = str(best_ckpt) if best_ckpt else final_dir
            api.upload_folder(folder_path=upload_path, repo_id=HF_REPO_ID, repo_type="model",
                              commit_message=f"ConflictBench GRPO - {TRAIN_SCENARIOS} scenarios, {NUM_EPOCHS} epochs",
                              ignore_patterns=["optimizer.pt", "rng_state.pth", "scaler.pt", "scheduler.pt", "training_args.bin"])
            emit(f"✅ Uploaded to https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            emit(f"❌ Upload failed: {e}")

    emit("🎉 Training pipeline complete!")
    return plot_paths


if __name__ == "__main__":
    run_training()
