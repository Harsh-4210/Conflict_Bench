import gradio as gr
import ast
import hmac
import threading
import time
import queue
import os
from pathlib import Path

LOG_QUEUE = queue.Queue()
TRAINING_LOCK = threading.Lock()
PLOT_PATHS = []

# Training costs real GPU money, so the button only works with this secret.
# Set it in the Space settings (Settings → Variables and secrets → TRAIN_PASSWORD).
TRAIN_PASSWORD = os.getenv("TRAIN_PASSWORD")


def read_train_config() -> dict:
    """Read the top-level constants of train_script.py without importing it (importing loads unsloth)."""
    tree = ast.parse((Path(__file__).parent / "train_script.py").read_text(encoding="utf-8"))
    config = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                config[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return config


CFG = read_train_config()


def _mix(weights: dict) -> str:
    return " / ".join(f"{int(w * 100)}% Diff-{d}" for d, w in weights.items())


def log_callback(msg):
    LOG_QUEUE.put(f"[{time.strftime('%H:%M:%S')}] {msg}")

def start_training(password):
    global PLOT_PATHS
    if not TRAIN_PASSWORD:
        yield "🔒 Training is disabled: TRAIN_PASSWORD is not set for this Space.", *[None]*4
        return
    if not hmac.compare_digest((password or "").encode(), TRAIN_PASSWORD.encode()):
        yield "🔒 Wrong password. Training not started.", *[None]*4
        return
    if not TRAINING_LOCK.acquire(blocking=False):
        yield "⚠️ Training already running...", *[None]*4
        return

    PLOT_PATHS.clear()

    def run():
        try:
            from train_script import run_training
            paths = run_training(progress_callback=log_callback)
            PLOT_PATHS.extend(paths or [])
            log_callback("🎉 Done! Check plots below.")
        except Exception as e:
            log_callback(f"❌ Training failed: {e}")
            import traceback
            log_callback(traceback.format_exc())
        finally:
            TRAINING_LOCK.release()
            LOG_QUEUE.put("<<FINISHED>>")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    
    logs = []
    while thread.is_alive() or not LOG_QUEUE.empty():
        try:
            msg = LOG_QUEUE.get(timeout=1.0)
            if msg == "<<FINISHED>>":
                continue
            logs.append(msg)
            if len(logs) > 300:
                logs = logs[-300:]
            
            # Prepare UI outputs
            current_log_text = "\n".join(logs)
            imgs = [p if os.path.exists(p) else None for p in PLOT_PATHS[:4]]
            while len(imgs) < 4:
                imgs.append(None)
                
            yield current_log_text, *imgs
        except queue.Empty:
            # Yield current state even if no new messages
            current_log_text = "\n".join(logs)
            imgs = [p if os.path.exists(p) else None for p in PLOT_PATHS[:4]]
            while len(imgs) < 4:
                imgs.append(None)
            yield current_log_text, *imgs

CSS = """
.main-title { text-align: center; margin-bottom: 0.5em; }
.log-box textarea { font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
                     font-size: 12px !important; background: #0d1117 !important;
                     color: #c9d1d9 !important; }
"""

with gr.Blocks(title="ConflictBench GRPO Trainer (A100)") as demo:
    gr.Markdown("# ⚔️ ConflictBench — GRPO Training Dashboard (A100 Target)", elem_classes="main-title")
    gr.Markdown("**One-click** production GRPO training script mapped to Run 2 parameters. "
                "Automatically streams logs and generates plots.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Run 2 Configuration")
            # Values are read from train_script.py so this table can't drift from the real config.
            # Batch and LoRA rank are set inline in run_training(), not as constants.
            gr.Markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Model | Qwen2.5-3B-Instruct (4-bit) |
            | Scenarios | {CFG.get('TRAIN_SCENARIOS')} train / {CFG.get('EVAL_SCENARIOS')} eval |
            | Curriculum | {_mix(CFG.get('DIFFICULTY_TRAIN_WEIGHTS', {}))} |
            | Generations | {CFG.get('NUM_GENERATIONS')} |
            | Max Output | {CFG.get('MAX_NEW_TOKENS')} tokens |
            | LoRA rank | 16 |
            | Epochs | {CFG.get('NUM_EPOCHS')} |
            | β (KL) | {CFG.get('BETA')} |
            | LR | {CFG.get('LEARNING_RATE')} |
            """)
            password_box = gr.Textbox(
                label="Training password",
                type="password",
                placeholder="Required: training uses paid GPU time",
                interactive=bool(TRAIN_PASSWORD),
            )
            start_btn = gr.Button(
                "🚀 Start Training" if TRAIN_PASSWORD else "🔒 Training disabled (no TRAIN_PASSWORD set)",
                variant="primary",
                size="lg",
                interactive=bool(TRAIN_PASSWORD),
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Live Training Logs")
            log_box = gr.Textbox(label="", lines=25, max_lines=25, interactive=False,
                                 elem_classes="log-box")

    gr.Markdown("---")
    gr.Markdown("### 📊 Training Plots")
    with gr.Row():
        plot1 = gr.Image(label="Reward Curve", type="filepath")
        plot2 = gr.Image(label="Loss Curve", type="filepath")
    with gr.Row():
        plot3 = gr.Image(label="KL Divergence", type="filepath")
        plot4 = gr.Image(label="Training Dashboard", type="filepath")

    # Start training and stream outputs to logs and plots
    start_btn.click(
        fn=start_training,
        inputs=[password_box],
        outputs=[log_box, plot1, plot2, plot3, plot4]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=CSS, ssr_mode=False)
