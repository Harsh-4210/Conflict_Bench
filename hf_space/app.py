"""
ConflictBench GRPO Trainer — Gradio UI
One-click training with live log streaming and plot display.
"""

import gradio as gr
import threading, time, os

LOGS = []
TRAINING_ACTIVE = False
PLOT_PATHS = []


def log_callback(msg):
    LOGS.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def get_logs():
    return "\n".join(LOGS[-200:])


def start_training():
    global TRAINING_ACTIVE, PLOT_PATHS
    if TRAINING_ACTIVE:
        return "⚠️ Training already in progress!", None, None, None, None

    TRAINING_ACTIVE = True
    LOGS.clear()
    PLOT_PATHS.clear()
    log_callback("Initializing training pipeline...")

    try:
        from train import run_training
        paths = run_training(progress_callback=log_callback)
        PLOT_PATHS.extend(paths or [])
        log_callback("🎉 Done! Check plots below.")
    except Exception as e:
        log_callback(f"❌ Training failed: {e}")
        import traceback
        log_callback(traceback.format_exc())
    finally:
        TRAINING_ACTIVE = False

    # Return plot images (up to 4)
    imgs = [p if os.path.exists(p) else None for p in PLOT_PATHS[:4]]
    while len(imgs) < 4:
        imgs.append(None)
    return get_logs(), *imgs


def start_training_async():
    """Launch training in background thread, return immediately."""
    global TRAINING_ACTIVE
    if TRAINING_ACTIVE:
        return "⚠️ Training already running. Check logs below."
    LOGS.clear()
    thread = threading.Thread(target=start_training, daemon=True)
    thread.start()
    return "🚀 Training started! Logs will stream below. This takes several hours."


def refresh_logs():
    return get_logs()


def refresh_plots():
    imgs = [p if os.path.exists(p) else None for p in PLOT_PATHS[:4]]
    while len(imgs) < 4:
        imgs.append(None)
    return imgs


CSS = """
.main-title { text-align: center; margin-bottom: 0.5em; }
.log-box textarea { font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
                     font-size: 12px !important; background: #0d1117 !important;
                     color: #c9d1d9 !important; }
"""

with gr.Blocks(css=CSS, title="ConflictBench GRPO Trainer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚔️ ConflictBench — GRPO Training Dashboard", elem_classes="main-title")
    gr.Markdown("**One-click** production GRPO training for instruction priority resolution. "
                "Clones repo → generates scenarios → trains model → plots → uploads adapter.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            gr.Markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Model | Qwen2.5-3B-Instruct |
            | Method | GRPO (6 rubrics) |
            | Scenarios | 120 train / 20 eval |
            | Curriculum | 100% Difficulty 1 |
            | LoRA rank | 16 |
            | Epochs | 2 |
            | β (KL) | 0.04 |
            """)
            start_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            status = gr.Textbox(label="Status", interactive=False, lines=2)

        with gr.Column(scale=2):
            gr.Markdown("### 📋 Live Training Logs")
            log_box = gr.Textbox(label="", lines=25, max_lines=25, interactive=False,
                                 elem_classes="log-box")
            refresh_btn = gr.Button("🔄 Refresh Logs", size="sm")

    gr.Markdown("---")
    gr.Markdown("### 📊 Training Plots")
    gr.Markdown("*Plots appear here after training completes.*")
    with gr.Row():
        plot1 = gr.Image(label="Reward Curve", type="filepath")
        plot2 = gr.Image(label="Loss Curve", type="filepath")
    with gr.Row():
        plot3 = gr.Image(label="KL Divergence", type="filepath")
        plot4 = gr.Image(label="Training Dashboard", type="filepath")
    refresh_plots_btn = gr.Button("🔄 Refresh Plots", size="sm")

    # Events
    start_btn.click(fn=start_training_async, outputs=status)
    refresh_btn.click(fn=refresh_logs, outputs=log_box)
    refresh_plots_btn.click(fn=refresh_plots, outputs=[plot1, plot2, plot3, plot4])

demo.launch(ssr_mode=False)
