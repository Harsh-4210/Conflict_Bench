import gradio as gr
import threading
import time
import queue
import os

LOG_QUEUE = queue.Queue()
TRAINING_ACTIVE = False
PLOT_PATHS = []

def log_callback(msg):
    LOG_QUEUE.put(f"[{time.strftime('%H:%M:%S')}] {msg}")

def start_training():
    global TRAINING_ACTIVE, PLOT_PATHS
    if TRAINING_ACTIVE:
        yield "⚠️ Training already running...", *[None]*4
        return
        
    TRAINING_ACTIVE = True
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
            global TRAINING_ACTIVE
            TRAINING_ACTIVE = False
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
            gr.Markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Model | Qwen2.5-3B-Instruct |
            | Scenarios | 600 train / 30 eval |
            | Curriculum | 60% Diff-1 / 40% Diff-2 |
            | Global Batch | 8 (1 * 8 accum) |
            | Generations | 8 |
            | Max Output | 768 tokens |
            | LoRA rank | 16 |
            | Epochs | 3 |
            | β (KL) | 0.04 |
            | LR | 3e-6 |
            """)
            start_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            
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
        outputs=[log_box, plot1, plot2, plot3, plot4]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=CSS, ssr_mode=False)
