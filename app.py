"""
ConflictBench — HuggingFace Spaces Demo (Gradio)
Showcases before/after training: base model vs fine-tuned model side by side.
Live scoring with animated reward breakdown.

Deploy: push this file + requirements to a HF Space (SDK: gradio).
"""

import sys
import os
import json
import random
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).parent))
from generator import ScenarioGenerator
from verifier import score as compute_score, parse_agent_output


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TRAINED_MODEL_ID = os.getenv("TRAINED_MODEL_ID", None)  # set this env var in HF Space settings

_base_model = None
_base_tokenizer = None
_trained_model = None
_trained_tokenizer = None
_generator = ScenarioGenerator(seed=None)
def load_models():
    global _base_model, _base_tokenizer, _trained_model, _trained_tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading base model...")
    _base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    _base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    if TRAINED_MODEL_ID:
        print(f"Loading trained model from {TRAINED_MODEL_ID}...")
        _trained_tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_ID)
        _trained_model = AutoModelForCausalLM.from_pretrained(
            TRAINED_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
    else:
        print("No trained model ID set. Demo will run base model only.")


def infer(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    if model is None:
        return '{"error": "Model not loaded"}'
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


# ---------------------------------------------------------------------------
# Score & JSON Formatters (HTML/CSS)
# ---------------------------------------------------------------------------

def format_score_html(breakdown_dict: dict) -> str:
    rubric_labels = {
        "correct_final_state":    "Correct Final State",
        "no_contradictions":      "No Contradictions",
        "conflict_identification": "Conflict Identification",
        "efficiency":             "Plan Efficiency",
        "format_compliance":      "Format Compliance",
    }
    
    html = '<div class="score-container">'
    for k, label in rubric_labels.items():
        v = breakdown_dict.get(k, 0.0)
        pct = int(v * 100)
        color = "#10b981" if pct == 100 else ("#f59e0b" if pct > 0 else "#ef4444")
        html += f'''
        <div class="score-row">
            <div class="score-label">{label}</div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width: {pct}%; background-color: {color};"></div>
            </div>
            <div class="score-value">{v:.2f}</div>
        </div>
        '''
    
    composite = breakdown_dict.get("composite", 0.0)
    comp_pct = int(composite * 100)
    comp_color = "#3b82f6" if comp_pct > 80 else ("#f59e0b" if comp_pct > 40 else "#ef4444")
    html += f'''
        <div class="score-row composite-row">
            <div class="score-label" style="font-weight: bold;">Composite Score</div>
            <div class="score-bar-bg" style="height: 12px;">
                <div class="score-bar-fill" style="width: {comp_pct}%; background-color: {comp_color}; height: 12px;"></div>
            </div>
            <div class="score-value" style="font-weight: bold;">{composite:.3f}</div>
        </div>
    </div>
    '''
    return html


def format_json_html(raw: str) -> str:
    """Renders the parsed JSON into a beautiful HTML card format."""
    parsed = parse_agent_output(raw)
    if parsed is None:
        return f'<div class="error-box"><strong>Failed to parse JSON.</strong> Raw output:<br><pre>{raw}</pre></div>'
    
    html = '<div class="json-visualizer">'
    
    # Execution Plan
    html += '<div class="card"><div class="card-title">🚀 Execution Plan</div><ul class="instruction-list">'
    if not parsed["execution_plan"]:
        html += '<li><em>None</em></li>'
    else:
        for ins in parsed["execution_plan"]:
            html += f'<li><span class="ins-badge accept">{ins}</span></li>'
    html += '</ul></div>'
    
    # Overridden
    html += '<div class="card"><div class="card-title">🚫 Overridden Instructions</div><ul class="instruction-list">'
    if not parsed["overridden_instructions"]:
        html += '<li><em>None</em></li>'
    else:
        for ins in parsed["overridden_instructions"]:
            html += f'<li><span class="ins-badge reject">{ins}</span></li>'
    html += '</ul></div>'
    
    # Conflicts
    html += '<div class="card"><div class="card-title">⚔️ Identified Conflicts</div>'
    if not parsed["identified_conflicts"]:
        html += '<em>No conflicts identified.</em>'
    else:
        for c in parsed["identified_conflicts"]:
            res = c.get("resolution", "UNKNOWN")
            res_class = "accept" if res != "UNKNOWN" else "reject"
            html += f'''
            <div class="conflict-box">
                <div class="conflict-header">
                    <span class="ins-badge conflict-a">{c.get("instruction_a", "?")}</span> 
                    <span class="vs">vs</span> 
                    <span class="ins-badge conflict-b">{c.get("instruction_b", "?")}</span>
                </div>
                <div class="conflict-type"><strong>Type:</strong> {c.get("conflict_type", "N/A")}</div>
                <div class="conflict-reasoning">"{c.get("reasoning", "No reasoning provided.")}"</div>
                <div class="conflict-resolution"><strong>Resolution:</strong> <span class="ins-badge {res_class}">{res}</span></div>
            </div>
            '''
    html += '</div></div>'
    return html


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def generate_scenario(difficulty: int):
    scenario = _generator.generate(difficulty=difficulty)
    preview = scenario.prompt
    meta = (
        f"**Domain:** {scenario.domain} | "
        f"**Difficulty Level:** {scenario.difficulty} | "
        f"**Total Instructions:** {len(scenario.instructions)} | "
        f"**Embedded Conflicts:** {len(scenario.conflicts)}"
    )
    return scenario, preview, meta, gr.update(interactive=True), gr.update(interactive=True)


def run_base_model(scenario):
    if scenario is None:
        gr.Warning("Please generate a scenario first!")
        return "", ""
    gr.Info("Base model generating response...")
    response = infer(_base_model, _base_tokenizer, scenario.prompt)
    breakdown = compute_score(response, scenario)
    return format_json_html(response), format_score_html(breakdown.to_dict())


def run_trained_model(scenario):
    if scenario is None:
        gr.Warning("Please generate a scenario first!")
        return "", ""
    if _trained_model is None:
        gr.Warning("Trained model is not loaded. Ensure TRAINED_MODEL_ID is set.")
        return "", ""
    gr.Info("Fine-tuned model generating response...")
    response = infer(_trained_model, _trained_tokenizer, scenario.prompt)
    breakdown = compute_score(response, scenario)
    return format_json_html(response), format_score_html(breakdown.to_dict())


def show_ground_truth(scenario):
    if scenario is None:
        return '<div class="error-box">Generate a scenario first.</div>'
    gt_json = json.dumps({
        "identified_conflicts": [
            {
                "instruction_a": c.instruction_a_id,
                "instruction_b": c.instruction_b_id,
                "conflict_type": c.conflict_type,
                "resolution": c.resolution_id,
                "reasoning": c.explanation,
            }
            for c in scenario.conflicts
        ],
        "execution_plan": scenario.ground_truth_followed,
        "overridden_instructions": scenario.ground_truth_overridden,
    })
    return format_json_html(gt_json)


# ---------------------------------------------------------------------------
# CSS & Styling
# ---------------------------------------------------------------------------

custom_css = """
/* Glassmorphic & Modern Dark Theme Elements */
body { font-family: 'Inter', 'Segoe UI', sans-serif; }
.score-container { background: #1f2937; padding: 15px; border-radius: 8px; border: 1px solid #374151; margin-top: 10px; }
.score-row { display: flex; align-items: center; margin-bottom: 8px; font-size: 14px; color: #e5e7eb; }
.score-label { width: 160px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.score-bar-bg { flex-grow: 1; background: #374151; height: 8px; border-radius: 4px; margin: 0 15px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease-in-out; }
.score-value { width: 40px; text-align: right; font-family: monospace; }
.composite-row { margin-top: 12px; padding-top: 12px; border-top: 1px solid #4b5563; }

.json-visualizer { font-family: 'Inter', sans-serif; font-size: 14px; color: #d1d5db; }
.card { background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 15px; margin-bottom: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
.card-title { font-size: 16px; font-weight: 600; margin-bottom: 10px; color: #f9fafb; border-bottom: 1px solid #374151; padding-bottom: 5px; }
.instruction-list { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 8px; }
.ins-badge { padding: 4px 8px; border-radius: 6px; font-family: monospace; font-size: 12px; font-weight: bold; }
.ins-badge.accept { background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #34d399; }
.ins-badge.reject { background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #f87171; }
.ins-badge.conflict-a { background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; color: #60a5fa; }
.ins-badge.conflict-b { background: rgba(139, 92, 246, 0.2); border: 1px solid #8b5cf6; color: #a78bfa; }

.conflict-box { background: #1f2937; border-left: 3px solid #f59e0b; padding: 12px; margin-bottom: 10px; border-radius: 0 6px 6px 0; }
.conflict-header { margin-bottom: 8px; }
.vs { margin: 0 8px; color: #9ca3af; font-size: 12px; font-style: italic; }
.conflict-reasoning { font-style: italic; color: #9ca3af; margin: 8px 0; padding-left: 10px; border-left: 2px solid #4b5563; }
.error-box { background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; color: #f87171; padding: 15px; border-radius: 8px; }
"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="ConflictBench — Instruction Priority Resolver") as demo:
    
    # State variable for concurrent usage
    scenario_state = gr.State(None)

    # Header
    with gr.Row():
        gr.HTML("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="font-weight: 800; font-size: 2.5rem; background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ⚔️ ConflictBench
                </h1>
                <p style="font-size: 1.1rem; color: #9ca3af;">The AI "Middle Management" Stress Test</p>
            </div>
        """)

    with gr.Row():
        # LEFT SIDEBAR: Controls & Info
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Scenario Generation")
                difficulty_slider = gr.Slider(
                    minimum=1, maximum=3, step=1, value=2, 
                    label="Difficulty Profile", 
                    info="1 = Easy (2 Conflicts), 2 = Medium (4), 3 = Hard (6)"
                )
                gen_btn = gr.Button("🎲 Generate New Scenario", variant="primary", size="lg")
                
            scenario_meta = gr.Markdown("<div style='padding: 10px; text-align: center; color: #9ca3af;'>*Initialize a scenario to begin*</div>")
            
            with gr.Accordion("📚 Document Preview", open=True):
                scenario_display = gr.Textbox(
                    label="Raw Business Instructions", 
                    lines=18, interactive=False, 
                    elem_classes="glass-textbox"
                )

            gr.Markdown(
                """
                ---
                **Links:**
                [🤗 HuggingFace Space](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench) |
                [🐙 GitHub Repo](https://github.com/Harsh-4210/Conflict_Bench)
                """
            )

        # RIGHT MAIN AREA: Models
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("⚔️ Arena (Side-by-Side)"):
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("<h3 style='text-align:center; color: #9ca3af;'>Base Model</h3><p style='text-align:center; font-size: 12px; margin-top:-10px;'>Qwen2.5-3B-Instruct</p>")
                            base_btn = gr.Button("▶️ Run Base Model", interactive=False)
                            base_score = gr.HTML("<div class='score-container'><div style='text-align:center; color:#6b7280;'>Awaiting inference...</div></div>")
                            base_output = gr.HTML("<div class='card' style='text-align:center; color:#6b7280;'>Output will appear here</div>")

                        with gr.Column():
                            gr.HTML("<h3 style='text-align:center; color: #34d399;'>Fine-Tuned Model</h3><p style='text-align:center; font-size: 12px; margin-top:-10px;'>ConflictBench Policy</p>")
                            trained_btn = gr.Button("🚀 Run Fine-Tuned Model", interactive=False, variant="primary")
                            trained_score = gr.HTML("<div class='score-container'><div style='text-align:center; color:#6b7280;'>Awaiting inference...</div></div>")
                            trained_output = gr.HTML("<div class='card' style='text-align:center; color:#6b7280;'>Output will appear here</div>")

                with gr.TabItem("✅ Ground Truth Inspector"):
                    gr.Markdown("Deep dive into the hidden conflicts of the currently generated scenario.")
                    gt_btn = gr.Button("Reveal Ground Truth Solutions")
                    gt_output = gr.HTML("<div class='card' style='text-align:center; color:#6b7280;'>Click reveal to see correct logic.</div>")

                with gr.TabItem("ℹ️ Rubric & Methodology"):
                    gr.Markdown("""
                    ### How Agents Are Evaluated
                    
                    ConflictBench tests if an LLM can understand implicit corporate hierarchy. 
                    If two instructions conflict, the agent MUST resolve it by deferring to the higher authority:
                    
                    **Hierarchy:** `Legal/Regulatory > C-Suite > VP > Director > Manager > Team Lead`
                    
                    The agent receives a strict JSON constraint. We score:
                    1. **Format:** Is it parsable JSON?
                    2. **Identification:** Did it find all the hidden conflicts?
                    3. **Contradictions:** Did it mistakenly execute conflicting directives?
                    4. **Final State:** Did it execute the highest-authority directives?
                    """)

    # Events wiring
    gen_btn.click(
        fn=generate_scenario,
        inputs=[difficulty_slider],
        outputs=[scenario_state, scenario_display, scenario_meta, base_btn, trained_btn],
    ).then(
        fn=lambda: ("<div class='score-container'><div style='text-align:center; color:#6b7280;'>Awaiting inference...</div></div>", 
                    "<div class='score-container'><div style='text-align:center; color:#6b7280;'>Awaiting inference...</div></div>",
                    "<div class='card' style='text-align:center; color:#6b7280;'>Output will appear here</div>",
                    "<div class='card' style='text-align:center; color:#6b7280;'>Output will appear here</div>"),
        outputs=[base_score, trained_score, base_output, trained_output]
    )
    
    base_btn.click(fn=run_base_model, inputs=[scenario_state], outputs=[base_output, base_score])
    trained_btn.click(fn=run_trained_model, inputs=[scenario_state], outputs=[trained_output, trained_score])
    gt_btn.click(fn=show_ground_truth, inputs=[scenario_state], outputs=[gt_output])


# Load models on startup
try:
    load_models()
except Exception as e:
    print(f"Model loading deferred: {e}")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Base(), css=custom_css)
