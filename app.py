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
_current_scenario = None


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
# Score formatter — returns markdown table of rubric scores
# ---------------------------------------------------------------------------

def format_score_md(breakdown_dict: dict) -> str:
    rubric_labels = {
        "correct_final_state":    "Correct final state",
        "no_contradictions":      "No contradictory actions",
        "conflict_identification": "Conflict identification",
        "efficiency":             "Plan efficiency",
        "format_compliance":      "Format compliance",
    }
    bar_chars = 10
    lines = ["| Rubric | Score | Bar |", "|---|---|---|"]
    for k, label in rubric_labels.items():
        v = breakdown_dict.get(k, 0.0)
        filled = int(round(v * bar_chars))
        bar = "█" * filled + "░" * (bar_chars - filled)
        lines.append(f"| {label} | {v:.2f} | {bar} |")

    composite = breakdown_dict.get("composite", 0.0)
    lines.append(f"| **Composite** | **{composite:.3f}** | |")
    return "\n".join(lines)


def format_json_pretty(raw: str) -> str:
    """Try to pretty-print JSON, fallback to raw."""
    try:
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.dumps(json.loads(m.group(0)), indent=2)
    except Exception:
        pass
    return raw


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def generate_scenario(difficulty: int):
    global _current_scenario
    _current_scenario = _generator.generate(difficulty=difficulty)
    preview = _current_scenario.prompt[:2000] + ("\n..." if len(_current_scenario.prompt) > 2000 else "")
    meta = (
        f"**Domain:** {_current_scenario.domain} | "
        f"**Difficulty:** {_current_scenario.difficulty} | "
        f"**Instructions:** {len(_current_scenario.instructions)} | "
        f"**Conflicts:** {len(_current_scenario.conflicts)}"
    )
    return preview, meta, gr.update(interactive=True), gr.update(interactive=True)


def run_base_model():
    if _current_scenario is None:
        return "Generate a scenario first.", "", ""
    response = infer(_base_model, _base_tokenizer, _current_scenario.prompt)
    breakdown = compute_score(response, _current_scenario)
    return format_json_pretty(response), format_score_md(breakdown.to_dict()), ""


def run_trained_model():
    if _current_scenario is None:
        return "Generate a scenario first.", "", ""
    if _trained_model is None:
        return "Trained model not loaded. Set TRAINED_MODEL_ID env var.", "", ""
    response = infer(_trained_model, _trained_tokenizer, _current_scenario.prompt)
    breakdown = compute_score(response, _current_scenario)
    return format_json_pretty(response), format_score_md(breakdown.to_dict()), ""


def show_ground_truth():
    if _current_scenario is None:
        return "Generate a scenario first."
    gt = {
        "identified_conflicts": [
            {
                "instruction_a": c.instruction_a_id,
                "instruction_b": c.instruction_b_id,
                "conflict_type": c.conflict_type,
                "resolution": c.resolution_id,
                "reasoning": c.explanation,
            }
            for c in _current_scenario.conflicts
        ],
        "execution_plan": _current_scenario.ground_truth_followed,
        "overridden_instructions": _current_scenario.ground_truth_overridden,
    }
    return json.dumps(gt, indent=2)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="ConflictBench — Instruction Priority Resolver",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # ConflictBench — Instruction Priority Resolution Environment
        **OpenEnv Hackathon 2026** | Theme 2: Long-Horizon Instruction Following + Scale AI bonus

        An LLM environment that trains agents to resolve conflicting business instructions
        according to an implicit stakeholder authority hierarchy.
        **Left = base Qwen2.5-3B-Instruct | Right = ConflictBench fine-tuned**
        """
    )

    with gr.Row():
        difficulty_slider = gr.Slider(
            minimum=1, maximum=3, step=1, value=2, label="Difficulty (1=2 conflicts, 2=4, 3=6)"
        )
        gen_btn = gr.Button("Generate New Scenario", variant="primary")

    scenario_meta = gr.Markdown("*No scenario loaded yet.*")
    scenario_display = gr.Textbox(
        label="Business Instruction Document", lines=14, interactive=False
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base Model (Qwen2.5-3B-Instruct)")
            base_btn = gr.Button("Run Base Model", interactive=False)
            base_output = gr.Textbox(label="Response", lines=10, interactive=False)
            base_score = gr.Markdown("*Score will appear here.*")

        with gr.Column():
            gr.Markdown("### Fine-tuned Model (ConflictBench)")
            trained_btn = gr.Button("Run Fine-tuned Model", interactive=False)
            trained_output = gr.Textbox(label="Response", lines=10, interactive=False)
            trained_score = gr.Markdown("*Score will appear here.*")

    with gr.Accordion("Show Ground Truth Answer", open=False):
        gt_btn = gr.Button("Reveal Ground Truth")
        gt_output = gr.Textbox(label="Ground Truth", lines=12, interactive=False)

    # Events
    gen_btn.click(
        fn=generate_scenario,
        inputs=[difficulty_slider],
        outputs=[scenario_display, scenario_meta, base_btn, trained_btn],
    )
    base_btn.click(fn=run_base_model, outputs=[base_output, base_score, gr.Textbox(visible=False)])
    trained_btn.click(fn=run_trained_model, outputs=[trained_output, trained_score, gr.Textbox(visible=False)])
    gt_btn.click(fn=show_ground_truth, outputs=[gt_output])

    gr.Markdown(
        """
        ---
        **Environment:** [HuggingFace Space](https://huggingface.co/spaces/your-username/conflictbench) |
        **Code:** [GitHub](https://github.com/your-username/conflictbench) |
        **Blog:** [HF Blog Post](https://huggingface.co/blog/your-username/conflictbench)
        """
    )


# Load models on startup
try:
    load_models()
except Exception as e:
    print(f"Model loading deferred: {e}")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
