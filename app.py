"""
ConflictBench — HuggingFace Spaces Demo (Gradio)
Premium UI Redesign — Dark Glassmorphism + Violet/Cyan Palette
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

BASE_MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
TRAINED_MODEL_ID = os.getenv("TRAINED_MODEL_ID", None)

_base_model      = None
_base_tokenizer  = None
_trained_model   = None
_trained_tokenizer = None
_generator       = ScenarioGenerator(seed=None)


def load_models():
    global _base_model, _base_tokenizer, _trained_model, _trained_tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading base model…")
    _base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    _base_model     = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    if TRAINED_MODEL_ID:
        print(f"Loading trained LoRA adapter from {TRAINED_MODEL_ID}…")
        try:
            from peft import PeftModel
            _trained_tokenizer = _base_tokenizer  # LoRA shares base tokenizer
            _trained_model = PeftModel.from_pretrained(
                _base_model,
                TRAINED_MODEL_ID,
            )
            _trained_model.eval()
            print("✅ LoRA adapter loaded and merged on top of base model.")
        except ImportError:
            print("⚠ peft not installed — falling back to base model for trained slot.")
            _trained_model = _base_model
            _trained_tokenizer = _base_tokenizer
        except Exception as e:
            print(f"⚠ Failed to load LoRA adapter: {e}")
            _trained_tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_ID)
            _trained_model = AutoModelForCausalLM.from_pretrained(
                TRAINED_MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
    else:
        print("No TRAINED_MODEL_ID set — demo runs base model only.")


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
# HTML Renderers
# ---------------------------------------------------------------------------

def _rubric_bar(label: str, value: float, weight_pct: int) -> str:
    pct   = int(value * 100)
    grade = "A" if pct >= 80 else ("B" if pct >= 60 else ("C" if pct >= 40 else "D"))
    color = (
        "#10b981" if pct >= 80 else
        "#06b6d4" if pct >= 60 else
        "#f59e0b" if pct >= 40 else
        "#f43f5e"
    )
    return f"""
    <div class="rb-row">
      <div class="rb-meta">
        <span class="rb-label">{label}</span>
        <span class="rb-weight">×{weight_pct}%</span>
      </div>
      <div class="rb-track">
        <div class="rb-fill" style="width:{pct}%;background:{color};"></div>
      </div>
      <div class="rb-right">
        <span class="rb-grade" style="color:{color};">{grade}</span>
        <span class="rb-val">{value:.2f}</span>
      </div>
    </div>"""


def format_score_html(bd: dict) -> str:
    composite = bd.get("composite", 0.0)
    cpct      = int(composite * 100)
    c_color   = "#10b981" if cpct >= 80 else ("#06b6d4" if cpct >= 60 else ("#f59e0b" if cpct >= 40 else "#f43f5e"))
    tier      = "EXCELLENT" if cpct >= 80 else ("GOOD" if cpct >= 60 else ("FAIR" if cpct >= 40 else "POOR"))

    bars = (
        _rubric_bar("Correct Final State",     bd.get("correct_final_state",     0.0), 35) +
        _rubric_bar("No Contradictions",        bd.get("no_contradictions",        0.0), 25) +
        _rubric_bar("Conflict Identification",  bd.get("conflict_identification",  0.0), 20) +
        _rubric_bar("Plan Efficiency",          bd.get("efficiency",               0.0), 10) +
        _rubric_bar("Format Compliance",        bd.get("format_compliance",        0.0), 10)
    )

    return f"""
    <div class="score-wrap">
      <div class="score-hero">
        <svg class="score-ring" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="10"/>
          <circle cx="60" cy="60" r="52" fill="none" stroke="{c_color}"
            stroke-width="10" stroke-linecap="round"
            stroke-dasharray="{int(327 * composite)} 327"
            transform="rotate(-90 60 60)" style="transition:stroke-dasharray 0.8s ease;"/>
        </svg>
        <div class="score-hero-text">
          <span class="score-num" style="color:{c_color};">{cpct}</span>
          <span class="score-denom">/100</span>
          <span class="score-tier" style="color:{c_color};">{tier}</span>
        </div>
      </div>
      <div class="rb-list">{bars}</div>
    </div>"""


def format_json_html(raw: str) -> str:
    parsed = parse_agent_output(raw)
    if parsed is None or not parsed.parsed_ok:
        raw_escaped = str(raw).replace("<", "&lt;").replace(">", "&gt;")
        return f"""
        <div class="parse-error">
          <div class="pe-title">⚠ Parse Failed</div>
          <pre class="pe-raw">{raw_escaped[:1200]}</pre>
        </div>"""

    # ── Execution Plan ──────────────────────────────────────────────────
    plan_pills = "".join(
        f'<span class="pill pill-follow">{ins}</span>'
        for ins in (parsed.execution_plan or [])
    ) or '<span class="empty-state">None</span>'

    # ── Overridden ──────────────────────────────────────────────────────
    over_pills = "".join(
        f'<span class="pill pill-override">{ins}</span>'
        for ins in (parsed.overridden_instructions or [])
    ) or '<span class="empty-state">None</span>'

    # ── Conflicts ───────────────────────────────────────────────────────
    conflicts_html = ""
    for c in (parsed.identified_conflicts or []):
        res       = c.get("resolution", "UNKNOWN")
        ctype     = c.get("conflict_type", "N/A")
        reasoning = c.get("reasoning", "No reasoning provided.")[:200]
        res_class = "pill-follow" if res != "UNKNOWN" else "pill-override"
        conflicts_html += f"""
        <div class="conflict-card">
          <div class="conflict-ids">
            <span class="pill pill-a">{c.get('instruction_a','?')}</span>
            <span class="vs-sep">⚡ vs</span>
            <span class="pill pill-b">{c.get('instruction_b','?')}</span>
          </div>
          <div class="conflict-row"><span class="meta-key">Type</span><span class="meta-val">{ctype}</span></div>
          <div class="conflict-reasoning">"{reasoning}"</div>
          <div class="conflict-row" style="margin-top:10px;">
            <span class="meta-key">Resolved by</span>
            <span class="pill {res_class}" style="font-size:11px;">{res}</span>
          </div>
        </div>"""

    if not conflicts_html:
        conflicts_html = '<span class="empty-state">No conflicts identified.</span>'

    return f"""
    <div class="jv-root">
      <div class="jv-section">
        <div class="jv-header"><span class="jv-icon">✅</span>Execution Plan
          <span class="jv-count">{len(parsed.execution_plan or [])}</span>
        </div>
        <div class="jv-pills">{plan_pills}</div>
      </div>
      <div class="jv-section">
        <div class="jv-header"><span class="jv-icon">🚫</span>Overridden
          <span class="jv-count">{len(parsed.overridden_instructions or [])}</span>
        </div>
        <div class="jv-pills">{over_pills}</div>
      </div>
      <div class="jv-section">
        <div class="jv-header"><span class="jv-icon">⚔️</span>Conflict Analysis
          <span class="jv-count">{len(parsed.identified_conflicts or [])}</span>
        </div>
        {conflicts_html}
      </div>
    </div>"""


def format_ground_truth_html(scenario) -> str:
    if scenario is None:
        return '<div class="parse-error"><div class="pe-title">Generate a scenario first.</div></div>'
    gt_json = json.dumps({
        "identified_conflicts": [
            {
                "instruction_a": c.instruction_a_id,
                "instruction_b": c.instruction_b_id,
                "conflict_type": c.conflict_type,
                "resolution":    c.resolution_id,
                "reasoning":     c.explanation,
            }
            for c in scenario.conflicts
        ],
        "execution_plan":          scenario.ground_truth_followed,
        "overridden_instructions": scenario.ground_truth_overridden,
    })
    return format_json_html(gt_json)


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def generate_scenario(difficulty: int):
    scenario = _generator.generate(difficulty=difficulty)
    diff_map = {1: "Easy", 2: "Medium", 3: "Hard"}
    meta_html = f"""
    <div class="meta-strip">
      <div class="meta-chip"><span class="mc-label">Domain</span><span class="mc-val">{scenario.domain}</span></div>
      <div class="meta-chip"><span class="mc-label">Difficulty</span>
        <span class="mc-val diff-{difficulty}">{diff_map.get(difficulty,'?')}</span></div>
      <div class="meta-chip"><span class="mc-label">Instructions</span>
        <span class="mc-val">{len(scenario.instructions)}</span></div>
      <div class="meta-chip"><span class="mc-label">Conflicts</span>
        <span class="mc-val">{len(scenario.conflicts)}</span></div>
    </div>"""
    reset_score = "<div class='score-wrap pending-state'>Run inference to see scores</div>"
    reset_out   = "<div class='jv-root pending-state'>Output will appear here</div>"
    return (
        scenario,
        scenario.prompt,
        meta_html,
        gr.update(interactive=True),
        gr.update(interactive=True),
        reset_score, reset_score,
        reset_out,   reset_out,
    )


def run_base_model(scenario):
    if scenario is None:
        gr.Warning("Generate a scenario first!")
        return "", ""
    gr.Info("Running base model inference…")
    response  = infer(_base_model, _base_tokenizer, scenario.prompt)
    breakdown = compute_score(response, scenario)
    return format_json_html(response), format_score_html(breakdown.to_dict())


def run_trained_model(scenario):
    if scenario is None:
        gr.Warning("Generate a scenario first!")
        return "", ""
    if _trained_model is None:
        gr.Warning("Trained model not loaded — set TRAINED_MODEL_ID env var.")
        return "", ""
    gr.Info("Running fine-tuned model inference…")
    response  = infer(_trained_model, _trained_tokenizer, scenario.prompt)
    breakdown = compute_score(response, scenario)
    return format_json_html(response), format_score_html(breakdown.to_dict())


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── Google Fonts ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root / Reset ──────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background: #06060a !important;
  color: #e2e8f0 !important;
}

.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

/* Hide Gradio branding */
footer { display: none !important; }

/* ── Scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0f0f18; }
::-webkit-scrollbar-thumb { background: #2d2d44; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4c4c6e; }

/* ── Hero Header ───────────────────────────────────────────────────── */
.cb-hero {
  text-align: center;
  padding: 48px 24px 36px;
  position: relative;
  overflow: hidden;
}
.cb-hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 70% 60% at 50% -10%, rgba(124,58,237,0.22) 0%, transparent 70%),
    radial-gradient(ellipse 50% 40% at 80% 50%,  rgba(6,182,212,0.10) 0%, transparent 60%);
  pointer-events: none;
}
.cb-wordmark {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}
.cb-wordmark-icon {
  width: 44px; height: 44px;
  background: linear-gradient(135deg, #7c3aed, #06b6d4);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  box-shadow: 0 0 28px rgba(124,58,237,0.45);
}
.cb-title {
  font-size: 2.1rem;
  font-weight: 700;
  letter-spacing: -0.8px;
  background: linear-gradient(135deg, #c4b5fd 0%, #67e8f9 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
}
.cb-subtitle {
  color: #64748b;
  font-size: 0.88rem;
  letter-spacing: 0.3px;
  margin: 6px 0 20px;
}
.cb-badges {
  display: flex;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}
.cb-badge {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.5px;
  border: 1px solid;
}
.cb-badge-purple { background: rgba(124,58,237,0.12); border-color: rgba(124,58,237,0.3); color: #a78bfa; }
.cb-badge-cyan   { background: rgba(6,182,212,0.10);  border-color: rgba(6,182,212,0.25);  color: #67e8f9; }
.cb-badge-amber  { background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.25); color: #fcd34d; }

/* ── Authority Hierarchy Banner ────────────────────────────────────── */
.hier-banner {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  margin: 0 0 24px;
  flex-wrap: wrap;
  padding: 10px 16px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
}
.hier-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.4px;
  white-space: nowrap;
}
.hier-1 { background: rgba(239,68,68,0.12);  color: #fca5a5; }
.hier-2 { background: rgba(245,158,11,0.10); color: #fcd34d; }
.hier-3 { background: rgba(16,185,129,0.10); color: #6ee7b7; }
.hier-4 { background: rgba(6,182,212,0.10);  color: #67e8f9; }
.hier-5 { background: rgba(124,58,237,0.10); color: #c4b5fd; }
.hier-6 { background: rgba(100,116,139,0.12);color: #94a3b8; }
.hier-arrow {
  color: #334155;
  font-size: 14px;
  margin: 0 2px;
}

/* ── Panel Cards ───────────────────────────────────────────────────── */
.panel-card {
  background: rgba(15, 15, 25, 0.7);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 14px;
  padding: 20px;
  margin-bottom: 12px;
}
.panel-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  font-weight: 600;
  color: #475569;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 7px;
}
.panel-title-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: linear-gradient(135deg, #7c3aed, #06b6d4);
}

/* ── Meta Strip ────────────────────────────────────────────────────── */
.meta-strip {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  padding: 12px 0 4px;
}
.meta-chip {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 14px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 8px;
  min-width: 80px;
}
.mc-label {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: #475569;
  margin-bottom: 3px;
}
.mc-val {
  font-size: 13px;
  font-weight: 600;
  color: #cbd5e1;
}
.diff-1 { color: #6ee7b7 !important; }
.diff-2 { color: #fcd34d !important; }
.diff-3 { color: #fca5a5 !important; }

/* ── Gradio Textbox override ───────────────────────────────────────── */
.gradio-textbox textarea {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 8px !important;
  color: #94a3b8 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
  line-height: 1.7 !important;
}

/* ── Buttons ───────────────────────────────────────────────────────── */
.btn-generate {
  background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.3px !important;
  box-shadow: 0 0 24px rgba(124,58,237,0.35) !important;
  transition: all 0.2s ease !important;
}
.btn-generate:hover { box-shadow: 0 0 36px rgba(124,58,237,0.5) !important; transform: translateY(-1px); }

.btn-base {
  background: rgba(100,116,139,0.15) !important;
  border: 1px solid rgba(100,116,139,0.25) !important;
  border-radius: 8px !important;
  color: #94a3b8 !important;
  font-weight: 500 !important;
}
.btn-trained {
  background: linear-gradient(135deg, rgba(6,182,212,0.2), rgba(124,58,237,0.2)) !important;
  border: 1px solid rgba(6,182,212,0.3) !important;
  border-radius: 8px !important;
  color: #67e8f9 !important;
  font-weight: 600 !important;
}

/* ── Model Column Headers ──────────────────────────────────────────── */
.model-header-base {
  text-align: center;
  padding: 14px 0 18px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  margin-bottom: 14px;
}
.model-header-trained {
  text-align: center;
  padding: 14px 0 18px;
  border-bottom: 1px solid rgba(6,182,212,0.15);
  margin-bottom: 14px;
}
.mh-tag {
  display: inline-block;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 4px;
  margin-bottom: 6px;
}
.mh-tag-base    { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.25); }
.mh-tag-trained { background: rgba(6,182,212,0.12);  color: #67e8f9; border: 1px solid rgba(6,182,212,0.3);  }
.mh-name {
  font-size: 14px;
  font-weight: 600;
  color: #e2e8f0;
  display: block;
}
.mh-sub {
  font-size: 11px;
  color: #475569;
  display: block;
  margin-top: 2px;
}

/* ── Score Widget ──────────────────────────────────────────────────── */
.score-wrap {
  padding: 16px;
  background: rgba(10,10,20,0.6);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px;
}
.pending-state {
  text-align: center;
  color: #334155;
  font-size: 13px;
  padding: 30px 16px;
}
.score-hero {
  position: relative;
  width: 120px;
  height: 120px;
  margin: 0 auto 16px;
}
.score-ring {
  width: 120px; height: 120px;
}
.score-hero-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  line-height: 1;
}
.score-num  { font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.score-denom{ font-size: 11px; color: #475569; display: block; }
.score-tier { font-size: 9px;  font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; display: block; margin-top: 3px; }

.rb-list { display: flex; flex-direction: column; gap: 10px; }
.rb-row  { display: flex; align-items: center; gap: 8px; }
.rb-meta { display: flex; flex-direction: column; width: 145px; flex-shrink: 0; }
.rb-label  { font-size: 12px; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rb-weight { font-size: 10px; color: #334155; margin-top: 1px; }
.rb-track {
  flex: 1;
  height: 5px;
  background: rgba(255,255,255,0.05);
  border-radius: 3px;
  overflow: hidden;
}
.rb-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.rb-right  { display: flex; align-items: center; gap: 8px; width: 52px; justify-content: flex-end; }
.rb-grade  { font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.rb-val    { font-size: 11px; color: #475569; font-family: 'JetBrains Mono', monospace; }

/* ── JSON Visualizer ───────────────────────────────────────────────── */
.jv-root {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.jv-section {
  background: rgba(10,10,20,0.5);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 14px 16px;
}
.jv-header {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 600;
  color: #475569;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.jv-icon { font-size: 14px; }
.jv-count {
  margin-left: auto;
  background: rgba(255,255,255,0.05);
  color: #64748b;
  border-radius: 4px;
  padding: 1px 7px;
  font-size: 11px;
}
.jv-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}
.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 5px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.3px;
  border: 1px solid transparent;
}
.pill-follow   { background: rgba(16,185,129,0.10);  border-color: rgba(16,185,129,0.25);  color: #6ee7b7; }
.pill-override { background: rgba(244,63,94,0.10);   border-color: rgba(244,63,94,0.25);   color: #fda4af; }
.pill-a        { background: rgba(124,58,237,0.12);  border-color: rgba(124,58,237,0.3);   color: #c4b5fd; }
.pill-b        { background: rgba(245,158,11,0.10);  border-color: rgba(245,158,11,0.25);  color: #fcd34d; }
.empty-state   { color: #334155; font-size: 12px; font-style: italic; }

/* ── Conflict Cards ────────────────────────────────────────────────── */
.conflict-card {
  background: rgba(0,0,0,0.25);
  border: 1px solid rgba(255,255,255,0.06);
  border-left: 3px solid #7c3aed;
  border-radius: 0 8px 8px 0;
  padding: 13px 16px;
  margin-bottom: 12px;
}
.conflict-card:last-child { margin-bottom: 0; }
.conflict-ids {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.vs-sep { color: #334155; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; }
.conflict-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  margin-bottom: 6px;
}
.meta-key { color: #475569; font-size: 11px; min-width: 80px; }
.meta-val { color: #94a3b8; }
.conflict-reasoning {
  font-style: italic;
  color: #475569;
  font-size: 11px;
  line-height: 1.6;
  padding: 8px 12px;
  border-left: 2px solid rgba(255,255,255,0.06);
  margin: 8px 0;
}

/* ── Parse Error ───────────────────────────────────────────────────── */
.parse-error {
  background: rgba(239,68,68,0.06);
  border: 1px solid rgba(239,68,68,0.2);
  border-radius: 8px;
  padding: 16px;
}
.pe-title { color: #fca5a5; font-weight: 600; font-size: 13px; margin-bottom: 10px; }
.pe-raw {
  white-space: pre-wrap;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #64748b;
  max-height: 200px;
  overflow-y: auto;
}

/* ── Tabs override ─────────────────────────────────────────────────── */
.tab-nav button {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: #64748b !important;
  border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
  color: #a78bfa !important;
  border-bottom-color: #7c3aed !important;
}

/* ── Methodology Card ──────────────────────────────────────────────── */
.methodology-card {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.mc-item {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 10px;
  padding: 14px;
}
.mc-item-num {
  font-size: 20px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  color: #7c3aed;
  margin-bottom: 6px;
}
.mc-item-title { font-size: 12px; font-weight: 600; color: #cbd5e1; margin-bottom: 4px; }
.mc-item-body  { font-size: 11px; color: #475569; line-height: 1.5; }
.weight-tag {
  display: inline-block;
  background: rgba(124,58,237,0.12);
  color: #a78bfa;
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 10px;
  font-weight: 600;
  margin-top: 6px;
}

/* ── Divider ───────────────────────────────────────────────────────── */
.divider {
  height: 1px;
  background: rgba(255,255,255,0.05);
  margin: 20px 0;
}

/* ── Footer Links ──────────────────────────────────────────────────── */
.cb-links {
  text-align: center;
  padding: 12px 0;
  display: flex;
  justify-content: center;
  gap: 20px;
}
.cb-link {
  font-size: 12px;
  color: #475569;
  text-decoration: none;
  transition: color 0.2s;
}
.cb-link:hover { color: #a78bfa; }

/* ── Gradio component label overrides ─────────────────────────────── */
label.svelte-1b6s6s, .label-wrap { color: #64748b !important; font-size: 12px !important; }
.block.svelte-90oupt { background: transparent !important; border: none !important; }
"""


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

HIER_HTML = """
<div class="hier-banner">
  <div class="hier-item hier-1">⚖️ Legal / Regulatory</div>
  <span class="hier-arrow">›</span>
  <div class="hier-item hier-2">👔 C-Suite</div>
  <span class="hier-arrow">›</span>
  <div class="hier-item hier-3">📋 VP</div>
  <span class="hier-arrow">›</span>
  <div class="hier-item hier-4">🗂 Director</div>
  <span class="hier-arrow">›</span>
  <div class="hier-item hier-5">👷 Manager</div>
  <span class="hier-arrow">›</span>
  <div class="hier-item hier-6">🔧 Team Lead</div>
</div>
"""

HEADER_HTML = """
<div class="cb-hero">
  <div class="cb-wordmark">
    <div class="cb-wordmark-icon">⚔️</div>
    <h1 class="cb-title">ConflictBench</h1>
  </div>
  <p class="cb-subtitle">Instruction Priority Resolution · RL Training Environment · OpenEnv Hackathon India 2026</p>
  <div class="cb-badges">
    <span class="cb-badge cb-badge-purple">GRPO Training</span>
    <span class="cb-badge cb-badge-cyan">Qwen2.5-3B</span>
    <span class="cb-badge cb-badge-amber">Deterministic Scoring</span>
    <span class="cb-badge cb-badge-purple">Scale AI Bonus</span>
  </div>
</div>
"""

METHODOLOGY_HTML = """
<div class="methodology-card">
  <div class="mc-item">
    <div class="mc-item-num">01</div>
    <div class="mc-item-title">Format Compliance</div>
    <div class="mc-item-body">Validates JSON structure with all required keys and sub-fields.</div>
    <span class="weight-tag">×10%</span>
  </div>
  <div class="mc-item">
    <div class="mc-item-num">02</div>
    <div class="mc-item-title">Conflict Identification</div>
    <div class="mc-item-body">F1 score on detected conflict pairs plus resolution accuracy bonus.</div>
    <span class="weight-tag">×20%</span>
  </div>
  <div class="mc-item">
    <div class="mc-item-num">03</div>
    <div class="mc-item-title">No Contradictions</div>
    <div class="mc-item-body">Penalises co-execution of mutually exclusive action keys in the plan.</div>
    <span class="weight-tag">×25%</span>
  </div>
  <div class="mc-item">
    <div class="mc-item-num">04</div>
    <div class="mc-item-title">Correct Final State</div>
    <div class="mc-item-body">F1 of execution plan vs ground-truth — the primary signal for GRPO.</div>
    <span class="weight-tag">×35%</span>
  </div>
  <div class="mc-item">
    <div class="mc-item-num">05</div>
    <div class="mc-item-title">Plan Efficiency</div>
    <div class="mc-item-body">Penalises bloated plans; rewards precision over inclusion-by-default.</div>
    <span class="weight-tag">×10%</span>
  </div>
</div>
"""


with gr.Blocks(
    title="ConflictBench — Instruction Priority Resolver",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    ),
    css=CUSTOM_CSS,
) as demo:

    scenario_state = gr.State(None)

    # ── Hero ─────────────────────────────────────────────────────────────
    gr.HTML(HEADER_HTML)
    gr.HTML(HIER_HTML)

    # ── Main layout ──────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── LEFT SIDEBAR ─────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):

            gr.HTML('<div class="panel-title"><div class="panel-title-dot"></div>SCENARIO CONFIG</div>')

            difficulty_slider = gr.Slider(
                minimum=1, maximum=3, step=1, value=2,
                label="Difficulty",
                info="1 = Easy (2 conflicts) · 2 = Medium (4) · 3 = Hard (6)",
            )

            gen_btn = gr.Button(
                "⚡  Generate Scenario",
                variant="primary",
                size="lg",
                elem_classes="btn-generate",
            )

            scenario_meta = gr.HTML(
                "<div class='pending-state' style='padding:16px;'>Configure difficulty and generate a scenario to begin.</div>"
            )

            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="panel-title"><div class="panel-title-dot"></div>DOCUMENT PREVIEW</div>')

            scenario_display = gr.Textbox(
                label="",
                lines=22,
                interactive=False,
                placeholder="Business instruction document will appear here…",
                elem_classes="gradio-textbox",
            )

            gr.HTML("""
            <div class="cb-links" style="margin-top:8px;">
              <a class="cb-link" href="https://huggingface.co/spaces/Harsh-9209/Conflict_Bench" target="_blank">🤗 HF Space</a>
              <a class="cb-link" href="https://github.com/Harsh-4210/Conflict_Bench" target="_blank">⎈ GitHub</a>
            </div>
            """)

        # ── RIGHT MAIN PANEL ──────────────────────────────────────────────
        with gr.Column(scale=3):

            with gr.Tabs(elem_classes="tab-nav"):

                # ── TAB 1 : Arena ─────────────────────────────────────────
                with gr.TabItem("⚔️  Model Arena"):

                    with gr.Row(equal_height=False):

                        # BASE MODEL
                        with gr.Column():
                            gr.HTML("""
                            <div class="model-header-base">
                              <span class="mh-tag mh-tag-base">Base Model</span>
                              <span class="mh-name">Qwen2.5-3B-Instruct</span>
                              <span class="mh-sub">No conflict resolution training</span>
                            </div>""")

                            base_btn = gr.Button(
                                "▶  Run Inference",
                                interactive=False,
                                elem_classes="btn-base",
                            )

                            gr.HTML('<div style="margin:14px 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#334155;font-weight:600;">Score Breakdown</div>')
                            base_score = gr.HTML(
                                "<div class='score-wrap pending-state'>Run inference to see scores</div>"
                            )

                            gr.HTML('<div style="margin:14px 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#334155;font-weight:600;">Output Analysis</div>')
                            base_output = gr.HTML(
                                "<div class='jv-root pending-state'>Output will appear here after inference</div>"
                            )

                        # FINE-TUNED MODEL
                        with gr.Column():
                            gr.HTML("""
                            <div class="model-header-trained">
                              <span class="mh-tag mh-tag-trained">Fine-Tuned Policy</span>
                              <span class="mh-name">ConflictBench Checkpoint</span>
                              <span class="mh-sub">GRPO-trained on authority hierarchy</span>
                            </div>""")

                            trained_btn = gr.Button(
                                "▶  Run Inference",
                                interactive=False,
                                variant="primary",
                                elem_classes="btn-trained",
                            )

                            gr.HTML('<div style="margin:14px 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#334155;font-weight:600;">Score Breakdown</div>')
                            trained_score = gr.HTML(
                                "<div class='score-wrap pending-state'>Run inference to see scores</div>"
                            )

                            gr.HTML('<div style="margin:14px 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#334155;font-weight:600;">Output Analysis</div>')
                            trained_output = gr.HTML(
                                "<div class='jv-root pending-state'>Output will appear here after inference</div>"
                            )

                # ── TAB 2 : Ground Truth ──────────────────────────────────
                with gr.TabItem("🔍  Ground Truth"):
                    gr.HTML("""
                    <div style="margin-bottom:16px;color:#64748b;font-size:13px;line-height:1.7;">
                      Reveal the hidden ground-truth resolution for the active scenario.
                      This is what a perfect agent should produce — compare against model outputs above.
                    </div>""")
                    gt_btn    = gr.Button("🔓  Reveal Ground Truth", variant="secondary")
                    gt_output = gr.HTML(
                        "<div class='jv-root pending-state'>Generate a scenario and click Reveal to see the correct answer.</div>"
                    )

                # ── TAB 3 : Methodology ───────────────────────────────────
                with gr.TabItem("📐  Methodology"):
                    gr.HTML(f"""
                    <div style="margin-bottom:20px;">
                      <h3 style="color:#c4b5fd;font-size:15px;font-weight:600;margin-bottom:8px;">Deterministic Scoring — No LLM Judge</h3>
                      <p style="color:#64748b;font-size:13px;line-height:1.7;">
                        ConflictBench uses five independent, rule-based rubric functions scored against
                        programmatically-generated ground truth. The composite reward gives GRPO a rich,
                        non-gameable gradient signal. An agent cannot win by blindly following or blindly
                        overriding every instruction.
                      </p>
                    </div>
                    {METHODOLOGY_HTML}
                    <div class="divider"></div>
                    <h3 style="color:#c4b5fd;font-size:15px;font-weight:600;margin-bottom:12px;">Why This Is Hard for LLMs</h3>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;color:#64748b;line-height:1.6;">
                      <div>• Conflicts are <em>implicit</em> — no instruction labels itself as conflicting</div>
                      <div>• Authority hierarchy is <em>never stated</em> in the prompt; it must be inferred</div>
                      <div>• Wrong early decisions <em>cascade</em> through the execution plan</div>
                      <div>• Requires structured JSON output with <em>correct instruction IDs</em></div>
                    </div>
                    """)

                # ── TAB 4 : Training Results ──────────────────────────────
                with gr.TabItem("📈  Training Results"):
                    gr.HTML("""
                    <div style="margin-bottom:20px;">
                      <h3 style="color:#c4b5fd;font-size:15px;font-weight:600;margin-bottom:8px;">GRPO Training on Qwen2.5-3B-Instruct</h3>
                      <p style="color:#64748b;font-size:13px;line-height:1.7;">
                        Demo run: 240 steps · 2 epochs · 120 scenarios · Kaggle T4 (4-bit QLoRA) · 5h 22m total.
                        Best checkpoint: step 200 (eval_reward = 0.5003).
                      </p>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px;">
                      <div class="mc-item" style="text-align:center;">
                        <div style="font-size:26px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#10b981;">0.500</div>
                        <div style="font-size:12px;color:#94a3b8;margin-top:6px;">Best Eval Reward</div>
                        <div style="font-size:11px;color:#475569;margin-top:3px;">checkpoint-200</div>
                      </div>
                      <div class="mc-item" style="text-align:center;">
                        <div style="font-size:26px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#06b6d4;">53%↓</div>
                        <div style="font-size:12px;color:#94a3b8;margin-top:6px;">Reward Std Reduction</div>
                        <div style="font-size:11px;color:#475569;margin-top:3px;">0.239 → 0.111</div>
                      </div>
                      <div class="mc-item" style="text-align:center;">
                        <div style="font-size:26px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#f59e0b;">87%↓</div>
                        <div style="font-size:12px;color:#94a3b8;margin-top:6px;">Eval Loss Drop</div>
                        <div style="font-size:11px;color:#475569;margin-top:3px;">0.033 → 0.004</div>
                      </div>
                    </div>
                    <div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.15);border-radius:8px;padding:14px;font-size:12px;color:#6ee7b7;line-height:1.6;">
                      ✅ Demo run complete. Model showed consistent improvement with reward plateau at ~0.50.
                      Overfitting observed after step 200 (eval dropped to 0.461 at step 240), confirming
                      checkpoint-200 as optimal. Production run with 600+ scenarios on A100 is planned to
                      break through the 0.50 ceiling.
                    </div>
                    """)

    # ── Event Wiring ──────────────────────────────────────────────────────
    gen_btn.click(
        fn=generate_scenario,
        inputs=[difficulty_slider],
        outputs=[
            scenario_state, scenario_display, scenario_meta,
            base_btn, trained_btn,
            base_score, trained_score,
            base_output, trained_output,
        ],
    )

    base_btn.click(
        fn=run_base_model,
        inputs=[scenario_state],
        outputs=[base_output, base_score],
    )

    trained_btn.click(
        fn=run_trained_model,
        inputs=[scenario_state],
        outputs=[trained_output, trained_score],
    )

    gt_btn.click(
        fn=format_ground_truth_html,
        inputs=[scenario_state],
        outputs=[gt_output],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

try:
    load_models()
except Exception as e:
    print(f"Model loading deferred (will retry at inference time): {e}")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)