"""
ConflictBench — HuggingFace Spaces Demo (Gradio)
Premium UI Redesign — Dark Glassmorphism + Violet/Cyan Palette
Compares the base model with the LoRA-fine-tuned model on the same scenario.
Live scoring with a reward breakdown.

Both models are the SAME network: the LoRA adapter is loaded once; the "base"
run executes inside ``model.disable_adapter()`` and the "trained" run uses the
adapter. Models are loaded lazily on the first inference call, so importing
this module needs neither torch nor a download.

Deploy: push this file + requirements to a HF Space (SDK: gradio).
"""

import html
import json
import logging
import os
import sys
import threading
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from generator import ScenarioGenerator
from prompts import build_messages
from verifier import score as compute_score, parse_agent_output
import verifier as _verifier

logger = logging.getLogger("conflictbench.app")

# ---------------------------------------------------------------------------
# Model loading (lazy)
# ---------------------------------------------------------------------------

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
DEFAULT_ADAPTER_ID = "Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora"
# CONFLICTBENCH_ADAPTER (or the legacy TRAINED_MODEL_ID) overrides the adapter;
# set it to an empty string to run the base model only.
ADAPTER_ID = os.getenv("CONFLICTBENCH_ADAPTER", os.getenv("TRAINED_MODEL_ID", DEFAULT_ADAPTER_ID))

MAX_PROMPT_TOKENS = 3072   # longer prompts are rejected, never truncated
MAX_NEW_TOKENS = 512
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9

_generator = ScenarioGenerator(seed=None)
_load_lock = threading.Lock()
_state = {
    "loaded": False,
    "model": None,
    "tokenizer": None,
    "adapter_loaded": False,
    "adapter_error": None,
    "selfcheck": None,      # True = outputs differ with/without adapter, False = identical
}


class PromptTooLongError(ValueError):
    """The chat-templated prompt exceeds MAX_PROMPT_TOKENS."""


def _model_device(model):
    try:
        return model.device
    except Exception:
        import torch
        return next(model.parameters()).device


def _to_input_ids(encoded):
    """apply_chat_template returns a tensor (older transformers) or a BatchEncoding (newer)."""
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        return encoded["input_ids"]
    return encoded


def generate_text(model, tokenizer, scenario, sample: bool = False,
                  max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Run one completion for ``scenario`` using the training chat format.

    Greedy by default. Raises PromptTooLongError instead of truncating.
    The caller is responsible for adapter on/off (see run_model).
    """
    import torch

    encoded = tokenizer.apply_chat_template(
        build_messages(scenario),
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = _to_input_ids(encoded)
    n_prompt = int(input_ids.shape[-1])
    if n_prompt > MAX_PROMPT_TOKENS:
        raise PromptTooLongError(
            f"Prompt is {n_prompt} tokens; the limit is {MAX_PROMPT_TOKENS}. "
            "It was NOT truncated because that would cut off the output-format instructions."
        )

    input_ids = input_ids.to(_model_device(model))
    attention_mask = torch.ones_like(input_ids)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    if sample:
        gen_kwargs.update(do_sample=True, temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    return tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip()


def _selfcheck(model, tokenizer) -> None:
    """Same prompt, greedy, adapter on vs off. Identical output means the adapter has no effect."""
    sc = ScenarioGenerator(seed=0).generate(difficulty=1)
    with_adapter = generate_text(model, tokenizer, sc, sample=False, max_new_tokens=96)
    with model.disable_adapter():
        without_adapter = generate_text(model, tokenizer, sc, sample=False, max_new_tokens=96)
    _state["selfcheck"] = with_adapter != without_adapter
    if not _state["selfcheck"]:
        logger.warning(
            "Self-check FAILED: outputs with and without the LoRA adapter are identical. "
            "The base-vs-trained comparison would be meaningless (adapter missing, zero-initialised, "
            "or not applied)."
        )
    else:
        logger.info("Self-check passed: adapter changes the model output.")


def load_models() -> dict:
    """Load the base model once and attach the LoRA adapter (idempotent, thread-safe)."""
    with _load_lock:
        if _state["loaded"]:
            return _state
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        logger.info("Loading base model %s ...", BASE_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if cuda else torch.float32,
            device_map="auto" if cuda else None,
        )

        if ADAPTER_ID:
            try:
                from peft import PeftModel
                logger.info("Loading LoRA adapter %s ...", ADAPTER_ID)
                model = PeftModel.from_pretrained(model, ADAPTER_ID)
                _state["adapter_loaded"] = True
            except Exception as e:  # missing peft, bad repo, no network ...
                _state["adapter_error"] = f"{type(e).__name__}: {e}"
                logger.warning("Could not load LoRA adapter %s: %s", ADAPTER_ID, e)
        else:
            _state["adapter_error"] = "No adapter configured (CONFLICTBENCH_ADAPTER is empty)."
        model.eval()

        _state.update(model=model, tokenizer=tokenizer)
        if _state["adapter_loaded"]:
            try:
                _selfcheck(model, tokenizer)
            except Exception as e:
                logger.warning("Adapter self-check could not run: %s", e)
        _state["loaded"] = True
        return _state


def run_model(which: str, scenario, sample: bool = False) -> str:
    """Generate a completion from the 'base' or 'trained' variant of the shared model."""
    st = load_models()
    model, tokenizer = st["model"], st["tokenizer"]
    if which == "trained":
        if not st["adapter_loaded"]:
            raise RuntimeError(f"Trained adapter not available: {st['adapter_error']}")
        return generate_text(model, tokenizer, scenario, sample=sample)
    if st["adapter_loaded"]:
        with model.disable_adapter():
            return generate_text(model, tokenizer, scenario, sample=sample)
    return generate_text(model, tokenizer, scenario, sample=sample)


# ---------------------------------------------------------------------------
# HTML Renderers
# ---------------------------------------------------------------------------

def _esc(value) -> str:
    """HTML-escape any model-supplied value (coerced to str first)."""
    return html.escape(str(value), quote=True)


PENDING_SCORE_HTML = "<div class='score-wrap pending-state'>Run inference to see scores</div>"
PENDING_OUTPUT_HTML = "<div class='jv-root pending-state'>Output will appear here</div>"


def _error_html(title: str, detail: str = "") -> str:
    return f"""
        <div class="parse-error">
          <div class="pe-title">⚠ {_esc(title)}</div>
          <pre class="pe-raw">{_esc(detail)[:1200]}</pre>
        </div>"""


def _rubric_bar(label: str, value: float, weight_pct) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    pct   = max(0, min(100, int(value * 100)))
    grade = "A" if pct >= 80 else ("B" if pct >= 60 else ("C" if pct >= 40 else "D"))
    color = (
        "#10b981" if pct >= 80 else
        "#06b6d4" if pct >= 60 else
        "#f59e0b" if pct >= 40 else
        "#f43f5e"
    )
    weight = f'<span class="rb-weight">×{weight_pct}%</span>' if weight_pct is not None else ""
    return f"""
    <div class="rb-row">
      <div class="rb-meta">
        <span class="rb-label">{label}</span>
        {weight}
      </div>
      <div class="rb-track">
        <div class="rb-fill" style="width:{pct}%;background:{color};"></div>
      </div>
      <div class="rb-right">
        <span class="rb-grade" style="color:{color};">{grade}</span>
        <span class="rb-val">{value:.2f}</span>
      </div>
    </div>"""


def _weight_pct(key: str):
    """Weight of a rubric in the composite, read from the verifier (None if unavailable)."""
    weights = getattr(_verifier, "RUBRIC_WEIGHTS", None)
    try:
        return int(round(float(weights[key]) * 100))
    except Exception:
        return None


def format_score_html(bd: dict) -> str:
    composite = float(bd.get("composite", 0.0) or 0.0)
    composite = max(0.0, min(1.0, composite))
    cpct      = int(composite * 100)
    c_color   = "#10b981" if cpct >= 80 else ("#06b6d4" if cpct >= 60 else ("#f59e0b" if cpct >= 40 else "#f43f5e"))
    tier      = "EXCELLENT" if cpct >= 80 else ("GOOD" if cpct >= 60 else ("FAIR" if cpct >= 40 else "POOR"))

    rows = (
        ("Correct Final State",    "correct_final_state"),
        ("No Contradictions",      "no_contradictions"),
        ("Conflict Identification", "conflict_identification"),
        ("Plan Efficiency",        "efficiency"),
        ("Format Compliance",      "format_compliance"),
    )
    bars = "".join(_rubric_bar(label, bd.get(key, 0.0), _weight_pct(key)) for label, key in rows)

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


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _conflict_card(c) -> str:
    if not isinstance(c, dict):
        return f"""
        <div class="conflict-card">
          <div class="conflict-row"><span class="meta-key">Malformed entry</span>
          <span class="meta-val">{_esc(str(c)[:200])}</span></div>
        </div>"""
    res       = _esc(str(c.get("resolution", "UNKNOWN")))
    ctype     = _esc(str(c.get("conflict_type", "N/A")))
    reasoning = _esc(str(c.get("reasoning", "No reasoning provided."))[:200])
    ins_a     = _esc(str(c.get("instruction_a", "?")))
    ins_b     = _esc(str(c.get("instruction_b", "?")))
    res_class = "pill-follow" if res != "UNKNOWN" else "pill-override"
    return f"""
        <div class="conflict-card">
          <div class="conflict-ids">
            <span class="pill pill-a">{ins_a}</span>
            <span class="vs-sep">⚡ vs</span>
            <span class="pill pill-b">{ins_b}</span>
          </div>
          <div class="conflict-row"><span class="meta-key">Type</span><span class="meta-val">{ctype}</span></div>
          <div class="conflict-reasoning">"{reasoning}"</div>
          <div class="conflict-row" style="margin-top:10px;">
            <span class="meta-key">Resolved by</span>
            <span class="pill {res_class}" style="font-size:11px;">{res}</span>
          </div>
        </div>"""


def format_json_html(raw) -> str:
    try:
        parsed = parse_agent_output(raw)
    except Exception as e:  # unexpected JSON shape the parser does not handle
        return _error_html("Parse Failed", f"{type(e).__name__}: {e}\n\n{raw}")
    if parsed is None or not parsed.parsed_ok:
        return _error_html("Parse Failed", str(raw))

    plan      = _as_list(parsed.execution_plan)
    overridden = _as_list(parsed.overridden_instructions)
    conflicts = _as_list(parsed.identified_conflicts)

    plan_pills = "".join(
        f'<span class="pill pill-follow">{_esc(ins)}</span>' for ins in plan
    ) or '<span class="empty-state">None</span>'

    over_pills = "".join(
        f'<span class="pill pill-override">{_esc(ins)}</span>' for ins in overridden
    ) or '<span class="empty-state">None</span>'

    conflicts_html = "".join(_conflict_card(c) for c in conflicts)
    if not conflicts_html:
        conflicts_html = '<span class="empty-state">No conflicts identified.</span>'

    return f"""
    <div class="jv-root">
      <div class="jv-section">
        <div class="jv-header"><span class="jv-icon">✅</span>Execution Plan
          <span class="jv-count">{len(plan)}</span>
        </div>
        <div class="jv-pills">{plan_pills}</div>
      </div>
      <div class="jv-section">
        <div class="jv-header"><span class="jv-icon">🚫</span>Overridden
          <span class="jv-count">{len(overridden)}</span>
        </div>
        <div class="jv-pills">{over_pills}</div>
      </div>
      <div class="jv-section">
        <div class="jv-header"><span class="jv-icon">⚔️</span>Conflict Analysis
          <span class="jv-count">{len(conflicts)}</span>
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
      <div class="meta-chip"><span class="mc-label">Domain</span><span class="mc-val">{_esc(scenario.domain)}</span></div>
      <div class="meta-chip"><span class="mc-label">Difficulty</span>
        <span class="mc-val diff-{int(difficulty)}">{diff_map.get(difficulty,'?')}</span></div>
      <div class="meta-chip"><span class="mc-label">Instructions</span>
        <span class="mc-val">{len(scenario.instructions)}</span></div>
      <div class="meta-chip"><span class="mc-label">Conflicts</span>
        <span class="mc-val">{len(scenario.conflicts)}</span></div>
    </div>"""
    return (
        scenario,
        scenario.prompt,
        meta_html,
        gr.update(interactive=True),
        gr.update(interactive=True),
        PENDING_SCORE_HTML, PENDING_SCORE_HTML,
        PENDING_OUTPUT_HTML, PENDING_OUTPUT_HTML,
    )


def _run_and_render(which: str, scenario, sample: bool):
    """Shared body of the two run buttons. Returns (output_html, score_html)."""
    if scenario is None:
        gr.Warning("Generate a scenario first!")
        return "", ""
    gr.Info(f"Running {'fine-tuned' if which == 'trained' else 'base'} model inference…")
    try:
        response = run_model(which, scenario, sample=bool(sample))
    except PromptTooLongError as e:
        return _error_html("Prompt too long", str(e)), PENDING_SCORE_HTML
    except Exception as e:
        logger.exception("Inference failed")
        return _error_html("Inference failed", f"{type(e).__name__}: {e}"), PENDING_SCORE_HTML
    try:
        breakdown = compute_score(response, scenario).to_dict()
        score_html = format_score_html(breakdown)
    except Exception as e:
        score_html = _error_html("Scoring failed", f"{type(e).__name__}: {e}")
    return format_json_html(response), score_html


def run_base_model(scenario, sample=False):
    return _run_and_render("base", scenario, sample)


def run_trained_model(scenario, sample=False):
    return _run_and_render("trained", scenario, sample)


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

def _weight_tag(key: str) -> str:
    w = _weight_pct(key)
    return f'<span class="weight-tag">×{w}%</span>' if w is not None else ""


def build_methodology_html() -> str:
    """Rubric cards; weights are read from the verifier so they never go stale."""
    cards = (
        ("01", "Format Compliance", "format_compliance",
         "Validates JSON structure with all required keys and sub-fields."),
        ("02", "Conflict Identification", "conflict_identification",
         "F1 score on detected conflict pairs plus resolution accuracy bonus."),
        ("03", "No Contradictions", "no_contradictions",
         "Penalises co-execution of mutually exclusive action keys in the plan."),
        ("04", "Correct Final State", "correct_final_state",
         "F1 of execution plan vs ground-truth — the primary signal for GRPO."),
        ("05", "Plan Efficiency", "efficiency",
         "Penalises bloated plans; rewards precision over inclusion-by-default."),
    )
    items = "".join(f"""
  <div class="mc-item">
    <div class="mc-item-num">{num}</div>
    <div class="mc-item-title">{title}</div>
    <div class="mc-item-body">{body}</div>
    {_weight_tag(key)}
  </div>""" for num, title, key, body in cards)
    return f'<div class="methodology-card">{items}\n</div>'


METHODOLOGY_HTML = build_methodology_html()


# Gradio 6 moved theme/css from the Blocks constructor to launch().
_STYLE_KWARGS = dict(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    ),
    css=CUSTOM_CSS,
)
try:
    _GRADIO_MAJOR = int(gr.__version__.split(".")[0])
except Exception:
    _GRADIO_MAJOR = 5
_BLOCKS_STYLE = {} if _GRADIO_MAJOR >= 6 else _STYLE_KWARGS
_LAUNCH_STYLE = _STYLE_KWARGS if _GRADIO_MAJOR >= 6 else {}

with gr.Blocks(
    title="ConflictBench — Instruction Priority Resolver",
    **_BLOCKS_STYLE,
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

            sample_cb = gr.Checkbox(
                value=False,
                label="Sample (temperature 0.7)",
                info="Off = greedy decoding (deterministic, the default).",
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
                        ConflictBench scores answers with three rule-based rubrics (correct final
                        state, conflict identification, no contradictions) behind a format gate,
                        against ground truth computed by a deterministic resolver. An audit found
                        that the original reward could be gamed; the reward and the task were
                        redesigned and the cheat strategies are now regression tests (see the README).
                      </p>
                    </div>
                    {METHODOLOGY_HTML}
                    <div class="divider"></div>
                    <h3 style="color:#c4b5fd;font-size:15px;font-weight:600;margin-bottom:12px;">Why This Is Hard for LLMs</h3>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;color:#64748b;line-height:1.6;">
                      <div>• Conflicts are <em>implicit</em> — no instruction labels itself as conflicting</div>
                      <div>• By default the tier list is <em>not</em> shown: the model must rank people from their job titles and the notices in the document</div>
                      <div>• Every conflict must be resolved consistently in the final execution plan</div>
                      <div>• Requires structured JSON output with <em>correct instruction IDs</em></div>
                    </div>
                    """)

                # ── TAB 4 : Training Results ──────────────────────────────
                with gr.TabItem("📈  Training Results"):
                    gr.HTML("""
                    <div style="margin-bottom:20px;">
                      <h3 style="color:#c4b5fd;font-size:15px;font-weight:600;margin-bottom:8px;">GRPO Training on Qwen2.5-3B-Instruct</h3>
                      <p style="color:#64748b;font-size:13px;line-height:1.7;">
                        Results are being re-measured. Earlier figures were removed because they could not
                        be reproduced with the current verifier. Final numbers will be read from the
                        evaluation outputs in <code>results/</code> once the re-evaluation is complete.
                      </p>
                    </div>
                    <div style="background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.2);border-radius:8px;padding:14px;font-size:12px;color:#fcd34d;line-height:1.6;">
                      Not yet re-measured. Use the Model Arena tab to compare the base model and the
                      fine-tuned adapter on a scenario yourself (greedy decoding by default).
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
        inputs=[scenario_state, sample_cb],
        outputs=[base_output, base_score],
    )

    trained_btn.click(
        fn=run_trained_model,
        inputs=[scenario_state, sample_cb],
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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, **_LAUNCH_STYLE)