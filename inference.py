import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# ── Config ──────────────────────────────────────────────────────────────────
# Path to your trained LoRA adapter (if local) or HF Repo ID
# If you have it locally, use "./outputs/final"
MODEL_ID = "Harsh-9209/Conflict_Bench_Adapter" # Update with your backup repo or local path

SYSTEM_PROMPT = """You are an expert business operations coordinator.
Your task: given a set of business instructions from various stakeholders, identify ALL conflicts and produce a resolution plan.
Show your thinking in a <thought> block, then provide the resolution in a JSON block."""

# ── Load Model ──────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def resolve_conflicts(scenario_text: str):
    """
    Takes a business scenario and returns the model's reasoning and JSON resolution.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Scenario:\n{scenario_text}\n\nIdentify conflicts and resolve them."},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    print("\n🚀 Model is thinking...\n" + "="*50)
    
    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = 768,
        temperature = 0.7,
        top_p = 0.9,
        use_cache = True,
        streamer = TextStreamer(tokenizer, skip_prompt=True)
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ── Example Test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_scenario = """
    [INS-1] CEO: Hire 5 new developers by Friday to meet the launch deadline.
    [INS-2] CFO: No new hiring is permitted until the end of the quarter due to budget audits.
    [INS-3] HR Director: All new recruitment must undergo a mandatory 2-week background check period.
    """
    
    print("Testing Conflict Resolution...")
    resolve_conflicts(sample_scenario)
