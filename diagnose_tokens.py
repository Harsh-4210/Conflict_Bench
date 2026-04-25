"""
Run this BEFORE retraining to confirm your prompts fit in the token budget.
If you see 'clipped_ratio: 1.0' during training, run this first.

Usage: python diagnose_tokens.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from generator import ScenarioGenerator
from transformers import AutoTokenizer

TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512
MAX_PROMPT_LENGTH = 3200

print(f"Loading tokenizer: {TOKENIZER_ID}")
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

gen = ScenarioGenerator(seed=42)
print("\n=== Prompt length analysis across 30 scenarios ===\n")

results = {1: [], 2: [], 3: []}
for diff in [1, 2, 3]:
    for _ in range(10):
        sc = gen.generate(difficulty=diff)
        n_tokens = len(tok.encode(sc.prompt))
        results[diff].append(n_tokens)

for diff in [1, 2, 3]:
    lengths = results[diff]
    avg = sum(lengths) / len(lengths)
    mx = max(lengths)
    fits = sum(1 for l in lengths if l <= MAX_PROMPT_LENGTH)
    print(f"Difficulty {diff}: avg={avg:.0f} tokens | max={mx} tokens | fits in budget: {fits}/10")
    if avg + MAX_NEW_TOKENS > 4096:
        print(f"  ⚠️  WARNING: avg prompt + response ({avg:.0f}+{MAX_NEW_TOKENS}={avg+MAX_NEW_TOKENS:.0f}) exceeds MAX_SEQ_LENGTH=4096")
    else:
        print(f"  ✅ avg total: {avg+MAX_NEW_TOKENS:.0f} tokens — fits in 4096 window")

print(f"\n=== Recommendation ===")
d1_avg = sum(results[1]) / len(results[1])
if d1_avg + MAX_NEW_TOKENS < 4096:
    print(f"✅ FORCE_DIFFICULTY=1 is safe. Retrain with current config.")
else:
    print(f"⚠️  Even difficulty 1 is tight. Reduce MAX_NEW_TOKENS further or shorten prompts.")
