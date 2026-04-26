"""
Push the demo app (app.py) and dependencies to HF Space.
Uploads: app.py, generator.py, verifier.py, requirements.txt, README.md
"""

import os
import tempfile
import shutil
from pathlib import Path

from huggingface_hub import HfApi, login

SPACE_ID = "Harsh-9209/Conflict_Bench"
ROOT = Path(__file__).parent

# Login
token = os.getenv("HF_TOKEN")
if not token:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")

if token:
    login(token=token)
else:
    print("No HF_TOKEN found — will use cached credentials")

api = HfApi()

# Create a staging directory with only the files needed for the Space
staging = Path(tempfile.mkdtemp())

# Files to upload from root
files_to_copy = [
    "app.py",           # Demo UI (base vs fine-tuned comparison)
    "generator.py",     # Scenario generator (imported by app.py)
    "verifier.py",      # Deterministic scorer (imported by app.py)
]

for f in files_to_copy:
    src = ROOT / f
    if src.exists():
        shutil.copy2(src, staging / f)
        print(f"  Staged: {f}")
    else:
        print(f"  WARNING: {f} not found!")

# Create a Space-specific requirements.txt
(staging / "requirements.txt").write_text("""\
gradio>=4.0.0
torch>=2.1.0
transformers>=4.45.0
peft>=0.12.0
accelerate>=0.33.0
""", encoding="utf-8")
print("  Staged: requirements.txt (space-specific)")

# Create Space-specific README.md (HF card metadata)
(staging / "README.md").write_text("""\
---
title: ConflictBench
emoji: "\u2694\uFE0F"
colorFrom: indigo
colorTo: purple
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: true
license: mit
---

# ConflictBench -- Instruction Priority Resolution via GRPO

Interactive demo: generate business instruction scenarios with embedded conflicts,
then compare base model vs GRPO-trained model on authority-aware conflict resolution.

- **Environment:** [GitHub](https://github.com/Harsh-4210/Conflict_Bench)
- **Trained Model:** [Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora](https://huggingface.co/Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora)
- **Colab:** [Open in Colab](https://colab.research.google.com/drive/18UJSpREGN152swrVjkEbGa0aWJR7eROH?usp=sharing)
- **Blog:** [Read the writeup](https://github.com/Harsh-4210/Conflict_Bench/blob/main/blog.md)
""", encoding="utf-8")
print("  Staged: README.md (space card)")

# Upload
print(f"\nUploading to HF Space: {SPACE_ID} ...")
api.upload_folder(
    folder_path=str(staging),
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="Deploy demo app (base vs fine-tuned comparison UI)",
)

print(f"\nDone! Space updated: https://huggingface.co/spaces/{SPACE_ID}")

# Cleanup
shutil.rmtree(staging)
