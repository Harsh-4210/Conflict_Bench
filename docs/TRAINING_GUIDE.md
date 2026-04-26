# ConflictBench — Training Guide

## Choosing a Platform

| Platform | GPU | VRAM | Est. Time | Cost | Recommended For |
|---|---|---|---|---|---|
| HF Spaces (L40S) | L40S | 48GB | ~8h | ~$14 | Production runs |
| HF Spaces (L4) | L4 | 24GB | ~14h | ~$11 | Production runs |
| Google Colab Pro | L4 | 24GB | ~14h | ~$10 | Experimentation |
| Google Colab Pro+ | A100 | 40GB | ~6h | ~$20 | Fast iteration |
| Kaggle (free) | T4 | 16GB | ~28h (2 sessions) | Free | Budget runs |
| Local (RTX 3090) | — | 24GB | ~12h | Electricity | Research |

---

## Training Scripts

This repository has two training entry points:

**`hf_space_l40s/train.py`** — for HF Spaces and Colab. Integrates with the Gradio dashboard, streams live logs, auto-uploads to HF Hub. Pre-configured for L40S but auto-detects GPU.

**`train_grpo.py`** — for local machines, Kaggle, and research use. Exposes all hyperparameters directly. More verbose logging. Supports checkpoint resume.

Both scripts use the same generator, verifier, and GRPO configuration logic.

---

## Key Configuration Parameters

```python
TRAIN_SCENARIOS = 400       # number of unique training scenarios per epoch
EVAL_SCENARIOS  = 60        # held-out evaluation scenarios
NUM_EPOCHS      = 2         # full passes over the training set
LEARNING_RATE   = 3e-6      # conservative; prevents catastrophic forgetting
BETA            = 0.04      # KL penalty; 0.02 causes excessive drift (use 0.04+)
num_generations = 4         # GRPO group size; 6 recommended on L40S
MAX_NEW_TOKENS  = 768       # completion budget; average ~300 tokens in practice
MAX_PROMPT_LENGTH = 3200    # prompt budget; generator enforces 4000 char limit
SAVE_STEPS      = 50        # checkpoint frequency
EVAL_STEPS      = 50        # evaluation frequency
```

---

## Checkpoint Strategy

The best checkpoint is typically **not** the final one. Reward peaks mid-training (around step 250 in Run 2) and may decline slightly as KL divergence increases. Always evaluate multiple checkpoints.

```bash
# List all checkpoints
ls conflictbench-grpo-output/checkpoint-*/

# The trainer saves with load_best_model_at_end=True,
# but verify by comparing checkpoint-250 vs final manually.
```

---

## Resuming After Session Interruption (Kaggle / Colab)

The training script supports automatic resume:

```python
# In train_grpo.py, GRPOConfig already has:
save_steps=50           # checkpoints every 50 steps
save_total_limit=10     # keeps last 10 checkpoints
```

To resume, simply re-run the same script. If `OUTPUT_DIR` contains a `checkpoint-*` directory, TRL will automatically resume from the latest checkpoint.

For Colab: redirect `OUTPUT_DIR` to Google Drive to persist checkpoints across session resets:
```python
OUTPUT_DIR = "/content/drive/MyDrive/conflictbench-grpo-output"
```

---

## Uploading to HF Hub

Set `HF_TOKEN` and `HF_REPO_ID` before training:

```bash
export HF_TOKEN=hf_your_write_token
export HF_REPO_ID=your-username/your-model-name
```

The training script automatically uploads the best checkpoint after training completes. To upload manually:

```python
from huggingface_hub import HfApi
api = HfApi(token="hf_your_token")
api.upload_folder(
    folder_path="./conflictbench-grpo-output/checkpoint-250",
    repo_id="your-username/conflictbench-model",
    repo_type="model"
)
```
