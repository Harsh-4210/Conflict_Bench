# Local Linux GPU runtime. Build from the repository root:
# docker build -f docker/training/Dockerfile -t conflictbench-training:local .
# Mount the current checkout at /workspace when running; code is not baked in.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 xformers==0.0.34 \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements-train.txt /opt/conflictbench/requirements-train.txt
RUN python -m pip install --no-cache-dir -r /opt/conflictbench/requirements-train.txt \
    && python -m pip check

WORKDIR /workspace
CMD ["python", "-m", "training.train_grpo", "--config", "configs/grpo_smoke.yaml"]
