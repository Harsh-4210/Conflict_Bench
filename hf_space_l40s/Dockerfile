FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# 1. Install the entire "Gold Standard" ML stack in one go.
# This includes all vision/audio/quantization/acceleration tools to prevent ANY further 'ModuleNotFound' errors.
RUN pip install --no-cache-dir \
    "torch==2.5.0" \
    "torchvision==0.20.0" \
    "torchaudio==2.5.0" \
    "triton==3.1.0" \
    "xformers==0.0.28.post2" \
    "accelerate" \
    "peft" \
    "bitsandbytes" \
    "sentencepiece" \
    "protobuf" \
    "transformers>=4.46.0" \
    "trl>=0.12.0" \
    "datasets" \
    "huggingface_hub" \
    "matplotlib" \
    "scipy" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 2. Install Unsloth components (must be after torch stack)
RUN pip install --no-cache-dir \
    "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git" \
    "unsloth @ git+https://github.com/unslothai/unsloth.git"

# 3. CRITICAL: Uninstall torchao completely to bypass the 'int1' bug
RUN pip uninstall -y torchao

# 4. Install Gradio and remaining UI needs
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir -r $HOME/app/requirements.txt

# Copy application files
COPY --chown=user . $HOME/app

# Expose port
EXPOSE 7860

# Start the application
CMD ["python", "app.py"]
