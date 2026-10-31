# ConflictBench training Space. Reproducible build: nothing here follows a moving branch.
#
#  * torch / torchvision / xformers: exact released versions (same as requirements-train.txt header).
#  * unsloth / unsloth_zoo / trl / transformers / ...: exact released versions, read from the repo's
#    requirements-train.txt (installed from PyPI, never from git HEAD).
#  * The ConflictBench code is cloned at CONFLICTBENCH_REF, a git TAG or COMMIT SHA - never a branch name.
#
# CONFLICTBENCH_REF has no usable default on purpose: the build fails with a clear message until you set it.
# In the Space: Settings -> Variables and secrets -> New variable  CONFLICTBENCH_REF = <tag or commit SHA>
# (Docker Spaces expose variables as build arguments). Locally:
#   docker build --build-arg CONFLICTBENCH_REF=<tag-or-sha> -t conflictbench-train training_space
#
# Cutting a tag for a run:  git tag train-run-1 && git push origin train-run-1   then set CONFLICTBENCH_REF=train-run-1
FROM python:3.11-slim

ARG CONFLICTBENCH_REF=REPLACE_WITH_TAG_OR_COMMIT
ARG CONFLICTBENCH_REPO=https://github.com/Harsh-4210/Conflict_Bench.git

# Fail fast (before the slow installs) when no real ref was given.
RUN case "${CONFLICTBENCH_REF}" in \
      REPLACE_WITH_TAG_OR_COMMIT|""|main|master|HEAD) \
        echo "ERROR: set the CONFLICTBENCH_REF build argument to a git tag or commit SHA (got '${CONFLICTBENCH_REF}'); branch names are not reproducible." >&2; exit 1 ;; \
    esac

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# 1. Torch stack, pinned. Must come before unsloth.
RUN pip install --no-cache-dir \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    "xformers==0.0.34" \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 2. The code under training, at an exact tag/commit.
RUN git clone "${CONFLICTBENCH_REPO}" $HOME/conflictbench \
    && git -C $HOME/conflictbench checkout --detach "${CONFLICTBENCH_REF}" \
    && git -C $HOME/conflictbench rev-parse HEAD > $HOME/conflictbench_commit.txt
ENV CONFLICTBENCH_DIR=$HOME/conflictbench \
    CONFLICTBENCH_REF=${CONFLICTBENCH_REF}

# 3. The pinned training stack (unsloth, unsloth_zoo, trl, transformers, peft, ... released versions, one file).
RUN pip install --no-cache-dir -r $HOME/conflictbench/requirements-train.txt

# 4. Space UI (gradio). Installed last, but requirements.txt contains only gradio, so the pins above stay intact.
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir -r $HOME/app/requirements.txt

COPY --chown=user . $HOME/app

EXPOSE 7860
CMD ["python", "app.py"]
