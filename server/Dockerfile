# Blackwell (RTX 5070 Ti / sm_120) support: CUDA 12.8 + Python 3.11.
# The PyTorch runtime image avoids uv downloading a standalone Python during
# docker build and ships torch/torchvision wheels with cu128 support.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.cache/openpi/big_vision && \
    curl -fsSL https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
        -o /root/.cache/openpi/big_vision/paligemma_tokenizer.model

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY packages/policy-client ./packages/policy-client
COPY src ./src
COPY server ./server
COPY runtime_core ./runtime_core

# Create virtual environment and install locked deps including local workspace packages.
RUN uv venv --python /opt/conda/bin/python /workspace/.venv
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-project
RUN uv pip install --python /workspace/.venv/bin/python pytest

# Replace the PyPI torch wheel (cu126, no sm_120 kernels) with the cu128 build
# from the official PyTorch index. Required for Blackwell (RTX 5070 Ti);
# without it, any CUDA op raises "no kernel image is available for execution
# on the device". --no-deps keeps the rest of uv sync's resolution intact.
RUN uv pip install --python /workspace/.venv/bin/python \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/policy-client/src"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
