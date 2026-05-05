#!/usr/bin/env bash
set -euo pipefail

# POLICY_CHECKPOINT_DIR is set by docker-compose.yml (defaults to /policy_checkpoint).
# It is required by the openpi-backed adapter (`my_policy.adapter:MyPolicyAdapter`).
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"

ARGS=(
  "--host" "${HOST}"
  "--port" "${PORT}"
)

if [[ -n "${POLICY_CHECKPOINT_DIR:-}" ]]; then
  ARGS+=("--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}")
fi

# Set POLICY_MODULE to swap in your own policy class without editing
# serve_hsr_policy_ws.py, e.g. to bypass the model with the placeholder:
#   POLICY_MODULE="serve_hsr_policy_ws:ZeroPolicy"
if [[ -n "${POLICY_MODULE:-}" ]]; then
  ARGS+=("--policy-module" "${POLICY_MODULE}")
fi

# Optional torch device hint (informational for the JAX checkpoint, but the
# adapter accepts it for parity with the upstream openpi server CLI).
if [[ -n "${POLICY_PYTORCH_DEVICE:-}" ]]; then
  ARGS+=("--pytorch-device" "${POLICY_PYTORCH_DEVICE}")
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
