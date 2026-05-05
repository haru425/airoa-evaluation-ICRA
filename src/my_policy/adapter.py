"""Adapter that bridges the AIRoA HSR WebSocket I/O contract to an openpi
JAX policy trained with `openpi_offline_rl` (e.g. `pi05_hsr` checkpoint at step
50000).

What this file does
-------------------
1. Loads the trained JAX/Orbax checkpoint via openpi's
   `policy_config.create_trained_policy`, which wires up the full HSR transform
   stack (image resize → tokenizer → prompt injection → state/action norm).
2. Renames AIRoA's observation keys (`head_rgb`, `hand_rgb`, `state`, `prompt`)
   into openpi's expected keys (`observation/image`,
   `observation/wrist_image`, `observation/state`, `prompt`) before delegating
   to the wrapped policy.
3. Slices the model's action chunk to 11 dims (HSROutputs already does this,
   but we re-assert/cast to (T, 11) float32 to satisfy the AIRoA contract).

Notes
-----
- The training run that produced this checkpoint set
  `discrete_state_input=True` (it overrode the `pi05_hsr` config default of
  False on the command line). We replicate that override here so the prompt
  tokenizer matches what the model was trained against.
- Action normalization (`action_norm_mode=mean_std`) is auto-detected from
  the checkpoint's `assets/<asset_id>/normalization_config.json` by
  `policy_config.apply_checkpoint_normalization_config`, so we don't have to
  hard-code it here.
- Norm stats are loaded from `<checkpoint_dir>/assets/<asset_id>/norm_stats.json`.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

# openpi (JAX) bits — these come from src/openpi (copied from openpi_offline_rl).
import openpi.policies.policy_config as policy_config
import openpi.training.config as train_config_lib

logger = logging.getLogger(__name__)


# Default openpi training-config name for this checkpoint family.
DEFAULT_CONFIG_NAME = "pi05_hsr"


def _resolve_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class MyPolicyAdapter:
    """Wraps an openpi `Policy` and exposes the AIRoA HSR contract.

    The harness only calls ``policy.infer(obs) -> dict``. See README.md §5
    of `airoa-evaluation-ICRA` for the contract.
    """

    # 11D HSR action layout, as documented in the AIRoA README.
    ACTION_DIM = 11

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
        *,
        # Keyword-style aliases for compatibility with different server hooks.
        checkpoint_dir: str | os.PathLike[str] | None = None,
        config_name: str | None = None,
        device: str | None = None,
        default_prompt: str | None = None,
        discrete_state_input: bool | None = None,
    ) -> None:
        # Accept either positional `checkpoint_path` or keyword `checkpoint_dir`,
        # because the base-branch server template passes `checkpoint_dir=...`.
        ckpt = checkpoint_path if checkpoint_path is not None else checkpoint_dir
        if ckpt is None:
            ckpt = os.environ.get("POLICY_CHECKPOINT_DIR")
        if ckpt is None:
            raise ValueError(
                "MyPolicyAdapter requires a checkpoint directory "
                "(via constructor arg or POLICY_CHECKPOINT_DIR env var)."
            )

        ckpt_path = Path(ckpt).expanduser().resolve()
        if not ckpt_path.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory not found or not a directory: {ckpt_path}"
            )
        if not (ckpt_path / "params").exists():
            raise FileNotFoundError(
                f"Checkpoint at {ckpt_path} is missing the required 'params/' "
                "subdirectory. The harness expects a checkpoint *directory*, not a file."
            )

        cfg_name = config_name or os.environ.get("POLICY_CONFIG_NAME", DEFAULT_CONFIG_NAME)

        # The training run for this checkpoint enabled discrete proprioceptive state
        # tokens (`--model.discrete-state-input`). The default `pi05_hsr` train config
        # has `discrete_state_input=False`, so we patch the model config before
        # instantiating the policy. Override via env var or kwarg if you ever swap
        # in a checkpoint trained without that flag.
        env_dsi = os.environ.get("POLICY_DISCRETE_STATE_INPUT")
        dsi = _resolve_bool(
            discrete_state_input if discrete_state_input is not None else env_dsi,
            default=True,
        )

        prompt = default_prompt
        if prompt is None:
            prompt = os.environ.get("POLICY_DEFAULT_PROMPT")

        # Pytorch device kwarg is informational for JAX checkpoints (we still let
        # JAX auto-place on the visible GPU). We keep it for parity with the
        # serve_hsr_policy_ws.py CLI surface.
        self._pytorch_device = device or os.environ.get("POLICY_PYTORCH_DEVICE")

        logger.info(
            "Loading openpi policy: config_name=%s checkpoint=%s "
            "discrete_state_input=%s default_prompt=%r",
            cfg_name,
            ckpt_path,
            dsi,
            prompt,
        )

        train_cfg = train_config_lib.get_config(cfg_name)
        model_cfg = dataclasses.replace(train_cfg.model, discrete_state_input=dsi)
        train_cfg = dataclasses.replace(train_cfg, model=model_cfg)

        self._policy = policy_config.create_trained_policy(
            train_cfg,
            ckpt_path,
            default_prompt=prompt,
            pytorch_device=self._pytorch_device,
        )

        self._config_name = cfg_name
        self._checkpoint_dir = str(ckpt_path)
        self._discrete_state_input = dsi
        self._default_prompt = prompt

    # ------------------------------------------------------------------
    # AIRoA contract
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        meta = dict(getattr(self._policy, "metadata", {}) or {})
        meta.update(
            {
                "policy": "openpi.pi05_hsr",
                "config_name": self._config_name,
                "checkpoint_dir": self._checkpoint_dir,
                "discrete_state_input": self._discrete_state_input,
                "default_prompt": self._default_prompt,
                "action_dim": self.ACTION_DIM,
            }
        )
        return meta

    def infer(self, obs: dict) -> dict:
        # Validate the contract loudly — these mismatches are the most common
        # integration bug noted in docs/INTEGRATION_GUIDE_ja.md §10.
        for key in ("head_rgb", "hand_rgb", "state"):
            if key not in obs:
                raise KeyError(
                    f"Observation missing required key '{key}'. "
                    "Expected keys: head_rgb, hand_rgb, state, prompt."
                )

        head_rgb = np.asarray(obs["head_rgb"])
        hand_rgb = np.asarray(obs["hand_rgb"])
        state = np.asarray(obs["state"]).astype(np.float32)
        prompt = obs.get("prompt", "")

        # Translate AIRoA keys → openpi HSRInputs keys. HSRInputs (in
        # openpi.policies.hsr_policy) reads `observation/image`,
        # `observation/wrist_image`, `observation/state`, and `prompt`.
        openpi_obs = {
            "observation/image": head_rgb,
            "observation/wrist_image": hand_rgb,
            "observation/state": state,
            "prompt": prompt,
        }

        result = self._policy.infer(openpi_obs)

        actions = np.asarray(result["actions"], dtype=np.float32)
        # Pi05 + HSROutputs returns (T, 11). Defensive checks below keep us from
        # silently shipping wrong-shaped chunks (#10 in the integration guide).
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != self.ACTION_DIM or actions.shape[0] < 1:
            raise ValueError(
                f"Policy returned actions with bad shape {actions.shape}; "
                f"expected (T, {self.ACTION_DIM}) with T >= 1."
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("Policy returned non-finite actions (NaN/Inf).")

        return {"actions": actions}
