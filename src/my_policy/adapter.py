from __future__ import annotations

import os
from typing import Any

import numpy as np

from .model import HSRPortableOpenPIPolicy
from .model import validate_actions


class MyPolicyAdapter:
    def __init__(
        self,
        checkpoint_dir: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        checkpoint = (
            checkpoint_dir
            or checkpoint_path
            or os.environ.get("POLICY_CHECKPOINT_DIR")
            or os.environ.get("POLICY_CHECKPOINT_PATH")
        )
        if not checkpoint:
            raise ValueError("Set POLICY_CHECKPOINT_PATH to the HSR689 S2 portable policy bundle directory.")

        self._policy = HSRPortableOpenPIPolicy(checkpoint, pytorch_device=device)
        self.metadata = dict(self._policy.metadata)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        head_rgb = self._image(obs, "head_rgb")
        hand_rgb = self._image(obs, "hand_rgb")
        state = self._state(obs)
        prompt = str(obs.get("instruction") or obs.get("prompt") or "")
        resolved_prompt, resolver_info = self._policy.prepare_prompt(prompt)

        prepared = {
            "observation/image": head_rgb,
            "observation/wrist_image": hand_rgb,
            "observation/state": state,
            "prompt": resolved_prompt,
        }
        outputs = dict(self._policy.infer(prepared))
        actions = validate_actions(outputs["actions"])
        return {
            "actions": actions,
            "resolved_prompt": resolved_prompt,
            "resolver_match_type": resolver_info.get("match_type", "unknown"),
            "resolver_score": float(resolver_info.get("score", 1.0)),
            "resolved_original_instruction": resolver_info.get("matched_original_instruction", prompt),
            "resolved_canonical_refined_instruction": resolver_info.get("canonical_refined_instruction", ""),
            "resolved_canonical_parent_task": resolver_info.get("canonical_parent_task", ""),
            "policy_timing": outputs.get("policy_timing", {}),
        }

    @staticmethod
    def _image(obs: dict[str, Any], key: str) -> np.ndarray:
        if key not in obs:
            raise KeyError(f"Observation is missing {key!r}.")
        image = np.asarray(obs[key])
        if image.ndim != 3:
            raise ValueError(f"{key} must be a 3D image array, got shape {image.shape}.")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] != 3:
            raise ValueError(f"{key} must have 3 color channels, got shape {image.shape}.")
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    @staticmethod
    def _state(obs: dict[str, Any]) -> np.ndarray:
        if "state" not in obs:
            raise KeyError("Observation is missing 'state'.")
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"state must have shape (8,), got {state.shape}.")
        if not np.all(np.isfinite(state)):
            raise ValueError("state contains non-finite values.")
        return state
