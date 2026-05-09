from __future__ import annotations

import os
from typing import Any

import numpy as np

from .model import XVLAHSRModel


class MyPolicyAdapter:
    """Policy object consumed by runtime_core.websocket_policy_server.WebsocketPolicyServer."""

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        device: str | None = None,
        **_: Any,
    ) -> None:
        checkpoint = checkpoint_dir or checkpoint_path
        self.model = XVLAHSRModel(checkpoint_dir=checkpoint, device=device)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.model.metadata

    def reset(self) -> None:
        self.model.reset()

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self.model.predict_action_chunk(obs)
        if actions.ndim != 2 or actions.shape[1] != 11:
            raise ValueError(f"Expected actions with shape (T, 11), got {actions.shape}")
        return {"actions": np.asarray(actions, dtype=np.float32)}
