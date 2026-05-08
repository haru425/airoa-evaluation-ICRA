from __future__ import annotations

from typing import Any

import numpy as np

from .model import VLAAdapterModel


class MyPolicyAdapter:
    """AIRoA websocket adapter for the portable VLA-Adapter HSR bundle."""

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        **_: Any,
    ) -> None:
        checkpoint = checkpoint_dir or checkpoint_path
        if checkpoint is None:
            raise ValueError("MyPolicyAdapter requires checkpoint_dir or checkpoint_path.")
        self.model = VLAAdapterModel(checkpoint, device=device)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.model.metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self.model.infer(obs)
        return {"actions": actions}

    def close(self) -> None:
        self.model.close()
