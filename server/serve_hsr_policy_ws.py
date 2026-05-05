#!/usr/bin/env python3
"""WebSocket policy server for the AIRoA HSR evaluation harness.

Loads `MyPolicyAdapter` (`src/my_policy/adapter.py`), which wraps an openpi
JAX `pi05_hsr` checkpoint trained with `openpi_offline_rl`. The adapter
exposes the single method the harness calls:

    policy.infer(obs: dict) -> dict

See `docs/INTEGRATION_GUIDE_ja.md` for the integration contract and
`README.md` §5 for the WebSocket I/O dictionary shapes.

`POLICY_MODULE` (env var) can still override which adapter is loaded — this
is handy for swapping in a `ZeroPolicy` to confirm the harness round-trip
without spinning up the full model.
"""

import argparse
import importlib
import logging
import os
from pathlib import Path

import numpy as np

from runtime_core.websocket_policy_server import WebsocketPolicyServer


class ZeroPolicy:
    """Placeholder policy that returns zero actions of the correct shape.

    Useful only for verifying the harness round-trip. Set
    `POLICY_MODULE=server.serve_hsr_policy_ws:ZeroPolicy` (or just
    `--policy-module` on the CLI) if you want to bypass the real model.
    """

    def __init__(self, checkpoint_dir: str | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            logging.info("ZeroPolicy: checkpoint_dir=%s (not loaded — placeholder)", checkpoint_dir)

    @property
    def metadata(self) -> dict:
        return {"policy": "zero", "actions_shape": [1, 11]}

    def infer(self, obs: dict) -> dict:
        return {"actions": np.zeros((1, 11), dtype=np.float32)}


def _load_policy(policy_module: str | None, checkpoint_dir: str | None, pytorch_device: str | None):
    """Resolve and instantiate the policy class.

    - If `policy_module` is provided as `module:Class`, dynamically import it.
      The class is called with `checkpoint_dir=...` and (when accepted)
      `device=...` so we don't have to hard-code the constructor signature.
    - Otherwise default to `my_policy.adapter:MyPolicyAdapter`.
    """
    if not policy_module:
        # Default path for this submission: the openpi-backed adapter.
        from my_policy.adapter import MyPolicyAdapter

        return MyPolicyAdapter(
            checkpoint_path=checkpoint_dir,
            device=pytorch_device,
        )

    if ":" not in policy_module:
        raise ValueError(
            f"--policy-module must be in 'module:Class' form, got: {policy_module!r}"
        )
    module_name, class_name = policy_module.split(":", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    # Best-effort kwargs — pass `device` only if the class accepts it.
    kwargs: dict = {"checkpoint_dir": checkpoint_dir}
    try:
        import inspect

        params = inspect.signature(cls).parameters
        if "device" in params and pytorch_device is not None:
            kwargs["device"] = pytorch_device
        elif "pytorch_device" in params and pytorch_device is not None:
            kwargs["pytorch_device"] = pytorch_device
    except (TypeError, ValueError):
        pass
    return cls(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIRoA HSR evaluation websocket policy server")
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("POLICY_CHECKPOINT_DIR"),
        help="Path to checkpoint directory (passed to the policy class).",
    )
    parser.add_argument(
        "--policy-module",
        default=os.environ.get("POLICY_MODULE"),
        help=(
            "Optional override of the policy class in 'module:Class' form. "
            "If unset, defaults to 'my_policy.adapter:MyPolicyAdapter'."
        ),
    )
    parser.add_argument(
        "--pytorch-device",
        default=os.environ.get("POLICY_PYTORCH_DEVICE"),
        help='Optional torch device override (e.g. "cuda", "cuda:0", "cpu").',
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint_dir:
        checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")
    else:
        checkpoint_dir = None

    policy = _load_policy(args.policy_module, checkpoint_dir, args.pytorch_device)

    metadata = dict(getattr(policy, "metadata", {}) or {})
    metadata.update(
        {
            "checkpoint_dir": checkpoint_dir,
            "policy_module": args.policy_module or "my_policy.adapter:MyPolicyAdapter",
            "server_host": args.host,
            "server_port": args.port,
        }
    )

    logging.info(
        "Serving policy=%s checkpoint=%s on %s:%s",
        args.policy_module or "my_policy.adapter:MyPolicyAdapter",
        checkpoint_dir,
        args.host,
        args.port,
    )
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
