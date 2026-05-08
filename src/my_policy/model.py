from __future__ import annotations

import contextlib
import dataclasses
import glob
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

HSR_ACTION_NAMES: tuple[str, ...] = (
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "gripper",
    "head_pan_joint",
    "head_tilt_joint",
    "base_x",
    "base_y",
    "base_t",
)

# The VLA-Adapter loader mutates these files at load time. In AIRoA the
# checkpoint is mounted read-only, so mutable files must be copied into /tmp
# while large weight files can stay as symlinks.
MUTABLE_FILENAMES = {
    "config.json",
    "configuration_prismatic.py",
    "modeling_prismatic.py",
    "training_config.json",
}


@dataclasses.dataclass
class StagedTree:
    source_root: Path
    staged_root: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.staged_root, ignore_errors=True)


@dataclasses.dataclass(frozen=True)
class BundleLayout:
    bundle_root: Path
    checkpoint_root: Path
    manifest: dict[str, Any]

    @property
    def backend_config(self) -> dict[str, Any]:
        backend_config = self.manifest.get("backend_config", {})
        return backend_config if isinstance(backend_config, dict) else {}


class VLAAdapterModel:
    """Loads the portable VLA-Adapter HSR bundle and returns AIRoA actions."""

    def __init__(self, checkpoint_dir: str | os.PathLike[str], device: str | None = None) -> None:
        self.layout = _resolve_bundle_layout(Path(checkpoint_dir).expanduser())
        self.project_root = self.layout.bundle_root / "archive" / "vla_adapter"
        if not self.project_root.is_dir():
            raise FileNotFoundError(
                f"VLA-Adapter archive not found under portable bundle: {self.project_root}"
            )

        _configure_vla_imports(self.project_root)

        self.staged_tree = _stage_model_tree(self.layout.checkpoint_root, prefix="airoa_vla_adapter_")
        try:
            _rewrite_training_config_for_portable_bundle(self.staged_tree.staged_root, self.layout.bundle_root)

            backend = self.layout.backend_config
            cfg = SimpleNamespace(
                pretrained_checkpoint=str(self.staged_tree.staged_root),
                unnorm_key=str(
                    backend.get("unnorm_key")
                    or backend.get("dataset_repo_id")
                    or self.layout.checkpoint_root.name
                ),
                use_l1_regression=bool(backend.get("use_l1_regression", True)),
                use_film=bool(backend.get("use_film", False)),
                num_images_in_input=int(backend.get("num_images_in_input", 2)),
                use_proprio=bool(backend.get("use_proprio", True)),
                center_crop=bool(backend.get("center_crop", True)),
                load_in_8bit=bool(backend.get("load_in_8bit", False)),
                load_in_4bit=bool(backend.get("load_in_4bit", False)),
                num_open_loop_steps=int(backend.get("num_open_loop_steps", 64)),
                use_minivlm=bool(backend.get("use_minivlm", True)),
                use_pro_version=bool(backend.get("use_pro_version", True)),
                save_version=str(backend.get("save_version", "")),
            )
            self.cfg = cfg
            self.device = _resolve_torch_device(device)

            from experiments.robot import openvla_utils
            from experiments.robot.openvla_utils import get_action_head
            from experiments.robot.openvla_utils import get_processor
            from experiments.robot.openvla_utils import get_proprio_projector
            from experiments.robot.openvla_utils import get_vla
            from experiments.robot.openvla_utils import get_vla_action

            # openvla_utils uses a module-level DEVICE constant. Override it after
            # import so POLICY_PYTORCH_DEVICE=cpu remains useful for debugging.
            openvla_utils.DEVICE = self.device
            self._get_vla_action = get_vla_action

            with _pushd(self.project_root):
                self.vla = get_vla(cfg)
                self.processor = get_processor(cfg)
                self.action_head = get_action_head(cfg, self.vla.llm_dim)
                self.proprio_projector = None
                if cfg.use_proprio:
                    from prismatic.vla.constants import PROPRIO_DIM

                    self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)
        except Exception:
            self.staged_tree.cleanup()
            raise

        self._metadata = {
            "policy": "vla_adapter",
            "action_order": "hsr_raw_v1",
            "action_names": list(HSR_ACTION_NAMES),
            "bundle_root": str(self.layout.bundle_root),
            "checkpoint_root": str(self.layout.checkpoint_root),
            "staged_checkpoint_root": str(self.staged_tree.staged_root),
            "unnorm_key": cfg.unnorm_key,
            "num_open_loop_steps": cfg.num_open_loop_steps,
            "num_images_in_input": cfg.num_images_in_input,
            "use_minivlm": cfg.use_minivlm,
            "use_proprio": cfg.use_proprio,
            "device": str(self.device),
        }
        LOGGER.info(
            "Loaded VLA-Adapter bundle=%s checkpoint=%s staged=%s device=%s",
            self.layout.bundle_root,
            self.layout.checkpoint_root,
            self.staged_tree.staged_root,
            self.device,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        head_rgb = _coerce_rgb(obs["head_rgb"], name="head_rgb")
        hand_rgb = _coerce_rgb(obs["hand_rgb"], name="hand_rgb")
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"Expected obs['state'] shape (8,), got {state.shape}.")

        prompt = str(obs.get("prompt") or _default_prompt(self.layout.manifest))
        if not prompt:
            raise ValueError("VLA-Adapter inference requires obs['prompt'] or POLICY_DEFAULT_PROMPT.")

        observation = {
            "full_image": head_rgb,
            "wrist_image": hand_rgb,
            "state": state.copy(),
        }
        actions = self._get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            observation,
            prompt,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
            use_minivlm=self.cfg.use_minivlm,
        )
        return _coerce_hsr_action_chunk(actions)

    def close(self) -> None:
        self.staged_tree.cleanup()


def _resolve_bundle_layout(checkpoint_dir: Path) -> BundleLayout:
    root = checkpoint_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {root}")

    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        checkpoint_value = manifest.get("checkpoint_path", "checkpoint")
        checkpoint_root = _resolve_path_relative_to(root, str(checkpoint_value))
        if not checkpoint_root.is_dir():
            raise FileNotFoundError(f"manifest checkpoint_path does not exist: {checkpoint_root}")
        return BundleLayout(bundle_root=root, checkpoint_root=checkpoint_root, manifest=manifest)

    # Also support passing the inner checkpoint/ directory directly.
    if (root / "training_config.json").is_file():
        parent_manifest = root.parent / "manifest.json"
        manifest = _read_json(parent_manifest) if parent_manifest.is_file() else {}
        return BundleLayout(bundle_root=root.parent, checkpoint_root=root, manifest=manifest)

    raise FileNotFoundError(
        "Expected POLICY_CHECKPOINT_PATH to point at the portable bundle root "
        "or its checkpoint/ directory."
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_path_relative_to(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _configure_vla_imports(project_root: Path) -> None:
    os.environ.setdefault("PRISMATIC_USE_FLASH_ATTENTION_2", "0")
    os.environ.setdefault("VLA_ROBOT_PLATFORM_OVERRIDE", "HSR")

    site_packages = _bundle_site_packages(project_root)
    if site_packages and os.environ.get("VLA_ADAPTER_USE_BUNDLE_SITE_PACKAGES", "1") != "0":
        if os.environ.get("VLA_ADAPTER_PREFER_BUNDLE_SITE_PACKAGES", "1") == "1":
            _import_current_core_modules()
            _prepend_sys_path(site_packages)
        else:
            _append_sys_path(site_packages)

    # Keep archived VLA-Adapter code ahead of installed packages.
    _prepend_sys_path(project_root)
    _prepend_sys_path(project_root / "experiments" / "robot")


def _bundle_site_packages(project_root: Path) -> Path | None:
    matches = sorted(glob.glob(str(project_root / ".venv" / "lib" / "python*" / "site-packages")))
    current = f"python{sys.version_info.major}.{sys.version_info.minor}"
    compatible = [Path(path) for path in matches if Path(path).parent.name == current]
    if compatible:
        return compatible[-1]
    if matches:
        LOGGER.warning(
            "Ignoring bundled site-packages because it does not match this Python (%s): %s",
            current,
            matches,
        )
    return None


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _append_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.append(value)


def _import_current_core_modules() -> None:
    # When the portable bundle's site-packages is used, keep binary packages
    # from the container environment loaded in sys.modules. The bundle provides
    # VLA-specific pure Python deps, but the container owns CUDA/JAX/NumPy ABI.
    with contextlib.suppress(Exception):
        import numpy  # noqa: F401

    with contextlib.suppress(Exception):
        import ml_dtypes  # noqa: F401

    with contextlib.suppress(Exception):
        import jax  # noqa: F401

    with contextlib.suppress(Exception):
        import typing_extensions  # noqa: F401

    with contextlib.suppress(Exception):
        import pydantic  # noqa: F401

    with contextlib.suppress(Exception):
        import pydantic_core  # noqa: F401

    with contextlib.suppress(Exception):
        import wandb  # noqa: F401

    with contextlib.suppress(Exception):
        import torch  # noqa: F401

    with contextlib.suppress(Exception):
        import torchvision  # noqa: F401


def _resolve_torch_device(device: str | None):
    import torch

    requested = device or os.environ.get("POLICY_PYTORCH_DEVICE") or "cuda"
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def _stage_model_tree(source_root: Path, *, prefix: str) -> StagedTree:
    staged_root = Path(tempfile.mkdtemp(prefix=prefix))
    for source_path in sorted(source_root.rglob("*")):
        rel_path = source_path.relative_to(source_root)
        dest_path = staged_root / rel_path
        if source_path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.name in MUTABLE_FILENAMES:
            shutil.copy2(source_path, dest_path)
        else:
            os.symlink(source_path, dest_path)

    (staged_root / "stage_manifest.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "staged_root": str(staged_root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return StagedTree(source_root=source_root, staged_root=staged_root)


def _rewrite_training_config_for_portable_bundle(staged_checkpoint: Path, bundle_root: Path) -> None:
    training_cfg_path = staged_checkpoint / "training_config.json"
    portable_vlm_root = bundle_root / "_portable_assets" / "vla_adapter_base_vlm"
    if not training_cfg_path.is_file() or not portable_vlm_root.exists():
        return

    training_cfg = _read_json(training_cfg_path)
    training_cfg["vlm_path"] = str(portable_vlm_root.resolve())
    if training_cfg.get("config_file_path"):
        training_cfg["config_file_path"] = str(staged_checkpoint.resolve())
    training_cfg_path.write_text(json.dumps(training_cfg, indent=2))


def _coerce_rgb(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected {name} to have shape (H, W, 3), got {arr.shape}.")
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _coerce_hsr_action_chunk(actions: Any) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected actions to have shape (T, 11), got {arr.shape}.")
    if arr.shape[1] > len(HSR_ACTION_NAMES):
        tail = arr[:, len(HSR_ACTION_NAMES) :]
        if np.allclose(tail, 0.0, atol=1e-6):
            arr = arr[:, : len(HSR_ACTION_NAMES)]
    if arr.shape[1] != len(HSR_ACTION_NAMES):
        raise ValueError(f"Expected action dim 11, got {arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError("Expected at least one action step.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Policy produced non-finite actions.")
    return arr.astype(np.float32, copy=False)


def _default_prompt(manifest: dict[str, Any]) -> str:
    env_prompt = os.environ.get("POLICY_DEFAULT_PROMPT", "")
    if env_prompt:
        return env_prompt
    serve = manifest.get("serve", {})
    if isinstance(serve, dict) and serve.get("default_prompt"):
        return str(serve["default_prompt"])
    return ""


@contextlib.contextmanager
def _pushd(path: Path):
    cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)
