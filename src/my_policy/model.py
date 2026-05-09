from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)

HSR_RAW_ACTION_NAMES: tuple[str, ...] = (
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "hand_motor_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "base_x",
    "base_y",
    "base_t",
)

DEFAULT_RENAME_MAP: dict[str, str] = {
    "observation.image.head": "observation.images.image",
    "observation.image.hand": "observation.images.image2",
}


@dataclasses.dataclass(frozen=True)
class _BundlePaths:
    bundle_root: Path
    checkpoint_dir: Path
    manifest_path: Path | None
    lerobot_project_root: Path | None
    lerobot_src_root: Path | None


@dataclasses.dataclass(frozen=True)
class _FreezeSpec:
    enabled: bool
    indices: tuple[int, ...]
    means: tuple[float, ...]
    names: tuple[str, ...]


def _resolve_checkpoint_path(checkpoint_dir: str | os.PathLike[str] | None) -> Path:
    if checkpoint_dir:
        return Path(checkpoint_dir).expanduser().resolve()

    env_path = os.environ.get("POLICY_CHECKPOINT_DIR") or os.environ.get("POLICY_CHECKPOINT_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    raise ValueError(
        "checkpoint_dir is required. Set POLICY_CHECKPOINT_PATH to the portable bundle root "
        "and POLICY_MODULE=my_policy.adapter:MyPolicyAdapter."
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_bundle_paths(checkpoint_dir: str | os.PathLike[str] | None) -> _BundlePaths:
    root = _resolve_checkpoint_path(checkpoint_dir)
    if not root.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {root}")

    if (root / "manifest.json").is_file():
        manifest_path = root / "manifest.json"
        manifest = _read_json(manifest_path)
        checkpoint_rel = manifest.get("checkpoint_path", "checkpoint")
        checkpoint_path = Path(checkpoint_rel)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        bundle_root = root
    elif (root / "config.json").is_file() and (root / "model.safetensors").is_file():
        checkpoint_path = root
        bundle_root = root.parent
        manifest_path = bundle_root / "manifest.json"
        if not manifest_path.is_file():
            manifest_path = None
    else:
        raise FileNotFoundError(
            "Expected either a portable bundle root containing manifest.json, "
            f"or a LeRobot checkpoint directory containing config.json/model.safetensors. Got: {root}"
        )

    checkpoint_path = checkpoint_path.resolve()
    if not (checkpoint_path / "config.json").is_file():
        raise FileNotFoundError(f"LeRobot checkpoint config not found: {checkpoint_path / 'config.json'}")
    if not (checkpoint_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"LeRobot checkpoint weights not found: {checkpoint_path / 'model.safetensors'}")

    lerobot_project_root = bundle_root / "external" / "lerobot-xvla-official"
    lerobot_src_root = lerobot_project_root / "src"
    if not lerobot_src_root.is_dir():
        lerobot_project_root = None
        lerobot_src_root = None

    return _BundlePaths(
        bundle_root=bundle_root.resolve(),
        checkpoint_dir=checkpoint_path,
        manifest_path=manifest_path.resolve() if manifest_path else None,
        lerobot_project_root=lerobot_project_root.resolve() if lerobot_project_root else None,
        lerobot_src_root=lerobot_src_root.resolve() if lerobot_src_root else None,
    )


def _prepend_sys_path(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def _evict_lerobot_if_needed(expected_src_root: Path | None) -> None:
    if expected_src_root is None:
        return

    module = sys.modules.get("lerobot")
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return

    try:
        module_path = Path(module_file).resolve()
        module_path.relative_to(expected_src_root)
        return
    except ValueError:
        pass

    for name in list(sys.modules):
        if name == "lerobot" or name.startswith("lerobot."):
            del sys.modules[name]


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _make_lerobot_py311_compatible(src_root: Path) -> None:
    io_utils = src_root / "lerobot" / "utils" / "io_utils.py"
    _replace_once(io_utils, "from typing import Any\n", "from typing import Any, TypeVar\n")
    _replace_once(
        io_utils,
        'JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]\n',
        (
            'JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]\n'
            'T = TypeVar("T", bound=JsonLike)\n'
        ),
    )
    _replace_once(
        io_utils,
        "def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:",
        "def deserialize_json_into_object(fpath: Path, obj: T) -> T:",
    )

    motors_bus = src_root / "lerobot" / "motors" / "motors_bus.py"
    _replace_once(motors_bus, "type NameOrID = str | int\n", "NameOrID = str | int\n")
    _replace_once(motors_bus, "type Value = int | float\n", "Value = int | float\n")

    streaming_dataset = src_root / "lerobot" / "datasets" / "streaming_dataset.py"
    _replace_once(
        streaming_dataset,
        "from collections.abc import Callable, Generator, Iterable, Iterator\n",
        "from collections.abc import Callable, Generator, Iterable, Iterator\nfrom typing import Generic, TypeVar\n",
    )
    _replace_once(streaming_dataset, "class Backtrackable[T]:", "T = TypeVar(\"T\")\n\n\nclass Backtrackable(Generic[T]):")

    pipeline = src_root / "lerobot" / "processor" / "pipeline.py"
    _replace_once(
        pipeline,
        "from typing import Any, TypedDict, TypeVar, cast\n",
        "from typing import Any, Generic, TypedDict, TypeVar, cast\n",
    )
    _replace_once(
        pipeline,
        "class DataProcessorPipeline[TInput, TOutput](HubMixin):",
        "class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):",
    )


def _prepare_lerobot_src_root(src_root: Path | None) -> Path | None:
    if src_root is None or sys.version_info >= (3, 12):
        return src_root

    source_key = f"{src_root.resolve()}:{src_root.stat().st_mtime_ns}"
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
    compat_src = Path("/tmp") / "airoa_lerobot_xvla_py311" / digest
    if compat_src.is_dir():
        return compat_src

    tmp_src = compat_src.with_name(f"{compat_src.name}.tmp.{os.getpid()}")
    if tmp_src.exists():
        shutil.rmtree(tmp_src)
    shutil.copytree(
        src_root,
        tmp_src,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    _make_lerobot_py311_compatible(tmp_src)
    compat_src.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp_src.rename(compat_src)
    except FileExistsError:
        shutil.rmtree(tmp_src)
    return compat_src


def _select_device(device: str | None, manifest: dict[str, Any]):
    import torch

    requested = (
        device
        or os.environ.get("POLICY_PYTORCH_DEVICE")
        or manifest.get("serve", {}).get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("Requested device %s but CUDA is unavailable; falling back to cpu", requested)
        requested = "cpu"
    return torch.device(requested)


def _find_cached_bart_tokenizer() -> str | None:
    env_candidates = [
        os.environ.get("XVLA_TOKENIZER_PATH"),
        os.environ.get("BART_TOKENIZER_PATH"),
        os.environ.get("TOKENIZER_PATH"),
    ]
    for candidate in env_candidates:
        if candidate and (Path(candidate).expanduser() / "tokenizer.json").is_file():
            return str(Path(candidate).expanduser().resolve())

    cache_roots: list[Path] = []
    for env_name in ("HF_HOME", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_name)
        if value:
            cache_roots.append(Path(value).expanduser())
    cache_roots.extend(
        [
            Path("/policy_hf_cache"),
            Path.home() / ".cache" / "huggingface",
        ]
    )

    for cache_root in cache_roots:
        hub_root = cache_root / "hub" / "models--facebook--bart-large" / "snapshots"
        if not hub_root.is_dir():
            continue
        snapshots = sorted(hub_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
        for snapshot in snapshots:
            if (snapshot / "tokenizer.json").is_file() and (snapshot / "vocab.json").is_file():
                return str(snapshot.resolve())
    return None


def _coerce_rgb_hwc(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"{name} must be an RGB image in HWC layout, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def _coerce_state(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.shape != (8,):
        raise ValueError(f"obs['state'] must be an 8D HSR state vector, got shape {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("obs['state'] contains non-finite values")
    return state


def _coerce_hsr_action_chunk(actions: Any, *, expected_dim: int = len(HSR_RAW_ACTION_NAMES)) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected HSR actions with shape (T, D), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("Expected at least one action step")
    if arr.shape[-1] != expected_dim:
        if arr.shape[-1] > expected_dim and np.allclose(arr[:, expected_dim:], 0.0, atol=1e-6):
            arr = arr[:, :expected_dim]
        else:
            raise ValueError(f"Expected HSR action dim {expected_dim}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Model returned non-finite actions")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _resolve_optional_path(bundle_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = bundle_root / path
    return path.resolve()


def _load_freeze_spec(bundle_root: Path, manifest: dict[str, Any]) -> _FreezeSpec | None:
    freeze_cfg = manifest.get("freeze") or {}
    enabled = bool(freeze_cfg.get("enabled", False))
    screening_path = _resolve_optional_path(bundle_root, freeze_cfg.get("action_screening_path"))
    if screening_path is None:
        if enabled:
            raise FileNotFoundError("Freeze is enabled but manifest has no action_screening_path")
        return None
    if not screening_path.is_file():
        if enabled:
            raise FileNotFoundError(f"Freeze screening file not found: {screening_path}")
        return None

    payload = _read_json(screening_path)
    indices = tuple(int(value) for value in payload.get("recommended_frozen_raw_indices", []))
    means = tuple(float(value) for value in payload.get("recommended_frozen_raw_means", []))
    names = tuple(str(value) for value in payload.get("recommended_frozen_action_names", []))
    if len(indices) != len(means):
        raise ValueError(f"Freeze screening has mismatched indices/means: {screening_path}")
    for index in indices:
        if index < 0 or index >= len(HSR_RAW_ACTION_NAMES):
            raise ValueError(f"Freeze raw action index out of range: {index}")
    return _FreezeSpec(enabled=enabled and bool(indices), indices=indices, means=means, names=names)


def _apply_freeze(actions: np.ndarray, freeze_spec: _FreezeSpec | None) -> np.ndarray:
    if freeze_spec is None or not freeze_spec.enabled:
        return actions
    frozen = actions.copy()
    for index, mean in zip(freeze_spec.indices, freeze_spec.means, strict=True):
        frozen[:, index] = mean
    return frozen


class XVLAHSRModel:
    """AIRoA HSR wrapper around the xVLA LeRobot checkpoint bundled with the submission."""

    def __init__(self, checkpoint_dir: str | os.PathLike[str] | None, device: str | None = None) -> None:
        self.paths = _resolve_bundle_paths(checkpoint_dir)
        manifest = _read_json(self.paths.manifest_path) if self.paths.manifest_path else {}
        if manifest.get("backend") not in (None, "xvla"):
            raise ValueError(f"Expected an xvla bundle, got backend={manifest.get('backend')!r}")

        self.manifest = manifest
        self.rename_map = dict(manifest.get("backend_config", {}).get("rename_map") or DEFAULT_RENAME_MAP)
        serve_cfg = manifest.get("serve", {})
        self.default_prompt = serve_cfg.get("default_prompt") or ""
        self.robot_type = serve_cfg.get("robot_type") or "hsr"
        self.use_amp = bool(serve_cfg.get("use_amp", True))
        self.freeze_spec = _load_freeze_spec(self.paths.bundle_root, manifest)
        # Only the first step of each predicted chunk carries prompt-conditioned
        # signal for this checkpoint; later steps collapse to the dataset action
        # mean (gripper≈0.42, base_x≈0.015) which on HSR shows up as "slight
        # gripper close + constant forward drive". We return that many steps.
        self.useful_action_steps = max(1, int(os.environ.get("XVLA_USEFUL_ACTION_STEPS", "1")))

        lerobot_src_root = _prepare_lerobot_src_root(self.paths.lerobot_src_root)
        _prepend_sys_path(self.paths.lerobot_project_root)
        _prepend_sys_path(lerobot_src_root)
        _evict_lerobot_if_needed(lerobot_src_root)

        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import get_policy_class
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies import prepare_observation_for_inference

        self.torch = torch
        self.prepare_observation_for_inference = prepare_observation_for_inference
        self.device = _select_device(device, manifest)

        policy_cfg = PreTrainedConfig.from_pretrained(str(self.paths.checkpoint_dir))
        policy_cfg.device = str(self.device)
        policy_class = get_policy_class(policy_cfg.type)
        self.policy = policy_class.from_pretrained(str(self.paths.checkpoint_dir), config=policy_cfg)
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        self.policy_cfg = policy_cfg

        preprocessor_overrides: dict[str, dict[str, Any]] = {
            "device_processor": {"device": str(self.device)},
            "rename_observations_processor": {"rename_map": self.rename_map},
        }
        tokenizer_path = _find_cached_bart_tokenizer()
        if tokenizer_path is not None:
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_path}

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(self.paths.checkpoint_dir),
            preprocessor_overrides=preprocessor_overrides,
        )

        self._metadata = {
            "policy": "xvla",
            "policy_type": policy_cfg.type,
            "checkpoint_dir": str(self.paths.checkpoint_dir),
            "bundle_root": str(self.paths.bundle_root),
            "device": str(self.device),
            "action_order": "hsr_raw_v1",
            "action_names": list(HSR_RAW_ACTION_NAMES),
            "rename_map": self.rename_map,
            "tokenizer_path": tokenizer_path,
            "freeze": {
                "enabled": bool(self.freeze_spec and self.freeze_spec.enabled),
                "frozen_action_names": list(self.freeze_spec.names) if self.freeze_spec else [],
                "frozen_raw_indices": list(self.freeze_spec.indices) if self.freeze_spec else [],
                "frozen_raw_means": list(self.freeze_spec.means) if self.freeze_spec else [],
            },
        }
        LOGGER.info("Loaded xVLA HSR checkpoint=%s device=%s", self.paths.checkpoint_dir, self.device)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def predict_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        if obs.get("policy_reset", False):
            self.reset()

        observation = {
            "observation.image.head": _coerce_rgb_hwc(obs["head_rgb"], name="head_rgb"),
            "observation.image.hand": _coerce_rgb_hwc(obs["hand_rgb"], name="hand_rgb"),
            "observation.state": _coerce_state(obs["state"]),
        }
        prompt = str(obs.get("prompt") or self.default_prompt or "")

        autocast_context = (
            self.torch.autocast(device_type=self.device.type)
            if self.use_amp and self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with self.torch.inference_mode(), autocast_context:
            prepared = self.prepare_observation_for_inference(
                observation,
                self.device,
                task=prompt,
                robot_type=self.robot_type,
            )
            batch = self.preprocessor(prepared)
            action_tensor = self.policy.predict_action_chunk(batch)
            if action_tensor.ndim == 2:
                action_tensor = action_tensor.unsqueeze(0)

            num_useful = min(self.useful_action_steps, action_tensor.shape[1])
            processed_actions = []
            for step_index in range(num_useful):
                single_action = action_tensor[:, step_index, :]
                processed_actions.append(self.postprocessor(single_action))

            action_chunk = self.torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu().numpy()

        actions = _coerce_hsr_action_chunk(action_chunk)
        return _apply_freeze(actions, self.freeze_spec)
