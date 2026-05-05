import dataclasses
from collections.abc import Iterator, Sequence
import logging
import multiprocessing as mp
import os
from types import SimpleNamespace
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar
import random

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import json

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


_LIBERO_LEFT_RIGHT_TRANSLATION_SCALE = np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
_LIBERO_LEFT_RIGHT_ROTATION_SCALE = np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)

_SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_INSTRUCTION = "discriminator_instruction"
_SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_JOINT = "discriminator_joint"
_SIMILARITY_FILTER_CONTEXT_CRITIC_INSTRUCTION = "critic_instruction"
_SIMILARITY_FILTER_CONTEXT_ACTOR_INSTRUCTION = "actor_instruction"
_SIMILARITY_FILTER_CONTEXT_ACTOR_JOINT = "actor_joint"
_LANGUAGE_SWITCH_SIMILARITIES_FIELD = "language_switch_similarities"
_LANGUAGE_SWITCH_STATE_SIMILARITIES_FIELD = "language_switch_state_similarities"
_SIMILARITY_FILTER_CONTEXTS = (
    _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_INSTRUCTION,
    _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_JOINT,
    _SIMILARITY_FILTER_CONTEXT_CRITIC_INSTRUCTION,
    _SIMILARITY_FILTER_CONTEXT_ACTOR_INSTRUCTION,
    _SIMILARITY_FILTER_CONTEXT_ACTOR_JOINT,
)

_LOADER_ROLE_GENERIC = 0
_LOADER_ROLE_DISCRIMINATOR = 1
_LOADER_ROLE_CRITIC = 2
_LOADER_ROLE_ACTOR = 3
_LOADER_ROLE_TO_NAME = {
    _LOADER_ROLE_GENERIC: "generic",
    _LOADER_ROLE_DISCRIMINATOR: "discriminator",
    _LOADER_ROLE_CRITIC: "critic",
    _LOADER_ROLE_ACTOR: "actor",
}
_LOADER_ROLE_FROM_NAME = {name: role for role, name in _LOADER_ROLE_TO_NAME.items()}


def _make_local_shared_value(value: float | int) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _encode_optional_similarity_bound(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _decode_optional_similarity_bound(value: object) -> float | None:
    try:
        bound = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric similarity bound, got {value!r}.") from exc
    if np.isnan(bound):
        return None
    return bound


def resolve_instruction_relabeling_similarity_filters(
    data_config: _config.DataConfig,
) -> dict[str, tuple[float | None, float | None]]:
    return {
        _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_INSTRUCTION: (
            data_config.discriminator_instruction_relabeling_similarity_min,
            data_config.discriminator_instruction_relabeling_similarity_max,
        ),
        _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_JOINT: (
            data_config.discriminator_joint_instruction_relabeling_similarity_min,
            data_config.discriminator_joint_instruction_relabeling_similarity_max,
        ),
        _SIMILARITY_FILTER_CONTEXT_CRITIC_INSTRUCTION: (
            data_config.critic_instruction_relabeling_similarity_min,
            data_config.critic_instruction_relabeling_similarity_max,
        ),
        _SIMILARITY_FILTER_CONTEXT_ACTOR_INSTRUCTION: (
            data_config.actor_instruction_relabeling_similarity_min,
            data_config.actor_instruction_relabeling_similarity_max,
        ),
        _SIMILARITY_FILTER_CONTEXT_ACTOR_JOINT: (
            data_config.actor_joint_instruction_relabeling_similarity_min,
            data_config.actor_joint_instruction_relabeling_similarity_max,
        ),
    }


def infer_iql_loader_role(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
) -> str:
    candidate_roles: set[str] = set()
    discriminator_ratios = (
        float(data_config.instruction_relabeling_ratio_for_discriminator),
        float(getattr(model_config, "action_relabeling_ratio_for_disc_l", 0.0)),
        float(getattr(model_config, "joint_relabeling_ratio_for_disc_l", 0.0)),
        float(getattr(model_config, "flipped_image_action_ratio_for_disc_l", 0.0)),
        float(getattr(model_config, "flipped_image_only_ratio_for_disc_l", 0.0)),
        float(getattr(model_config, "flipped_action_only_ratio_for_disc_l", 0.0)),
    )
    if any(ratio > 0.0 for ratio in discriminator_ratios):
        candidate_roles.add("discriminator")
    if float(data_config.instruction_relabeling_ratio_for_critic) > 0.0:
        candidate_roles.add("critic")
    if (
        float(data_config.instruction_relabeling_ratio_for_actor) > 0.0
        or float(data_config.actor_action_only_relabeling_ratio) > 0.0
        or float(data_config.actor_joint_relabeling_ratio) > 0.0
    ):
        candidate_roles.add("actor")

    if len(candidate_roles) == 1:
        return next(iter(candidate_roles))
    return "generic"


def _instruction_discriminator_uses_proprio(model_config: _model.BaseModelConfig) -> bool:
    if bool(getattr(model_config, "pi05", False)):
        return bool(getattr(model_config, "discrete_state_input", False))
    return float(getattr(model_config, "discriminator_proprio_zero_probability", 0.0)) < 1.0


def _get_image_width_axis(images: np.ndarray) -> int:
    if images.ndim < 2:
        raise ValueError(f"Expected image array with ndim >= 2, got shape={tuple(images.shape)}.")
    if images.ndim == 2:
        return -1

    is_channel_last = images.shape[-1] in (1, 3, 4)
    is_channel_first = images.shape[-3] in (1, 3, 4)
    if is_channel_last and not is_channel_first:
        return -2
    if is_channel_first and not is_channel_last:
        return -1
    if is_channel_last:
        return -2
    if is_channel_first:
        return -1
    return -1


def _mirror_libero_actions_left_right(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.shape[-1] != 7:
        raise ValueError(
            "Expected LIBERO actions with final dimension 7 "
            f"(dx, dy, dz, dax, day, daz, gripper), got shape={tuple(arr.shape)}."
        )
    mirrored = arr.copy()
    mirrored[..., :3] *= _LIBERO_LEFT_RIGHT_TRANSLATION_SCALE
    mirrored[..., 3:6] *= _LIBERO_LEFT_RIGHT_ROTATION_SCALE
    return mirrored


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


def validate_validation_config(*, validation_interval: int, validation_fraction: float) -> None:
    if validation_interval < 0:
        raise ValueError(f"`validation_interval` must be >= 0, got {validation_interval}.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(f"`validation_fraction` must be within [0.0, 1.0), got {validation_fraction}.")
    validation_enabled = validation_interval > 0 or validation_fraction > 0.0
    if validation_enabled and not (validation_interval > 0 and validation_fraction > 0.0):
        raise ValueError("Validation requires both `validation_interval > 0` and `validation_fraction > 0`.")


class IndexedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset[T_co], indices: np.ndarray):
        self._dataset = dataset
        self._indices = np.asarray(indices, dtype=np.int64).reshape(-1)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._dataset[int(self._indices[int(index)])]

    def __len__(self) -> int:
        return int(self._indices.shape[0])


def _extract_episode_indices_from_source(source: object) -> np.ndarray | None:
    try:
        episode_indices = source["episode_index"]
    except Exception:
        episode_indices = getattr(source, "episode_index", None)
    if episode_indices is None:
        return None
    try:
        return np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    except Exception:
        return None


def _iter_nested_dataset_sources(dataset: object) -> Iterator[object]:
    pending = [dataset]
    seen: set[int] = set()
    while pending:
        source = pending.pop()
        if source is None:
            continue
        source_id = id(source)
        if source_id in seen:
            continue
        seen.add(source_id)
        yield source
        children = [getattr(source, attr, None) for attr in ("hf_dataset", "dataset", "_dataset")]
        pending.extend(child for child in reversed(children) if child is not None)


def extract_episode_indices(dataset: Dataset[object]) -> np.ndarray | None:
    dataset_length = len(dataset)
    for source in _iter_nested_dataset_sources(dataset):
        episode_indices = _extract_episode_indices_from_source(source)
        if episode_indices is None:
            continue
        if episode_indices.shape[0] == dataset_length:
            return episode_indices
    return None


def split_train_validation_indices(
    dataset: Dataset[object],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    num_samples = len(dataset)
    if num_samples < 2:
        raise ValueError("Validation requires a dataset with at least 2 samples.")

    target_validation_samples = min(num_samples - 1, max(1, int(np.ceil(num_samples * validation_fraction))))
    rng = np.random.default_rng(seed)
    episode_indices = extract_episode_indices(dataset)

    if episode_indices is not None:
        unique_episodes, counts = np.unique(episode_indices, return_counts=True)
        if unique_episodes.shape[0] >= 2:
            selected_episode_ids: list[int] = []
            selected_validation_samples = 0
            for position, perm_index in enumerate(rng.permutation(unique_episodes.shape[0])):
                remaining_episodes = unique_episodes.shape[0] - (position + 1)
                if remaining_episodes == 0:
                    break
                selected_episode_ids.append(int(unique_episodes[perm_index]))
                selected_validation_samples += int(counts[perm_index])
                if selected_validation_samples >= target_validation_samples:
                    break

            if selected_episode_ids:
                validation_mask = np.isin(episode_indices, np.asarray(selected_episode_ids, dtype=np.int64))
                validation_indices = np.flatnonzero(validation_mask)
                train_indices = np.flatnonzero(~validation_mask)
                if train_indices.size > 0 and validation_indices.size > 0:
                    return train_indices, validation_indices, "episode"

    permutation = rng.permutation(num_samples)
    validation_indices = np.sort(permutation[:target_validation_samples])
    train_indices = np.sort(permutation[target_validation_samples:])
    return train_indices, validation_indices, "sample"


def _create_dynamic_action_chunk_sampler(
    *,
    repo_id: str,
    action_horizon: int,
    proprio_similarity_threshold: float | None,
    seed: int,
):
    from openpi.training.dynamic_action_chunk_sampler import DynamicActionChunkSampler

    return DynamicActionChunkSampler(
        repo_id=repo_id,
        action_horizon=action_horizon,
        proprio_similarity_threshold=proprio_similarity_threshold,
        seed=seed,
    )


def _drop_unused_hf_dataset_columns(
    dataset: "lerobot_dataset.LeRobotDataset",
    *,
    data_config: _config.DataConfig,
) -> None:
    # The precomputed `language_switch_action_chunks` field can be huge (e.g.
    # ~3.3 MB/row for CALVIN with 389 candidates × 30 neighbors), and is only
    # consumed by the static (precomputed) discriminator action-chunk
    # relabeling path. When dynamic action-chunk sampling is enabled it is
    # never read, so dropping it from the underlying HF dataset avoids
    # decoding multi-MB of unused data per sample.
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        return
    column_names = set(getattr(hf_dataset, "column_names", []) or [])
    if not column_names:
        return
    drop: list[str] = []
    if getattr(data_config, "discriminator_dynamic_action_relabeling", False):
        for col in ("language_switch_action_chunks", "language_switch_action_chunk_mask"):
            if col in column_names:
                drop.append(col)
    if drop:
        logging.info("Dropping unused HF dataset columns to speed up loading: %s", drop)
        dataset.hf_dataset = hf_dataset.remove_columns(drop)


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class CanonicalLeRobotDataset(Dataset[T_co]):
    """Aliases raw LeRobot feature keys to the canonical keys expected by OpenPI."""

    def __init__(self, dataset: Dataset, data_config: _config.DataConfig):
        self._dataset = dataset
        self.meta = getattr(dataset, "meta", None)

        aliases = {
            "image": data_config.observation_image_key,
            "wrist_image": data_config.observation_wrist_image_key,
            "state": data_config.observation_state_key,
        }
        if len(data_config.action_sequence_keys) == 1:
            raw_action_key = data_config.action_sequence_keys[0]
            aliases["actions"] = raw_action_key
            aliases["actions_is_pad"] = f"{raw_action_key}_is_pad"
        self._aliases = {
            canonical_key: raw_key for canonical_key, raw_key in aliases.items() if canonical_key != raw_key
        }

    def __getitem__(self, index: SupportsIndex) -> T_co:
        sample = self._dataset[index]
        if not isinstance(sample, dict):
            return sample

        out = dict(sample)
        for canonical_key, raw_key in self._aliases.items():
            if raw_key in sample and canonical_key not in out:
                out[canonical_key] = sample[raw_key]
        return out

    def __len__(self) -> int:
        return len(self._dataset)


class IQLDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset,
        relabeling_instruction_candidate_path,
        shared_instruction_ratio,
        shared_actor_action_only_ratio,
        shared_actor_joint_ratio,
        shared_discriminator_action_ratio,
        shared_discriminator_joint_ratio,
        shared_discriminator_flipped_image_action_ratio,
        shared_discriminator_flipped_image_only_ratio,
        shared_discriminator_flipped_action_only_ratio,
        *,
        data_repo_id: str | None = None,
        action_horizon: int | None = None,
        dynamic_action_chunk_sampling: bool = False,
        discriminator_dynamic_action_relabeling: bool = False,
        proprio_similarity_threshold: float | None = None,
        actor_action_relabeling_similarity_min: float | None = None,
        discriminator_action_relabeling_similarity_min: float | None = None,
        discriminator_action_relabeling_similarity_max: float | None = None,
        allow_dataset_action_relabeling: bool = True,
        instruction_discriminator_uses_proprio: bool = False,
        seed: int = 0,
        shared_loader_role=None,
        shared_discriminator_instruction_similarity_min=None,
        shared_discriminator_instruction_similarity_max=None,
        shared_discriminator_joint_similarity_min=None,
        shared_discriminator_joint_similarity_max=None,
        shared_critic_instruction_similarity_min=None,
        shared_critic_instruction_similarity_max=None,
        shared_actor_instruction_similarity_min=None,
        shared_actor_instruction_similarity_max=None,
        shared_actor_joint_similarity_min=None,
        shared_actor_joint_similarity_max=None,
    ):
        self._dataset = dataset
        self._shared_instruction_ratio = shared_instruction_ratio
        self._shared_actor_action_only_ratio = shared_actor_action_only_ratio
        self._shared_actor_joint_ratio = shared_actor_joint_ratio
        self._shared_discriminator_action_ratio = shared_discriminator_action_ratio
        self._shared_discriminator_joint_ratio = shared_discriminator_joint_ratio
        self._shared_discriminator_flipped_image_action_ratio = shared_discriminator_flipped_image_action_ratio
        self._shared_discriminator_flipped_image_only_ratio = shared_discriminator_flipped_image_only_ratio
        self._shared_discriminator_flipped_action_only_ratio = shared_discriminator_flipped_action_only_ratio
        self._cnt = 0
        self._task_to_task_index = self._build_task_to_task_index_map(dataset)
        self._task_index_to_task = self._build_task_index_to_task_map(dataset)
        if not self._task_index_to_task:
            self._task_index_to_task = {int(v): k for k, v in self._task_to_task_index.items()}
        self._task_to_task_indices = self._build_task_to_task_indices_map(
            task_index_to_task=self._task_index_to_task,
            task_to_task_index=self._task_to_task_index,
        )
        self._steps_to_episode_end = self._precompute_steps_to_episode_end(dataset)
        self._data_repo_id = data_repo_id
        self._action_horizon = int(action_horizon) if action_horizon is not None else None
        self._discriminator_dynamic_action_relabeling = bool(discriminator_dynamic_action_relabeling)
        self.dynamic_action_chunk_sampling = bool(
            dynamic_action_chunk_sampling or self._discriminator_dynamic_action_relabeling
        )
        self.proprio_similarity_threshold = (
            float(proprio_similarity_threshold) if proprio_similarity_threshold is not None else None
        )
        self.actor_action_relabeling_similarity_min = (
            float(actor_action_relabeling_similarity_min)
            if actor_action_relabeling_similarity_min is not None
            else None
        )
        self.discriminator_action_relabeling_similarity_min = (
            float(discriminator_action_relabeling_similarity_min)
            if discriminator_action_relabeling_similarity_min is not None
            else None
        )
        self.discriminator_action_relabeling_similarity_max = (
            float(discriminator_action_relabeling_similarity_max)
            if discriminator_action_relabeling_similarity_max is not None
            else None
        )
        self._allow_dataset_action_relabeling = bool(allow_dataset_action_relabeling)
        self._instruction_discriminator_uses_proprio = bool(instruction_discriminator_uses_proprio)
        self._seed = int(seed)
        self._relabel_rng = None
        self._dynamic_action_chunk_sampler = None
        self._shared_loader_role = shared_loader_role or _make_local_shared_value(_LOADER_ROLE_GENERIC)
        self._shared_similarity_filter_mins = {
            _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_INSTRUCTION: (
                shared_discriminator_instruction_similarity_min
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
            _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_JOINT: (
                shared_discriminator_joint_similarity_min
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
            _SIMILARITY_FILTER_CONTEXT_CRITIC_INSTRUCTION: (
                shared_critic_instruction_similarity_min
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
            _SIMILARITY_FILTER_CONTEXT_ACTOR_INSTRUCTION: (
                shared_actor_instruction_similarity_min
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
            _SIMILARITY_FILTER_CONTEXT_ACTOR_JOINT: (
                shared_actor_joint_similarity_min
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
        }
        self._shared_similarity_filter_maxs = {
            _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_INSTRUCTION: (
                shared_discriminator_instruction_similarity_max
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
            _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_JOINT: (
                shared_discriminator_joint_similarity_max
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
            _SIMILARITY_FILTER_CONTEXT_CRITIC_INSTRUCTION: (
                shared_critic_instruction_similarity_max
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
            _SIMILARITY_FILTER_CONTEXT_ACTOR_INSTRUCTION: (
                shared_actor_instruction_similarity_max
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
            _SIMILARITY_FILTER_CONTEXT_ACTOR_JOINT: (
                shared_actor_joint_similarity_max
                or _make_local_shared_value(_encode_optional_similarity_bound(None))
            ),
        }
        with open(relabeling_instruction_candidate_path, "r", encoding="utf-8") as f:
            self._instruction_candidate = json.load(f)
        if self.proprio_similarity_threshold is not None and not -1.0 <= self.proprio_similarity_threshold <= 1.0:
            raise ValueError(
                "`proprio_similarity_threshold` must be within [-1.0, 1.0], "
                f"got {self.proprio_similarity_threshold}."
            )
        self._validate_similarity_bound(
            "actor_action_relabeling_similarity_min",
            self.actor_action_relabeling_similarity_min,
        )
        self._validate_similarity_filter_range(
            "discriminator_action_relabeling",
            min_value=self.discriminator_action_relabeling_similarity_min,
            max_value=self.discriminator_action_relabeling_similarity_max,
        )
        self.change_loader_role(_LOADER_ROLE_TO_NAME[int(self._shared_loader_role.value)])
        for context in _SIMILARITY_FILTER_CONTEXTS:
            self.change_instruction_relabeling_similarity_filter(
                context,
                min_value=_decode_optional_similarity_bound(self._shared_similarity_filter_mins[context].value),
                max_value=_decode_optional_similarity_bound(self._shared_similarity_filter_maxs[context].value),
            )
        if self.dynamic_action_chunk_sampling:
            if not self._data_repo_id:
                raise ValueError(
                    "`dynamic_action_chunk_sampling=True` requires a non-empty dataset repo id."
                )
            if self._action_horizon is None:
                raise ValueError(
                    "`dynamic_action_chunk_sampling=True` requires a non-empty action horizon."
                )

    @staticmethod
    def _build_task_to_task_index_map(dataset) -> dict[str, int]:
        meta = getattr(dataset, "meta", None)
        mapping = getattr(meta, "task_to_task_index", None)
        if not isinstance(mapping, dict):
            return {}

        out: dict[str, int] = {}
        for task, task_index in mapping.items():
            norm_task = IQLDataset._decode_text(task).strip().lower()
            if not norm_task:
                continue
            try:
                out[norm_task] = int(task_index)
            except Exception:
                continue
        return out

    @staticmethod
    def _build_task_index_to_task_map(dataset) -> dict[int, str]:
        meta = getattr(dataset, "meta", None)
        mapping = getattr(meta, "tasks", None)
        if not isinstance(mapping, dict):
            return {}

        out: dict[int, str] = {}
        for task_index, task in mapping.items():
            norm_task = IQLDataset._decode_text(task).strip().lower()
            if not norm_task:
                continue
            try:
                out[int(task_index)] = norm_task
            except Exception:
                continue
        return out

    @staticmethod
    def _build_task_to_task_indices_map(
        *,
        task_index_to_task: dict[int, str],
        task_to_task_index: dict[str, int],
    ) -> dict[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = {}
        for task_index, task in task_index_to_task.items():
            grouped.setdefault(task, []).append(int(task_index))
        for task, raw_task_index in task_to_task_index.items():
            bucket = grouped.setdefault(task, [])
            task_index = int(raw_task_index)
            if task_index not in bucket:
                bucket.append(task_index)
        return {task: tuple(sorted(indices)) for task, indices in grouped.items()}

    @staticmethod
    def _decode_text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return IQLDataset._decode_text(value.item())
            if value.size == 1:
                return IQLDataset._decode_text(value.reshape(()).item())
        return str(value)

    @staticmethod
    def _coerce_optional_scalar_int(value: object) -> int | None:
        try:
            arr = IQLDataset._to_numpy_array(value)
        except Exception:
            return None
        if arr.size != 1:
            return None
        try:
            return int(arr.reshape(-1)[0])
        except Exception:
            return None

    def _sample_task_index_for_instruction(self, sample: dict, instruction: str) -> int | None:
        if "task" not in sample or "task_index" not in sample:
            return None
        sample_instruction = self._decode_text(sample["task"]).strip().lower()
        if sample_instruction != instruction.strip().lower():
            return None
        return self._coerce_optional_scalar_int(sample["task_index"])

    def _task_indices_for_instruction(
        self,
        sample: dict,
        instruction: str,
        *,
        prefer_sample_task_index: bool,
    ) -> tuple[int, ...]:
        target = instruction.strip().lower()
        indices: list[int] = []
        if prefer_sample_task_index:
            sample_task_index = self._sample_task_index_for_instruction(sample, instruction)
            if sample_task_index is not None:
                indices.append(sample_task_index)
        indices.extend(self._task_to_task_indices.get(target, ()))
        target_task_index = self._task_to_task_index.get(target)
        if target_task_index is not None:
            indices.append(int(target_task_index))

        deduped: list[int] = []
        seen: set[int] = set()
        for raw_task_index in indices:
            task_index = int(raw_task_index)
            if task_index in seen:
                continue
            deduped.append(task_index)
            seen.add(task_index)
        return tuple(deduped)

    @staticmethod
    def _precompute_steps_to_episode_end(dataset) -> np.ndarray:
        num_samples = len(dataset)
        steps_to_episode_end = np.zeros((num_samples,), dtype=np.int32)
        episode_indices = extract_episode_indices(dataset)
        next_episode_index: int | None = None
        steps_from_end = 0

        for index in range(num_samples - 1, -1, -1):
            if episode_indices is None:
                sample = dataset[index]
                episode_index = int(sample["episode_index"])
            else:
                episode_index = int(episode_indices[index])
            if next_episode_index is None or episode_index != next_episode_index:
                steps_from_end = 0
            else:
                steps_from_end += 1
            steps_to_episode_end[index] = steps_from_end
            next_episode_index = episode_index

        return steps_to_episode_end

    @staticmethod
    def _extract_episode_indices(dataset) -> np.ndarray | None:
        return extract_episode_indices(dataset)

    @staticmethod
    def _has_similarity_columns(sample: dict) -> bool:
        has_instructions = sample.get("language_switch_instructions") is not None
        has_task_indices = sample.get("language_switch_task_indices") is not None
        has_similarities = sample.get("language_switch_similarities") is not None

        if has_similarities and not (has_instructions or has_task_indices):
            raise ValueError(
                "`language_switch_similarities` is present but both "
                "`language_switch_instructions` and `language_switch_task_indices` are missing."
            )
        if (has_instructions or has_task_indices) and not has_similarities:
            raise ValueError(
                "`language_switch_instructions`/`language_switch_task_indices` must be provided together with "
                "`language_switch_similarities`; `language_switch_state_similarities` alone is not sufficient."
            )
        return has_similarities

    def _lookup_similarity(self, sample: dict, instruction: str) -> float | None:
        raw_instructions = sample.get("language_switch_instructions")
        raw_task_indices = sample.get("language_switch_task_indices")
        raw_similarities = sample.get("language_switch_similarities")

        if raw_similarities is None:
            return None

        similarities = np.asarray(raw_similarities, dtype=np.float32).reshape(-1)
        if raw_task_indices is not None:
            task_indices = np.asarray(raw_task_indices, dtype=np.int64).reshape(-1)
            if task_indices.shape[0] != similarities.shape[0]:
                raise ValueError(
                    "`language_switch_task_indices` and `language_switch_similarities` must have the same length. "
                    f"Got {task_indices.shape[0]} and {similarities.shape[0]}."
                )
            for target_task_index in self._task_indices_for_instruction(
                sample,
                instruction,
                prefer_sample_task_index=True,
            ):
                # Pad entries (negative task index) are filtered out implicitly
                # by the equality check.
                matches = np.flatnonzero(task_indices == int(target_task_index))
                if matches.size:
                    return float(similarities[int(matches[0])])

        target = instruction.strip().lower()
        if raw_instructions is not None:
            instructions = np.asarray(raw_instructions).reshape(-1)
            if instructions.shape[0] != similarities.shape[0]:
                raise ValueError(
                    "`language_switch_instructions` and `language_switch_similarities` must have the same length. "
                    f"Got {instructions.shape[0]} and {similarities.shape[0]}."
                )
            for idx, candidate_instruction in enumerate(instructions):
                if self._decode_text(candidate_instruction).strip().lower() == target:
                    return float(similarities[idx])
        return None

    def _compute_non_relabeled_weight(self, sample: dict, instruction: str) -> float:
        raw_instructions = sample.get("language_switch_instructions")
        raw_task_indices = sample.get("language_switch_task_indices")
        raw_similarities = sample.get("language_switch_similarities")
        if raw_similarities is None:
            raise ValueError("Missing language switch instructions or similarities in the sample for computing non-relabeled weight.")

        similarities = np.asarray(raw_similarities, dtype=np.float32).reshape(-1)
        target = instruction.strip().lower()

        total_similarity = 0.0
        count = 0
        if raw_instructions is not None:
            instructions = np.asarray(raw_instructions).reshape(-1)
            if instructions.shape[0] != similarities.shape[0]:
                raise ValueError(
                    "`language_switch_instructions` and `language_switch_similarities` must have the same length. "
                    f"Got {instructions.shape[0]} and {similarities.shape[0]}."
                )
            for idx, candidate_instruction in enumerate(instructions):
                if self._decode_text(candidate_instruction).strip().lower() == target:
                    continue
                total_similarity += float(similarities[idx])
                count += 1
        elif raw_task_indices is not None:
            target_task_indices = self._task_indices_for_instruction(
                sample,
                instruction,
                prefer_sample_task_index=True,
            )
            if not target_task_indices:
                raise ValueError(f"Task index for instruction {instruction!r} is not available in dataset metadata.")

            task_indices = np.asarray(raw_task_indices, dtype=np.int64).reshape(-1)
            if task_indices.shape[0] != similarities.shape[0]:
                raise ValueError(
                    "`language_switch_task_indices` and `language_switch_similarities` must have the same length. "
                    f"Got {task_indices.shape[0]} and {similarities.shape[0]}."
                )
            target_task_index_set = set(target_task_indices)
            include_mask = (task_indices >= 0) & ~np.isin(task_indices, list(target_task_index_set))
            count = int(include_mask.sum())
            if count > 0:
                total_similarity = float(similarities[include_mask].sum())
        else:
            raise ValueError("Missing language switch instructions/task indices in the sample for computing non-relabeled weight.")

        if count == 0:
            raise ValueError("No other instructions found for computing non-relabeled weight.")
        return 1.0 - total_similarity / count

    def _get_dynamic_action_chunk_sampler(self):
        if not self.dynamic_action_chunk_sampling:
            return None
        if self._dynamic_action_chunk_sampler is None:
            # Torch workers use `spawn`, so keep the heavy reference bank uninitialized
            # until the first sample is requested inside each worker process.
            assert self._data_repo_id is not None
            assert self._action_horizon is not None
            self._dynamic_action_chunk_sampler = _create_dynamic_action_chunk_sampler(
                repo_id=self._data_repo_id,
                action_horizon=self._action_horizon,
                proprio_similarity_threshold=self.proprio_similarity_threshold,
                seed=self._resolve_dynamic_action_chunk_sampler_seed(),
            )
        return self._dynamic_action_chunk_sampler

    def _resolve_dynamic_action_chunk_sampler_seed(self) -> int:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            return int(worker_info.seed)
        return self._seed

    def _get_relabel_rng(self) -> random.Random:
        if self._relabel_rng is None:
            self._relabel_rng = random.Random(self._resolve_dynamic_action_chunk_sampler_seed())
        return self._relabel_rng

    @staticmethod
    def _validate_similarity_bound(name: str, value: float | None) -> None:
        if value is None:
            return
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"`{name}` must be within [0.0, 1.0], got {value}.")

    def _validate_similarity_filter_range(
        self,
        context: str,
        *,
        min_value: float | None,
        max_value: float | None,
    ) -> None:
        self._validate_similarity_bound(f"{context}_similarity_min", min_value)
        self._validate_similarity_bound(f"{context}_similarity_max", max_value)
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError(
                f"`{context}_similarity_min` must be <= `{context}_similarity_max`, "
                f"got {min_value} > {max_value}."
            )

    def _get_similarity_filter_range(self, context: str) -> tuple[float | None, float | None]:
        if context not in _SIMILARITY_FILTER_CONTEXTS:
            raise ValueError(f"Unknown similarity filter context: {context!r}.")
        min_value = _decode_optional_similarity_bound(self._shared_similarity_filter_mins[context].value)
        max_value = _decode_optional_similarity_bound(self._shared_similarity_filter_maxs[context].value)
        self._validate_similarity_filter_range(context, min_value=min_value, max_value=max_value)
        return min_value, max_value

    def _get_loader_role(self) -> str:
        try:
            role_value = int(self._shared_loader_role.value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid loader role value: {self._shared_loader_role.value!r}.") from exc
        if role_value not in _LOADER_ROLE_TO_NAME:
            raise ValueError(f"Unknown loader role id: {role_value}.")
        return _LOADER_ROLE_TO_NAME[role_value]

    def _resolve_instruction_similarity_filter_context(self, *, relabel_mode: str) -> str | None:
        role = self._get_loader_role()
        if relabel_mode == "joint":
            if role == "discriminator":
                return _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_JOINT
            if role == "actor":
                return _SIMILARITY_FILTER_CONTEXT_ACTOR_JOINT
            return None
        if relabel_mode != "instruction":
            return None

        if role == "discriminator":
            return _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_INSTRUCTION
        if role == "critic":
            return _SIMILARITY_FILTER_CONTEXT_CRITIC_INSTRUCTION
        if role == "actor":
            return _SIMILARITY_FILTER_CONTEXT_ACTOR_INSTRUCTION
        return None

    def _build_similarity_map(self, sample: dict) -> dict[str, float]:
        return self._build_similarity_map_for_field(
            sample,
            similarity_field_name=_LANGUAGE_SWITCH_SIMILARITIES_FIELD,
        )

    def _resolve_similarity_field_name_for_filter_context(self, context: str) -> str:
        if context in {
            _SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_JOINT,
            _SIMILARITY_FILTER_CONTEXT_ACTOR_JOINT,
        }:
            return _LANGUAGE_SWITCH_STATE_SIMILARITIES_FIELD
        return _LANGUAGE_SWITCH_SIMILARITIES_FIELD

    def _build_similarity_map_for_field(
        self,
        sample: dict,
        *,
        similarity_field_name: str,
    ) -> dict[str, float]:
        raw_similarities = sample.get(similarity_field_name)
        raw_instructions = sample.get("language_switch_instructions")
        raw_task_indices = sample.get("language_switch_task_indices")

        if raw_similarities is None:
            raise ValueError(
                "Instruction relabeling similarity filtering requires "
                f"`{similarity_field_name}` in dataset samples."
            )

        similarities = np.asarray(raw_similarities, dtype=np.float32).reshape(-1)
        similarity_map: dict[str, float] = {}
        if raw_instructions is not None:
            instructions = np.asarray(raw_instructions).reshape(-1)
            if instructions.shape[0] != similarities.shape[0]:
                raise ValueError(
                    f"`language_switch_instructions` and `{similarity_field_name}` "
                    f"must have the same length. Got {instructions.shape[0]} and {similarities.shape[0]}."
                )
            for instruction, sim in zip(instructions, similarities, strict=True):
                key = self._decode_text(instruction).strip().lower()
                if key and key not in similarity_map:
                    similarity_map[key] = float(sim)
            return similarity_map

        if raw_task_indices is not None:
            task_indices = np.asarray(raw_task_indices, dtype=np.int64).reshape(-1)
            if task_indices.shape[0] != similarities.shape[0]:
                raise ValueError(
                    f"`language_switch_task_indices` and `{similarity_field_name}` "
                    f"must have the same length. Got {task_indices.shape[0]} and {similarities.shape[0]}."
                )
            for raw_task_index, sim in zip(task_indices, similarities, strict=True):
                task_index = int(raw_task_index)
                if task_index < 0:
                    continue
                task = self._task_index_to_task.get(task_index)
                if task is not None and task not in similarity_map:
                    similarity_map[task] = float(sim)
            return similarity_map

        raise ValueError(
            "Instruction relabeling similarity filtering requires either "
            "`language_switch_instructions` or `language_switch_task_indices` in dataset samples."
        )

    def _filter_candidates_by_similarity_range(
        self,
        sample: dict,
        candidates: list[str],
        *,
        min_value: float | None,
        max_value: float | None,
        similarity_field_name: str,
    ) -> list[str]:
        similarity_map = self._build_similarity_map_for_field(
            sample,
            similarity_field_name=similarity_field_name,
        )
        filtered_candidates: list[str] = []
        for candidate in candidates:
            sim = similarity_map.get(candidate.strip().lower())
            if sim is None:
                continue
            if min_value is not None and sim < min_value:
                continue
            if max_value is not None and sim > max_value:
                continue
            filtered_candidates.append(candidate)
        return filtered_candidates

    def _get_instruction_candidate_pool(
        self,
        sample: dict,
        source_instruction: str,
        *,
        similarity_filter_context: str | None = None,
    ) -> list[str]:
        if source_instruction not in self._instruction_candidate:
            raise KeyError(f"Source instruction {source_instruction!r} not found in instruction candidate JSON.")
        candidate_pool = list(self._instruction_candidate[source_instruction])
        if not candidate_pool:
            raise ValueError("Instruction candidate list is empty for relabeling.")
        if similarity_filter_context is not None:
            min_value, max_value = self._get_similarity_filter_range(similarity_filter_context)
            similarity_field_name = self._resolve_similarity_field_name_for_filter_context(
                similarity_filter_context
            )
            candidate_pool = self._filter_candidates_by_similarity_range(
                sample,
                candidate_pool,
                min_value=min_value,
                max_value=max_value,
                similarity_field_name=similarity_field_name,
            )
        return candidate_pool

    def change_instruction_relabeling_ratio(self, new_ratio: float) -> None:
        self._shared_instruction_ratio.value = new_ratio

    def change_loader_role(self, new_role: str) -> None:
        if new_role not in _LOADER_ROLE_FROM_NAME:
            valid = ", ".join(sorted(_LOADER_ROLE_FROM_NAME))
            raise ValueError(f"Unknown loader role {new_role!r}. Expected one of: {valid}.")
        self._shared_loader_role.value = _LOADER_ROLE_FROM_NAME[new_role]

    def change_instruction_relabeling_similarity_filter(
        self,
        context: str,
        *,
        min_value: float | None,
        max_value: float | None,
    ) -> None:
        if context not in _SIMILARITY_FILTER_CONTEXTS:
            valid = ", ".join(_SIMILARITY_FILTER_CONTEXTS)
            raise ValueError(f"Unknown similarity filter context {context!r}. Expected one of: {valid}.")
        self._validate_similarity_filter_range(context, min_value=min_value, max_value=max_value)
        self._shared_similarity_filter_mins[context].value = _encode_optional_similarity_bound(min_value)
        self._shared_similarity_filter_maxs[context].value = _encode_optional_similarity_bound(max_value)

    def _require_instruction_discriminator_branch_enabled(self, *, ratio_name: str, ratio: float) -> None:
        if ratio > 0.0 and not self._allow_dataset_action_relabeling:
            raise ValueError(
                f"`{ratio_name} > 0` requires `model.split_discriminator_head=True`; "
                "instruction-discriminator-only relabeling/flip modes are unavailable otherwise."
            )

    def _require_instruction_discriminator_flip_supported(self, *, ratio_name: str, ratio: float) -> None:
        self._require_instruction_discriminator_branch_enabled(ratio_name=ratio_name, ratio=ratio)
        if ratio > 0.0 and self._instruction_discriminator_uses_proprio:
            raise ValueError(
                f"`{ratio_name} > 0` is incompatible with proprio-enabled instruction discriminator setups."
            )

    def _require_dataset_action_relabeling_enabled(self, *, ratio_name: str, ratio: float) -> None:
        if ratio > 0.0 and not self._allow_dataset_action_relabeling:
            raise ValueError(
                f"`{ratio_name} > 0` requires `model.split_discriminator_head=True`; "
                "non-split IQL does not support dataset-based action relabeling."
            )

    def _require_dynamic_action_chunk_sampling_enabled(self, *, ratio_name: str, ratio: float) -> None:
        if ratio > 0.0 and not self.dynamic_action_chunk_sampling:
            raise ValueError(
                f"`{ratio_name} > 0` requires `data.dynamic_action_chunk_sampling=True` "
                "for dynamic action relabeling."
            )

    def change_discriminator_action_relabeling_ratio(self, new_ratio: float) -> None:
        self._require_dataset_action_relabeling_enabled(
            ratio_name="action_relabeling_ratio_for_disc_l",
            ratio=float(new_ratio),
        )
        if self._discriminator_dynamic_action_relabeling:
            self._require_dynamic_action_chunk_sampling_enabled(
                ratio_name="action_relabeling_ratio_for_disc_l",
                ratio=float(new_ratio),
            )
        self._shared_discriminator_action_ratio.value = new_ratio

    def change_discriminator_joint_relabeling_ratio(self, new_ratio: float) -> None:
        self._require_dataset_action_relabeling_enabled(
            ratio_name="joint_relabeling_ratio_for_disc_l",
            ratio=float(new_ratio),
        )
        if self._discriminator_dynamic_action_relabeling:
            self._require_dynamic_action_chunk_sampling_enabled(
                ratio_name="joint_relabeling_ratio_for_disc_l",
                ratio=float(new_ratio),
            )
        self._shared_discriminator_joint_ratio.value = new_ratio

    def change_discriminator_flipped_image_action_ratio(self, new_ratio: float) -> None:
        self._require_instruction_discriminator_flip_supported(
            ratio_name="flipped_image_action_ratio_for_disc_l",
            ratio=float(new_ratio),
        )
        self._shared_discriminator_flipped_image_action_ratio.value = new_ratio

    def change_discriminator_flipped_image_only_ratio(self, new_ratio: float) -> None:
        self._require_instruction_discriminator_flip_supported(
            ratio_name="flipped_image_only_ratio_for_disc_l",
            ratio=float(new_ratio),
        )
        self._shared_discriminator_flipped_image_only_ratio.value = new_ratio

    def change_discriminator_flipped_action_only_ratio(self, new_ratio: float) -> None:
        self._require_instruction_discriminator_flip_supported(
            ratio_name="flipped_action_only_ratio_for_disc_l",
            ratio=float(new_ratio),
        )
        self._shared_discriminator_flipped_action_only_ratio.value = new_ratio

    def change_actor_action_only_relabeling_ratio(self, new_ratio: float) -> None:
        self._require_dataset_action_relabeling_enabled(
            ratio_name="actor_action_only_relabeling_ratio",
            ratio=float(new_ratio),
        )
        self._require_dynamic_action_chunk_sampling_enabled(
            ratio_name="actor_action_only_relabeling_ratio",
            ratio=float(new_ratio),
        )
        self._shared_actor_action_only_ratio.value = new_ratio

    def change_actor_joint_relabeling_ratio(self, new_ratio: float) -> None:
        self._require_dataset_action_relabeling_enabled(
            ratio_name="actor_joint_relabeling_ratio",
            ratio=float(new_ratio),
        )
        self._require_dynamic_action_chunk_sampling_enabled(
            ratio_name="actor_joint_relabeling_ratio",
            ratio=float(new_ratio),
        )
        self._shared_actor_joint_ratio.value = new_ratio

    @staticmethod
    def _to_numpy_array(value: object, *, dtype: np.dtype | None = None) -> np.ndarray:
        if isinstance(value, np.ndarray):
            out = value
        elif torch.is_tensor(value):
            out = value.detach().cpu().numpy()
        else:
            out = np.asarray(value)
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def _cast_actions_like(self, reference_actions: object, selected_chunk: np.ndarray) -> object:
        if torch.is_tensor(reference_actions):
            return torch.as_tensor(selected_chunk, dtype=reference_actions.dtype, device=reference_actions.device)
        if isinstance(reference_actions, np.ndarray):
            return selected_chunk.astype(reference_actions.dtype, copy=False)
        return selected_chunk

    def _cast_array_like(self, reference_value: object, value: np.ndarray) -> object:
        if torch.is_tensor(reference_value):
            return torch.as_tensor(value, dtype=reference_value.dtype, device=reference_value.device)
        if isinstance(reference_value, np.ndarray):
            return value.astype(reference_value.dtype, copy=False)
        return value

    def _mirror_images_left_right_like(self, reference_images: object) -> object:
        images = self._to_numpy_array(reference_images)
        mirrored = np.flip(images, axis=_get_image_width_axis(images)).copy()
        return self._cast_array_like(reference_images, mirrored)

    def _mirror_libero_actions_left_right_like(self, reference_actions: object) -> object:
        actions = self._to_numpy_array(reference_actions, dtype=np.float32)
        if actions.shape[-1] < 7:
            raise ValueError(
                "LIBERO left-right action mirroring expects the last action dimension to be at least 7, "
                f"got shape={tuple(actions.shape)}."
            )

        mirrored = actions.copy()
        mirrored[..., :7] = _mirror_libero_actions_left_right(actions[..., :7])
        return self._cast_actions_like(reference_actions, mirrored)

    @staticmethod
    def _get_missing_precomputed_action_chunk_fields(sample: dict) -> list[str]:
        return [
            key
            for key, value in (
                ("language_switch_task_indices", sample.get("language_switch_task_indices")),
                ("language_switch_action_chunks", sample.get("language_switch_action_chunks")),
                ("language_switch_action_chunk_mask", sample.get("language_switch_action_chunk_mask")),
            )
            if value is None
        ]

    def _require_precomputed_action_chunk_fields(
        self,
        sample: dict,
        *,
        missing_fields_hint: str,
    ) -> None:
        missing = self._get_missing_precomputed_action_chunk_fields(sample)
        if missing:
            raise ValueError(f"{missing_fields_hint} requires {', '.join(missing)} in dataset samples.")

    def _sample_precomputed_action_chunk(
        self,
        sample: dict,
        *,
        selected_instruction: str,
        missing_fields_hint: str,
        rng: random.Random,
    ) -> object | None:
        base_actions = self._to_numpy_array(sample["actions"], dtype=np.float32)
        if base_actions.ndim != 2:
            raise ValueError(f"`actions` must have shape (H, Da), got {tuple(base_actions.shape)}.")

        self._require_precomputed_action_chunk_fields(sample, missing_fields_hint=missing_fields_hint)
        raw_task_indices = sample.get("language_switch_task_indices")
        raw_action_chunks = sample.get("language_switch_action_chunks")
        raw_action_chunk_mask = sample.get("language_switch_action_chunk_mask")

        task_indices = self._to_numpy_array(raw_task_indices, dtype=np.int64).reshape(-1)
        action_chunks = self._to_numpy_array(raw_action_chunks, dtype=np.float32)
        action_chunk_mask = self._to_numpy_array(raw_action_chunk_mask, dtype=np.bool_)

        if action_chunks.ndim != 4:
            raise ValueError(
                "`language_switch_action_chunks` must have shape (K, N, H, Da), "
                f"got {tuple(action_chunks.shape)}."
            )
        if action_chunk_mask.ndim != 2:
            raise ValueError(
                "`language_switch_action_chunk_mask` must have shape (K, N), "
                f"got {tuple(action_chunk_mask.shape)}."
            )
        if task_indices.shape[0] != action_chunks.shape[0] or action_chunk_mask.shape[0] != action_chunks.shape[0]:
            raise ValueError(
                "Mismatched K dimension across language-switch action chunk fields: "
                f"task_indices={task_indices.shape}, chunks={action_chunks.shape}, mask={action_chunk_mask.shape}."
            )
        if action_chunk_mask.shape[1] != action_chunks.shape[1]:
            raise ValueError(
                "Mismatched N dimension between `language_switch_action_chunks` and "
                f"`language_switch_action_chunk_mask`: chunks={action_chunks.shape}, mask={action_chunk_mask.shape}."
            )
        if action_chunks.shape[2:] != base_actions.shape:
            raise ValueError(
                "Action chunk candidate shape is incompatible with sample `actions`: "
                f"chunks per neighbor={tuple(action_chunks.shape[2:])}, actions={tuple(base_actions.shape)}."
            )

        target_task_indices = self._task_indices_for_instruction(
            sample,
            selected_instruction,
            prefer_sample_task_index=False,
        )
        if not target_task_indices:
            raise ValueError(
                f"Task index for relabeled instruction {selected_instruction!r} is not available in dataset metadata."
            )

        target_positions = np.concatenate(
            [np.nonzero(task_indices == int(task_index))[0] for task_index in target_task_indices]
        )
        if target_positions.size == 0:
            target_task_indices_str = ", ".join(str(int(task_index)) for task_index in target_task_indices)
            raise ValueError(
                "No action chunk task slot matches relabeled instruction "
                f"{selected_instruction!r} (task_indices=[{target_task_indices_str}])."
            )

        slot_idx = int(target_positions[0])
        valid_neighbors = np.nonzero(action_chunk_mask[slot_idx])[0]
        if valid_neighbors.size == 0:
            return None

        neighbor_idx = int(rng.choice(valid_neighbors.tolist()))
        selected_chunk = action_chunks[slot_idx, neighbor_idx]
        return self._cast_actions_like(sample["actions"], selected_chunk)

    def _extract_dynamic_action_chunk_sampling_proprio(self, sample: dict) -> np.ndarray:
        state = self._to_numpy_array(sample["state"], dtype=np.float32)
        if state.ndim == 2:
            if state.shape[0] <= 0:
                raise ValueError("`state` must contain at least one observation frame.")
            return state[0]
        if state.ndim == 1:
            return state
        raise ValueError(f"`state` must have shape (2, Ds) or (Ds,), got {tuple(state.shape)}.")

    def _maybe_relabel_actions_with_dynamic_sampler(
        self,
        sample: dict,
        *,
        ratio_name: str,
        selected_instruction: str | None,
        action_similarity_min: float | None = None,
        action_similarity_max: float | None = None,
        sampling_strategy: str = "weighted",
        use_proprio_threshold: bool = True,
        fallback_to_nearest_if_proprio_empty: bool = False,
    ) -> tuple[object, bool]:
        self._require_dataset_action_relabeling_enabled(
            ratio_name=ratio_name,
            ratio=1.0,
        )
        self._require_dynamic_action_chunk_sampling_enabled(
            ratio_name=ratio_name,
            ratio=1.0,
        )

        base_actions = self._to_numpy_array(sample["actions"], dtype=np.float32)
        if base_actions.ndim != 2:
            raise ValueError(f"`actions` must have shape (H, Da), got {tuple(base_actions.shape)}.")

        sampler = self._get_dynamic_action_chunk_sampler()
        proprio = self._extract_dynamic_action_chunk_sampling_proprio(sample)
        sampled_chunks, sampled_mask = sampler.sample(
            instruction=selected_instruction,
            proprio=proprio,
            action_chunk_size=int(base_actions.shape[0]),
            action_dim=int(base_actions.shape[1]),
            query_action_chunk=(
                base_actions
                if action_similarity_min is not None or action_similarity_max is not None
                else None
            ),
            action_similarity_min=action_similarity_min,
            action_similarity_max=action_similarity_max,
            sampling_strategy=sampling_strategy,
            use_proprio_threshold=use_proprio_threshold,
            fallback_to_nearest_if_proprio_empty=fallback_to_nearest_if_proprio_empty,
        )
        sampled_chunks = self._to_numpy_array(sampled_chunks, dtype=np.float32)
        sampled_mask = self._to_numpy_array(sampled_mask, dtype=np.bool_).reshape(-1)
        valid_neighbors = np.nonzero(sampled_mask)[0]
        if valid_neighbors.size == 0:
            return sample["actions"], False
        selected_chunk = sampled_chunks[int(valid_neighbors[0])]
        return self._cast_actions_like(sample["actions"], selected_chunk), True

    def _sample_actor_relabel_mode(
        self,
        *,
        instruction_ratio: float,
        action_only_ratio: float,
        joint_ratio: float,
        rng: random.Random,
    ) -> str:
        self._require_dataset_action_relabeling_enabled(
            ratio_name="actor_action_only_relabeling_ratio",
            ratio=action_only_ratio,
        )
        self._require_dynamic_action_chunk_sampling_enabled(
            ratio_name="actor_action_only_relabeling_ratio",
            ratio=action_only_ratio,
        )
        self._require_dataset_action_relabeling_enabled(
            ratio_name="actor_joint_relabeling_ratio",
            ratio=joint_ratio,
        )
        self._require_dynamic_action_chunk_sampling_enabled(
            ratio_name="actor_joint_relabeling_ratio",
            ratio=joint_ratio,
        )
        if joint_ratio > 0.0 and self.proprio_similarity_threshold is None:
            raise ValueError(
                "`actor_joint_relabeling_ratio > 0` requires "
                "`data.proprio_similarity_threshold` to be set."
            )
        if (action_only_ratio > 0.0 or joint_ratio > 0.0) and self.actor_action_relabeling_similarity_min is None:
            raise ValueError(
                "`actor_action_only_relabeling_ratio > 0` or `actor_joint_relabeling_ratio > 0` requires "
                "`data.actor_action_relabeling_similarity_min` to be set."
            )
        total_ratio = instruction_ratio + action_only_ratio + joint_ratio
        if total_ratio > 1.0:
            raise ValueError(
                "`instruction_relabeling_ratio_for_actor + actor_action_only_relabeling_ratio + "
                "`actor_joint_relabeling_ratio` must be <= 1.0, "
                f"got {total_ratio}."
            )
        for ratio_name, ratio in (
            ("instruction_relabeling_ratio_for_actor", instruction_ratio),
            ("actor_action_only_relabeling_ratio", action_only_ratio),
            ("actor_joint_relabeling_ratio", joint_ratio),
        ):
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError(f"`{ratio_name}` must be in [0, 1], got {ratio}.")

        sample_ratio = rng.random()
        if sample_ratio < instruction_ratio:
            return "instruction"
        if sample_ratio < instruction_ratio + action_only_ratio:
            return "action"
        if sample_ratio < instruction_ratio + action_only_ratio + joint_ratio:
            return "joint"
        return "original"

    def _sample_discriminator_relabel_mode(
        self,
        *,
        instruction_ratio: float,
        action_ratio: float,
        joint_ratio: float,
        flipped_image_action_ratio: float,
        flipped_image_only_ratio: float,
        flipped_action_only_ratio: float,
        rng: random.Random,
    ) -> str:
        self._require_dataset_action_relabeling_enabled(
            ratio_name="action_relabeling_ratio_for_disc_l",
            ratio=action_ratio,
        )
        self._require_dataset_action_relabeling_enabled(
            ratio_name="joint_relabeling_ratio_for_disc_l",
            ratio=joint_ratio,
        )
        if self._discriminator_dynamic_action_relabeling:
            self._require_dynamic_action_chunk_sampling_enabled(
                ratio_name="action_relabeling_ratio_for_disc_l",
                ratio=action_ratio,
            )
            self._require_dynamic_action_chunk_sampling_enabled(
                ratio_name="joint_relabeling_ratio_for_disc_l",
                ratio=joint_ratio,
            )
            if (
                action_ratio > 0.0
                and (
                    self.discriminator_action_relabeling_similarity_min is None
                    or self.discriminator_action_relabeling_similarity_max is None
                )
            ):
                raise ValueError(
                    "`action_relabeling_ratio_for_disc_l > 0` with "
                    "`data.discriminator_dynamic_action_relabeling=True` requires both "
                    "`data.discriminator_action_relabeling_similarity_min` and "
                    "`data.discriminator_action_relabeling_similarity_max`."
                )
            if joint_ratio > 0.0 and self.proprio_similarity_threshold is None:
                raise ValueError(
                    "`joint_relabeling_ratio_for_disc_l > 0` with "
                    "`data.discriminator_dynamic_action_relabeling=True` requires "
                    "`data.proprio_similarity_threshold` to be set."
                )
        for ratio_name, ratio in (
            ("flipped_image_action_ratio_for_disc_l", flipped_image_action_ratio),
            ("flipped_image_only_ratio_for_disc_l", flipped_image_only_ratio),
            ("flipped_action_only_ratio_for_disc_l", flipped_action_only_ratio),
        ):
            self._require_instruction_discriminator_flip_supported(
                ratio_name=ratio_name,
                ratio=ratio,
            )
        total_ratio = (
            instruction_ratio
            + action_ratio
            + joint_ratio
            + flipped_image_action_ratio
            + flipped_image_only_ratio
            + flipped_action_only_ratio
        )
        if total_ratio > 1.0:
            raise ValueError(
                "`instruction_relabeling_ratio + discriminator_action_relabeling_ratio + "
                "`discriminator_joint_relabeling_ratio + flipped_image_action_ratio_for_disc_l + "
                "`flipped_image_only_ratio_for_disc_l + flipped_action_only_ratio_for_disc_l` must be <= 1.0, "
                f"got {total_ratio}."
            )
        for ratio_name, ratio in (
            ("instruction_relabeling_ratio", instruction_ratio),
            ("discriminator_action_relabeling_ratio", action_ratio),
            ("discriminator_joint_relabeling_ratio", joint_ratio),
            ("flipped_image_action_ratio_for_disc_l", flipped_image_action_ratio),
            ("flipped_image_only_ratio_for_disc_l", flipped_image_only_ratio),
            ("flipped_action_only_ratio_for_disc_l", flipped_action_only_ratio),
        ):
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError(f"`{ratio_name}` must be in [0, 1], got {ratio}.")

        sample_ratio = rng.random()
        if sample_ratio < instruction_ratio:
            return "instruction"
        if sample_ratio < instruction_ratio + action_ratio:
            return "action"
        if sample_ratio < instruction_ratio + action_ratio + joint_ratio:
            return "joint"
        if sample_ratio < (
            instruction_ratio
            + action_ratio
            + joint_ratio
            + flipped_image_action_ratio
        ):
            return "flipped_image_action"
        if sample_ratio < (
            instruction_ratio
            + action_ratio
            + joint_ratio
            + flipped_image_action_ratio
            + flipped_image_only_ratio
        ):
            return "flipped_image_only"
        if sample_ratio < (
            instruction_ratio
            + action_ratio
            + joint_ratio
            + flipped_image_action_ratio
            + flipped_image_only_ratio
            + flipped_action_only_ratio
        ):
            return "flipped_action_only"
        return "original"
    
    def __getitem__(self, index: SupportsIndex) -> T_co:
        rng = self._get_relabel_rng()
        sample = self._dataset[index]
        # Avoid `self._dataset[index+1]` here: it forces an extra full sample
        # load (including image decoding) just to read `episode_index`. The
        # precomputed `_steps_to_episode_end` already encodes the same signal
        # — a value of 0 means index is the last frame of its episode (or the
        # final dataset sample), which is the exact `done` definition.
        done = bool(int(self._steps_to_episode_end[int(index)]) == 0)

        relabeled_instruction = False
        relabeled_action = False
        left_right_flipped_image = False
        left_right_flipped_action = False
        source_instruction = self._decode_text(sample["task"])
        lang = source_instruction
        instruction_ratio = float(self._shared_instruction_ratio.value)
        discriminator_action_ratio = float(self._shared_discriminator_action_ratio.value)
        discriminator_joint_ratio = float(self._shared_discriminator_joint_ratio.value)
        discriminator_flipped_image_action_ratio = float(
            self._shared_discriminator_flipped_image_action_ratio.value
        )
        discriminator_flipped_image_only_ratio = float(
            self._shared_discriminator_flipped_image_only_ratio.value
        )
        discriminator_flipped_action_only_ratio = float(
            self._shared_discriminator_flipped_action_only_ratio.value
        )
        actor_action_only_ratio = float(self._shared_actor_action_only_ratio.value)
        actor_joint_ratio = float(self._shared_actor_joint_ratio.value)
        loader_role = self._get_loader_role()
        discriminator_actions = sample["actions"]
        actions_for_actor = sample["actions"]
        actor_action_chunk_relabeled = False
        current_images = sample["image"]
        wrist_images = sample["wrist_image"]

        has_similarity_columns = self._has_similarity_columns(sample)
        if loader_role == "actor":
            if (instruction_ratio > 0.0 or actor_joint_ratio > 0.0) and not has_similarity_columns:
                raise ValueError(
                    "Actor instruction/joint relabeling is enabled but language-switch similarity columns are missing. "
                    "Augment the dataset with language-switch similarities or set "
                    "`instruction_relabeling_ratio_for_actor=0` and "
                    "`actor_joint_relabeling_ratio=0`."
                )

            actor_relabel_mode = self._sample_actor_relabel_mode(
                instruction_ratio=instruction_ratio,
                action_only_ratio=actor_action_only_ratio,
                joint_ratio=actor_joint_ratio,
                rng=rng,
            )
            if actor_relabel_mode in {"instruction", "joint"}:
                candidate_pool = self._get_instruction_candidate_pool(
                    sample,
                    source_instruction,
                    similarity_filter_context=self._resolve_instruction_similarity_filter_context(
                        relabel_mode=actor_relabel_mode
                    ),
                )
                if candidate_pool:
                    selected_candidate = rng.choice(candidate_pool)
                    if not isinstance(selected_candidate, str):
                        raise ValueError(
                            f"`selected_candidate` must be `str`, got {type(selected_candidate)}: {selected_candidate!r}"
                        )

                    if actor_relabel_mode == "instruction":
                        lang = selected_candidate
                        relabeled_instruction = True
                    else:
                        relabeled_actions, actor_action_chunk_relabeled = (
                            self._maybe_relabel_actions_with_dynamic_sampler(
                                sample,
                                selected_instruction=selected_candidate,
                                ratio_name="actor_joint_relabeling_ratio",
                                action_similarity_min=self.actor_action_relabeling_similarity_min,
                            )
                        )
                        if actor_action_chunk_relabeled:
                            actions_for_actor = relabeled_actions
                            lang = selected_candidate
                            relabeled_instruction = True
            elif actor_relabel_mode == "action":
                actions_for_actor, actor_action_chunk_relabeled = self._maybe_relabel_actions_with_dynamic_sampler(
                    sample,
                    selected_instruction=None,
                    ratio_name="actor_action_only_relabeling_ratio",
                    action_similarity_min=self.actor_action_relabeling_similarity_min,
                    sampling_strategy="uniform",
                    use_proprio_threshold=False,
                )
        else:
            if (
                instruction_ratio > 0.0 or discriminator_action_ratio > 0.0 or discriminator_joint_ratio > 0.0
            ) and not has_similarity_columns:
                raise ValueError(
                    "Discriminator relabeling is enabled but language-switch similarity columns are missing. "
                    "Augment the dataset with language-switch similarities or set "
                    "`instruction_relabeling_ratio=0`, `action_relabeling_ratio_for_disc_l=0`, and "
                    "`joint_relabeling_ratio_for_disc_l=0`."
                )
            if (
                not self._discriminator_dynamic_action_relabeling
                and (discriminator_action_ratio > 0.0 or discriminator_joint_ratio > 0.0)
            ):
                self._require_precomputed_action_chunk_fields(
                    sample,
                    missing_fields_hint=(
                        "`action_relabeling_ratio_for_disc_l > 0` or "
                        "`joint_relabeling_ratio_for_disc_l > 0`"
                    ),
                )

            relabel_mode = self._sample_discriminator_relabel_mode(
                instruction_ratio=instruction_ratio,
                action_ratio=discriminator_action_ratio,
                joint_ratio=discriminator_joint_ratio,
                flipped_image_action_ratio=discriminator_flipped_image_action_ratio,
                flipped_image_only_ratio=discriminator_flipped_image_only_ratio,
                flipped_action_only_ratio=discriminator_flipped_action_only_ratio,
                rng=rng,
            )
            if relabel_mode == "action" and self._discriminator_dynamic_action_relabeling:
                discriminator_actions, relabeled_action = self._maybe_relabel_actions_with_dynamic_sampler(
                    sample,
                    selected_instruction=None,
                    ratio_name="action_relabeling_ratio_for_disc_l",
                    action_similarity_min=self.discriminator_action_relabeling_similarity_min,
                    action_similarity_max=self.discriminator_action_relabeling_similarity_max,
                    sampling_strategy="uniform",
                    use_proprio_threshold=False,
                )
            elif relabel_mode in {"instruction", "action", "joint"}:
                candidate_pool = self._get_instruction_candidate_pool(
                    sample,
                    source_instruction,
                    similarity_filter_context=self._resolve_instruction_similarity_filter_context(
                        relabel_mode=relabel_mode
                    ),
                )
                if candidate_pool:
                    selected_candidate = rng.choice(candidate_pool)
                    if not isinstance(selected_candidate, str):
                        raise ValueError(
                            f"`selected_candidate` must be `str`, got {type(selected_candidate)}: {selected_candidate!r}"
                        )

                    if relabel_mode == "instruction":
                        lang = selected_candidate
                        relabeled_instruction = True
                    elif relabel_mode == "joint" and self._discriminator_dynamic_action_relabeling:
                        selected_chunk, relabeled_action = self._maybe_relabel_actions_with_dynamic_sampler(
                            sample,
                            selected_instruction=selected_candidate,
                            ratio_name="joint_relabeling_ratio_for_disc_l",
                            sampling_strategy="uniform",
                        )
                        if relabeled_action:
                            discriminator_actions = selected_chunk
                            lang = selected_candidate
                            relabeled_instruction = True
                    else:
                        selected_chunk = self._sample_precomputed_action_chunk(
                            sample,
                            selected_instruction=selected_candidate,
                            missing_fields_hint=(
                                "`action_relabeling_ratio_for_disc_l > 0` or "
                                "`joint_relabeling_ratio_for_disc_l > 0`"
                            ),
                            rng=rng,
                        )
                        if selected_chunk is not None:
                            discriminator_actions = selected_chunk
                            relabeled_action = True
                            if relabel_mode == "joint":
                                lang = selected_candidate
                                relabeled_instruction = True
            elif relabel_mode == "flipped_image_action":
                current_images = self._mirror_images_left_right_like(sample["image"])
                wrist_images = self._mirror_images_left_right_like(sample["wrist_image"])
                discriminator_actions = self._mirror_libero_actions_left_right_like(sample["actions"])
                left_right_flipped_image = True
                left_right_flipped_action = True
            elif relabel_mode == "flipped_image_only":
                current_images = self._mirror_images_left_right_like(sample["image"])
                wrist_images = self._mirror_images_left_right_like(sample["wrist_image"])
                left_right_flipped_image = True
            elif relabel_mode == "flipped_action_only":
                discriminator_actions = self._mirror_libero_actions_left_right_like(sample["actions"])
                left_right_flipped_action = True

            actions_for_actor = discriminator_actions

        if has_similarity_columns:
            maybe_similarity = self._lookup_similarity(sample, lang)
            if maybe_similarity is None:
                if relabeled_instruction:
                    raise ValueError(
                        "Missing similarity score for relabeled instruction. "
                        "Include language switch instructions/task indices and similarities in the sample."
                    )
                raise ValueError(
                    "Missing similarity score for original instruction. "
                    "Include language switch instructions/task indices and similarities in the sample."
                )

            relabeled_instruction_similarity = maybe_similarity

            if relabeled_instruction:
                relabeled_instruction_similarity_weight = 1.0 - relabeled_instruction_similarity
            else:
                relabeled_instruction_similarity_weight = self._compute_non_relabeled_weight(sample, lang)
        else:
            # Backward-compatible fallback for legacy datasets without language-switch similarity columns.
            relabeled_instruction_similarity = 1.0
            relabeled_instruction_similarity_weight = 1.0

        relabeled_instruction_similarity = float(np.clip(relabeled_instruction_similarity, 0.0, 1.0))
        relabeled_instruction_similarity_weight = float(np.clip(relabeled_instruction_similarity_weight, 0.0, 1.0))

        self._cnt += 1

        out = {
            "image": current_images[0],
            "wrist_image": wrist_images[0],
            "state": sample["state"][0],
            "actions": actions_for_actor,
            "next_obs": {
                "image": current_images[1],
                "wrist_image": wrist_images[1],
                "state": sample["state"][1],
            },
            "timestamp": sample["timestamp"],
            "frame_index": sample["frame_index"],
            "episode_index": sample["episode_index"],
            "index": sample["index"],
            "task_index": sample["task_index"],
            "actions_is_pad": sample["actions_is_pad"],
            "task": lang,
            "done": done,
            "steps_to_episode_end": np.int32(self._steps_to_episode_end[index]),
            "relabeled_instruction": np.bool_(relabeled_instruction),
            "relabeled_action": np.bool_(relabeled_action),
            "left_right_flipped_image": np.bool_(left_right_flipped_image),
            "left_right_flipped_action": np.bool_(left_right_flipped_action),
            "relabeled_instruction_similarity": np.float32(relabeled_instruction_similarity),
            "relabeled_instruction_similarity_weight": np.float32(relabeled_instruction_similarity_weight),
            "actor_action_chunk_relabeled": np.bool_(actor_action_chunk_relabeled),
        }
        return out

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _load_norm_stats_from_lerobot_metadata(
    data_config: _config.DataConfig,
) -> dict[str, _normalize.NormStats] | None:
    if data_config.repo_id is None or len(data_config.action_sequence_keys) != 1:
        return None

    try:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    except (FileNotFoundError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:  # pragma: no cover - fallback for offline/no-cache setups.
        logging.info("Unable to load LeRobot metadata stats for %s: %s", data_config.repo_id, exc)
        return None

    state_key = data_config.observation_state_key
    action_key = data_config.action_sequence_keys[0]
    state_stats = dataset_meta.stats.get(state_key)
    action_stats = dataset_meta.stats.get(action_key)
    if state_stats is None or action_stats is None:
        logging.info(
            "Metadata stats for %s do not contain both %s and %s; skipping metadata norm-stats fallback.",
            data_config.repo_id,
            state_key,
            action_key,
        )
        return None

    def _convert(stats: dict[str, np.ndarray]) -> _normalize.NormStats:
        # LeRobot metadata does not store quantiles; use min/max as a conservative fallback so
        # pi0.5 quantile normalization remains available without a separate preprocessing step.
        return _normalize.NormStats(
            mean=np.asarray(stats["mean"]),
            std=np.asarray(stats["std"]),
            q01=np.asarray(stats["min"]),
            q99=np.asarray(stats["max"]),
        )

    logging.info(
        "Loaded normalization stats for %s from LeRobot metadata using %s and %s.",
        data_config.repo_id,
        state_key,
        action_key,
    )
    return {
        "state": _convert(state_stats),
        "actions": _convert(action_stats),
    }


def _populate_missing_norm_stats(
    data_config: _config.DataConfig,
) -> _config.DataConfig:
    if data_config.repo_id in (None, "fake") or data_config.norm_stats is not None:
        return data_config

    fallback_stats = _load_norm_stats_from_lerobot_metadata(data_config)
    if fallback_stats is None:
        return data_config
    return typing.cast(_config.DataConfig, dataclasses.replace(data_config, norm_stats=fallback_stats))


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    seed: int = 0,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    relabeling_candidate_path = str(data_config.relabeling_instruction_candidate_path or "").strip()
    requires_iql_columns = bool(relabeling_candidate_path) or getattr(model_config, "use_iql", False)
    delta_timestamps = {}
    if requires_iql_columns:
        delta_timestamps.update(
            {
                data_config.observation_image_key: [0.0, 1.0 / dataset_meta.fps],
                data_config.observation_wrist_image_key: [0.0, 1.0 / dataset_meta.fps],
                data_config.observation_state_key: [0.0, 1.0 / dataset_meta.fps],
            }
        )

    for key in data_config.action_sequence_keys:
        delta_timestamps[key] = [t / dataset_meta.fps for t in range(action_horizon)]
    
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps=delta_timestamps,
        video_backend=data_config.video_backend,
    )
    _drop_unused_hf_dataset_columns(dataset, data_config=data_config)
    dataset = CanonicalLeRobotDataset(dataset, data_config)
    if relabeling_candidate_path:
        similarity_filters = resolve_instruction_relabeling_similarity_filters(data_config)
        initial_loader_role = infer_iql_loader_role(data_config, model_config)
        manager = mp.Manager()
        shared_instruction_ratio = manager.Value("d", 0.0)
        shared_actor_action_only_ratio = manager.Value("d", 0.0)
        shared_actor_joint_ratio = manager.Value("d", 0.0)
        shared_discriminator_action_ratio = manager.Value("d", 0.0)
        shared_discriminator_joint_ratio = manager.Value("d", 0.0)
        shared_discriminator_flipped_image_action_ratio = manager.Value("d", 0.0)
        shared_discriminator_flipped_image_only_ratio = manager.Value("d", 0.0)
        shared_discriminator_flipped_action_only_ratio = manager.Value("d", 0.0)
        shared_loader_role = manager.Value("i", _LOADER_ROLE_FROM_NAME[initial_loader_role])
        shared_discriminator_instruction_similarity_min = manager.Value(
            "d",
            _encode_optional_similarity_bound(
                similarity_filters[_SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_INSTRUCTION][0]
            ),
        )
        shared_discriminator_instruction_similarity_max = manager.Value(
            "d",
            _encode_optional_similarity_bound(
                similarity_filters[_SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_INSTRUCTION][1]
            ),
        )
        shared_discriminator_joint_similarity_min = manager.Value(
            "d",
            _encode_optional_similarity_bound(
                similarity_filters[_SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_JOINT][0]
            ),
        )
        shared_discriminator_joint_similarity_max = manager.Value(
            "d",
            _encode_optional_similarity_bound(
                similarity_filters[_SIMILARITY_FILTER_CONTEXT_DISCRIMINATOR_JOINT][1]
            ),
        )
        shared_critic_instruction_similarity_min = manager.Value(
            "d",
            _encode_optional_similarity_bound(similarity_filters[_SIMILARITY_FILTER_CONTEXT_CRITIC_INSTRUCTION][0]),
        )
        shared_critic_instruction_similarity_max = manager.Value(
            "d",
            _encode_optional_similarity_bound(similarity_filters[_SIMILARITY_FILTER_CONTEXT_CRITIC_INSTRUCTION][1]),
        )
        shared_actor_instruction_similarity_min = manager.Value(
            "d",
            _encode_optional_similarity_bound(similarity_filters[_SIMILARITY_FILTER_CONTEXT_ACTOR_INSTRUCTION][0]),
        )
        shared_actor_instruction_similarity_max = manager.Value(
            "d",
            _encode_optional_similarity_bound(similarity_filters[_SIMILARITY_FILTER_CONTEXT_ACTOR_INSTRUCTION][1]),
        )
        shared_actor_joint_similarity_min = manager.Value(
            "d",
            _encode_optional_similarity_bound(similarity_filters[_SIMILARITY_FILTER_CONTEXT_ACTOR_JOINT][0]),
        )
        shared_actor_joint_similarity_max = manager.Value(
            "d",
            _encode_optional_similarity_bound(similarity_filters[_SIMILARITY_FILTER_CONTEXT_ACTOR_JOINT][1]),
        )
        dataset = IQLDataset(
            dataset,
            relabeling_candidate_path,
            shared_instruction_ratio=shared_instruction_ratio,
            shared_actor_action_only_ratio=shared_actor_action_only_ratio,
            shared_actor_joint_ratio=shared_actor_joint_ratio,
            shared_discriminator_action_ratio=shared_discriminator_action_ratio,
            shared_discriminator_joint_ratio=shared_discriminator_joint_ratio,
            shared_discriminator_flipped_image_action_ratio=shared_discriminator_flipped_image_action_ratio,
            shared_discriminator_flipped_image_only_ratio=shared_discriminator_flipped_image_only_ratio,
            shared_discriminator_flipped_action_only_ratio=shared_discriminator_flipped_action_only_ratio,
            data_repo_id=data_config.repo_id,
            action_horizon=action_horizon,
            dynamic_action_chunk_sampling=data_config.dynamic_action_chunk_sampling,
            discriminator_dynamic_action_relabeling=data_config.discriminator_dynamic_action_relabeling,
            proprio_similarity_threshold=data_config.proprio_similarity_threshold,
            actor_action_relabeling_similarity_min=data_config.actor_action_relabeling_similarity_min,
            discriminator_action_relabeling_similarity_min=(
                data_config.discriminator_action_relabeling_similarity_min
            ),
            discriminator_action_relabeling_similarity_max=(
                data_config.discriminator_action_relabeling_similarity_max
            ),
            allow_dataset_action_relabeling=bool(getattr(model_config, "split_discriminator_head", False)),
            instruction_discriminator_uses_proprio=_instruction_discriminator_uses_proprio(model_config),
            seed=seed,
            shared_loader_role=shared_loader_role,
            shared_discriminator_instruction_similarity_min=shared_discriminator_instruction_similarity_min,
            shared_discriminator_instruction_similarity_max=shared_discriminator_instruction_similarity_max,
            shared_discriminator_joint_similarity_min=shared_discriminator_joint_similarity_min,
            shared_discriminator_joint_similarity_max=shared_discriminator_joint_similarity_max,
            shared_critic_instruction_similarity_min=shared_critic_instruction_similarity_min,
            shared_critic_instruction_similarity_max=shared_critic_instruction_similarity_max,
            shared_actor_instruction_similarity_min=shared_actor_instruction_similarity_min,
            shared_actor_instruction_similarity_max=shared_actor_instruction_similarity_max,
            shared_actor_joint_similarity_min=shared_actor_joint_similarity_min,
            shared_actor_joint_similarity_max=shared_actor_joint_similarity_max,
        )
    elif getattr(model_config, "use_iql", False):
        raise ValueError(
            "`model.use_iql=True` requires `data.relabeling_instruction_candidate_path` to be set."
        )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset,
            [
                _transforms.PromptFromLeRobotTask(
                    dataset_meta.tasks,
                    strip_task_name_prefix=data_config.strip_task_name_prefix_from_prompt,
                )
            ],
        )

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                key_norm_modes=_config.action_norm_key_modes(data_config),
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                key_norm_modes=_config.action_norm_key_modes(data_config),
            ),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    if not skip_norm_stats:
        data_config = _populate_missing_norm_stats(data_config)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = mp.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
