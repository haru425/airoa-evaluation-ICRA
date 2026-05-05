from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from lerobot.common.constants import HF_LEROBOT_HOME

try:
    import fcntl
except ImportError:  # pragma: no cover - not expected on Linux, but keeps module portable.
    fcntl = None


_DYNAMIC_ACTION_CHUNK_CACHE_ROOT = Path(
    os.environ.get(
        "OPENPI_DYNAMIC_ACTION_CHUNK_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "openpi_dynamic_action_chunk_sampler"),
    )
)
_UNIFORM_ACTION_FILTER_REJECTION_ATTEMPTS = 4096
_UNIFORM_ACTION_FILTER_REJECTION_BATCH_SIZE = 1024
_REFERENCE_BANKS_CACHE: dict[str, dict[str, "_RefBank"]] = {}
_FEATURE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "state": ("state", "observation.state"),
    "actions": ("actions", "action", "action.relative"),
}


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    task: str
    length: int
    source_path: Path


@dataclass(frozen=True)
class _RefBank:
    proprio: np.ndarray
    action_chunks: np.ndarray
    action_chunks_norm: np.ndarray


@dataclass(frozen=True)
class _CandidatePool:
    bank: _RefBank
    indices: np.ndarray
    similarities: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _resolve_feature_column(info: dict[str, Any], canonical_name: str) -> str:
    features = info.get("features", {})
    if not isinstance(features, dict):
        features = {}
    if not features:
        return _FEATURE_COLUMN_ALIASES.get(canonical_name, (canonical_name,))[0]
    for candidate in _FEATURE_COLUMN_ALIASES.get(canonical_name, (canonical_name,)):
        if candidate in features:
            return candidate
    available = ", ".join(sorted(str(name) for name in features))
    raise KeyError(
        f"Failed to resolve dataset column for {canonical_name!r}. "
        f"Tried aliases={_FEATURE_COLUMN_ALIASES.get(canonical_name, (canonical_name,))}, "
        f"available_features=[{available}]"
    )


def _resolve_stats_entry(stats: dict[str, Any], *, canonical_name: str, resolved_column: str) -> dict[str, Any]:
    for candidate in dict.fromkeys((resolved_column, *_FEATURE_COLUMN_ALIASES.get(canonical_name, (canonical_name,)))):
        field_stats = stats.get(candidate)
        if field_stats is not None:
            return field_stats
    raise KeyError(
        f"Failed to resolve stats entry for {canonical_name!r}. "
        f"Tried aliases={(resolved_column, *_FEATURE_COLUMN_ALIASES.get(canonical_name, (canonical_name,)))}."
    )


def _convert_metadata_stats(stats: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for field_name, field_stats in stats.items():
        if not isinstance(field_stats, dict):
            continue
        out = dict(field_stats)
        if "q01" not in out and out.get("min") is not None:
            out["q01"] = out["min"]
        if "q99" not in out and out.get("max") is not None:
            out["q99"] = out["max"]
        converted[str(field_name)] = out
    return converted


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(denom, eps)).astype(np.float32, copy=False)


def _normalize_q01_q99(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = 2.0 * (x - q01) / (q99 - q01 + 1e-8) - 1.0
    y = np.clip(y, -1.0, 1.0)
    flat = q01 == q99
    if np.any(flat):
        y = np.where(flat, 0.0, y)
    return y.astype(np.float32, copy=False)


def _normalize_mean_std(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = (x - mean) / np.maximum(std, 1e-8)
    flat = std == 0
    if np.any(flat):
        y = np.where(flat, 0.0, y)
    return y.astype(np.float32, copy=False)


def _make_action_chunks(actions: np.ndarray, *, action_horizon: int, fill_strategy: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"`actions` must be 2D (T, Da), got {actions.shape}.")

    n_steps, action_dim = actions.shape
    out = np.zeros((n_steps, action_horizon, action_dim), dtype=np.float32)
    for t in range(n_steps):
        chunk = actions[t : t + action_horizon]
        out[t, : len(chunk)] = chunk
        if len(chunk) < action_horizon:
            if fill_strategy == "repeat_last" and len(chunk) > 0:
                out[t, len(chunk) :] = chunk[-1]
            elif fill_strategy == "zeros":
                out[t, len(chunk) :] = 0.0
            else:
                raise ValueError(f"Unknown fill_strategy={fill_strategy!r}.")
    return out


class DynamicActionChunkSampler:
    def __init__(
        self,
        *,
        repo_id: str,
        action_horizon: int,
        fill_strategy: str = "zeros",
        proprio_similarity_threshold: float | None = None,
        similarity_temperature: float = 1.0,
        seed: int | None = 0,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be > 0, got {action_horizon}.")
        if proprio_similarity_threshold is not None and not -1.0 <= proprio_similarity_threshold <= 1.0:
            raise ValueError(
                "`proprio_similarity_threshold` must be within [-1.0, 1.0], "
                f"got {proprio_similarity_threshold}."
            )
        if similarity_temperature <= 0.0:
            raise ValueError(
                f"`similarity_temperature` must be positive, got {similarity_temperature}."
            )

        self.repo_id = repo_id
        self.action_horizon = int(action_horizon)
        self.fill_strategy = fill_strategy
        self.proprio_similarity_threshold = (
            float(proprio_similarity_threshold) if proprio_similarity_threshold is not None else None
        )
        self.similarity_temperature = float(similarity_temperature)
        self.repo_root = Path(HF_LEROBOT_HOME) / repo_id
        self.rng = np.random.default_rng(seed)

        if not self.repo_root.exists():
            raise FileNotFoundError(f"Dynamic action chunk sampler dataset root not found: {self.repo_root}")

        self.meta_dir = self.repo_root / "meta"
        self.info = _read_json(self.meta_dir / "info.json")
        self.state_key = _resolve_feature_column(self.info, "state")
        self.action_key = _resolve_feature_column(self.info, "actions")

        self._q01: np.ndarray | None = None
        self._q99: np.ndarray | None = None
        self._state_mean: np.ndarray | None = None
        self._state_std: np.ndarray | None = None
        self._action_q01: np.ndarray | None = None
        self._action_q99: np.ndarray | None = None
        self._action_mean: np.ndarray | None = None
        self._action_std: np.ndarray | None = None
        self._load_normalization_stats(self.meta_dir / "stats.json")
        self.ref_banks = self._load_reference_banks()

    @staticmethod
    def _instruction_key(instruction: str) -> str:
        return instruction.strip().lower()

    @staticmethod
    def _empty(action_chunk_size: int, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((1, action_chunk_size, action_dim), dtype=np.float32),
            np.zeros((1,), dtype=np.bool_),
        )

    def _load_normalization_stats(self, stats_path: Path) -> None:
        try:
            stats = _read_json(stats_path)
            stats_source = "meta/stats.json"
        except FileNotFoundError:
            try:
                from lerobot.common.datasets import lerobot_dataset

                metadata = lerobot_dataset.LeRobotDatasetMetadata(str(self.repo_root))
                stats = _convert_metadata_stats(metadata.stats)
                stats_source = "LeRobotDatasetMetadata.stats"
                warnings.warn(
                    "Dynamic action chunk sampling is using LeRobotDatasetMetadata.stats "
                    "because meta/stats.json is missing.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception as exc:
                raise FileNotFoundError(
                    f"Dynamic action chunk sampling could not load stats from {stats_path} "
                    "or LeRobotDatasetMetadata.stats."
                ) from exc

        state_stats = _resolve_stats_entry(stats, canonical_name="state", resolved_column=self.state_key)
        q01 = state_stats.get("q01")
        q99 = state_stats.get("q99")
        if q01 is not None and q99 is not None:
            self._q01 = np.asarray(q01, dtype=np.float32)
            self._q99 = np.asarray(q99, dtype=np.float32)
        else:
            mean = state_stats.get("mean")
            std = state_stats.get("std")
            if mean is not None and std is not None:
                self._state_mean = np.asarray(mean, dtype=np.float32)
                self._state_std = np.asarray(std, dtype=np.float32)
                warnings.warn(
                    "Dynamic action chunk sampling is falling back to `state.mean` and `state.std` "
                    f"because `{self.state_key}.q01`/`{self.state_key}.q99` are missing in {stats_source}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "Dynamic action chunk sampling requires stats to include either "
                    f"`{self.state_key}.q01`/`{self.state_key}.q99` or "
                    f"`{self.state_key}.mean`/`{self.state_key}.std`."
                )

        try:
            action_stats = _resolve_stats_entry(stats, canonical_name="actions", resolved_column=self.action_key)
        except KeyError:
            action_stats = {}
        action_q01 = action_stats.get("q01")
        action_q99 = action_stats.get("q99")
        if action_q01 is not None and action_q99 is not None:
            self._action_q01 = np.asarray(action_q01, dtype=np.float32)
            self._action_q99 = np.asarray(action_q99, dtype=np.float32)
            return

        action_mean = action_stats.get("mean")
        action_std = action_stats.get("std")
        if action_mean is not None and action_std is not None:
            self._action_mean = np.asarray(action_mean, dtype=np.float32)
            self._action_std = np.asarray(action_std, dtype=np.float32)
            warnings.warn(
                f"Dynamic action chunk sampling is falling back to `{self.action_key}.mean` and "
                f"`{self.action_key}.std` because `{self.action_key}.q01`/`{self.action_key}.q99` "
                f"are missing in {stats_source}.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _normalize_proprio(self, x: np.ndarray) -> np.ndarray:
        if self._q01 is not None and self._q99 is not None:
            return _normalize_q01_q99(x, self._q01, self._q99)
        if self._state_mean is not None and self._state_std is not None:
            return _normalize_mean_std(x, self._state_mean, self._state_std)
        return np.asarray(x, dtype=np.float32)

    def _normalize_action_chunk(self, x: np.ndarray) -> np.ndarray:
        if self._action_q01 is not None and self._action_q99 is not None:
            return _normalize_q01_q99(x, self._action_q01, self._action_q99)
        if self._action_mean is not None and self._action_std is not None:
            return _normalize_mean_std(x, self._action_mean, self._action_std)
        return np.asarray(x, dtype=np.float32)

    def _build_episode_infos(self) -> list[EpisodeInfo]:
        info = self.info
        rows = _read_jsonl(self.meta_dir / "episodes.jsonl")
        rows = sorted(rows, key=lambda r: int(r["episode_index"]))

        chunks_size = int(info["chunks_size"])
        data_path_pattern = str(info["data_path"])
        out: list[EpisodeInfo] = []
        for row in rows:
            episode_index = int(row["episode_index"])
            tasks = list(row.get("tasks", []))
            if len(tasks) != 1:
                raise ValueError(
                    f"Expected exactly one task per episode in {self.repo_root}, got {tasks} "
                    f"for episode {episode_index}."
                )
            rel_path = data_path_pattern.format(
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
            )
            source_path = self.repo_root / rel_path
            if not source_path.exists():
                raise FileNotFoundError(f"Episode parquet not found: {source_path}")
            out.append(
                EpisodeInfo(
                    episode_index=episode_index,
                    task=str(tasks[0]),
                    length=int(row["length"]),
                    source_path=source_path,
                )
            )
        return out

    def _build_reference_banks(self, episode_infos: list[EpisodeInfo]) -> dict[str, _RefBank]:
        banks_tmp: dict[str, dict[str, list[np.ndarray]]] = {}
        for episode_info in episode_infos:
            task_key = self._instruction_key(episode_info.task)
            rows = banks_tmp.setdefault(task_key, {"proprio": [], "action_chunks": [], "action_chunks_norm": []})

            table = pq.read_table(episode_info.source_path, columns=[self.state_key, self.action_key])
            states = np.asarray(table[self.state_key].to_pylist(), dtype=np.float32)
            actions = np.asarray(table[self.action_key].to_pylist(), dtype=np.float32)
            if states.ndim != 2:
                raise ValueError(
                    f"`{self.state_key}` must have shape (T, Ds), got {states.shape} "
                    f"in {episode_info.source_path}."
                )
            if actions.ndim != 2:
                raise ValueError(
                    f"`{self.action_key}` must have shape (T, Da), got {actions.shape} "
                    f"in {episode_info.source_path}."
                )

            state_norm = _normalize_rows(self._normalize_proprio(states))
            action_chunks = _make_action_chunks(
                actions,
                action_horizon=self.action_horizon,
                fill_strategy=self.fill_strategy,
            )
            action_chunks_norm = _normalize_rows(self._normalize_action_chunk(action_chunks))

            rows["proprio"].extend(state_norm.astype(np.float32, copy=False))
            rows["action_chunks"].extend(action_chunks.astype(np.float32, copy=False))
            rows["action_chunks_norm"].extend(action_chunks_norm.astype(np.float32, copy=False))

        out: dict[str, _RefBank] = {}
        if self._q01 is not None:
            state_dim = int(self._q01.shape[0])
        else:
            assert self._state_mean is not None
            state_dim = int(self._state_mean.shape[0])
        for task_key, rows in banks_tmp.items():
            if rows["proprio"]:
                proprio = np.stack(rows["proprio"], axis=0).astype(np.float32, copy=False)
                action_chunks = np.stack(rows["action_chunks"], axis=0).astype(np.float32, copy=False)
                action_chunks_norm = np.stack(rows["action_chunks_norm"], axis=0).astype(np.float32, copy=False)
            else:
                proprio = np.zeros((0, state_dim), dtype=np.float32)
                action_chunks = np.zeros((0, self.action_horizon, 0), dtype=np.float32)
                action_chunks_norm = np.zeros((0, self.action_horizon, 0), dtype=np.float32)
            out[task_key] = _RefBank(
                proprio=proprio,
                action_chunks=action_chunks,
                action_chunks_norm=action_chunks_norm,
            )
        return out

    def _reference_bank_cache_key(self) -> str:
        meta_dir = self.repo_root / "meta"
        fingerprint_parts = [
            str(self.repo_root.resolve()),
            str(self.action_horizon),
            self.fill_strategy,
        ]
        for name in ("info.json", "episodes.jsonl", "stats.json"):
            path = meta_dir / name
            try:
                stat = path.stat()
            except FileNotFoundError:
                fingerprint_parts.extend((name, "missing"))
            else:
                fingerprint_parts.extend((name, str(stat.st_size), str(stat.st_mtime_ns)))
        for episode_info in self._build_episode_infos():
            episode_path = episode_info.source_path
            episode_stat = episode_path.stat()
            fingerprint_parts.extend(
                (
                    str(episode_path.relative_to(self.repo_root)),
                    str(episode_stat.st_size),
                    str(episode_stat.st_mtime_ns),
                )
            )
        payload = "\n".join(fingerprint_parts).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

    def _reference_bank_cache_dir(self) -> Path:
        return _DYNAMIC_ACTION_CHUNK_CACHE_ROOT / self._reference_bank_cache_key()

    def _reference_bank_metadata_path(self) -> Path:
        return self._reference_bank_cache_dir() / "metadata.json"

    @staticmethod
    def _cache_metadata_is_compatible(metadata: dict[str, Any]) -> bool:
        if metadata.get("version") != 2:
            return False
        tasks = metadata.get("tasks")
        if not isinstance(tasks, dict):
            return False
        return all(
            isinstance(files, dict)
            and "proprio" in files
            and "action_chunks" in files
            and "action_chunks_norm" in files
            for files in tasks.values()
        )

    def _load_reference_banks_from_cache(self, metadata: dict[str, Any]) -> dict[str, _RefBank]:
        cache_dir = self._reference_bank_cache_dir()
        if not self._cache_metadata_is_compatible(metadata):
            raise ValueError(f"Incompatible dynamic action chunk cache metadata at {cache_dir}.")
        banks: dict[str, _RefBank] = {}
        tasks = metadata.get("tasks", {})
        if not isinstance(tasks, dict):
            raise ValueError(f"Invalid dynamic action chunk cache metadata at {cache_dir}.")

        for task_key, files in tasks.items():
            if not isinstance(files, dict):
                raise ValueError(f"Invalid cache entry for task {task_key!r} in {cache_dir}.")
            proprio_path = cache_dir / str(files["proprio"])
            action_chunks_path = cache_dir / str(files["action_chunks"])
            action_chunks_norm_path = cache_dir / str(files["action_chunks_norm"])
            banks[str(task_key)] = _RefBank(
                proprio=np.load(proprio_path, mmap_mode="r"),
                action_chunks=np.load(action_chunks_path, mmap_mode="r"),
                action_chunks_norm=np.load(action_chunks_norm_path, mmap_mode="r"),
            )
        return banks

    def _build_reference_bank_cache(self) -> dict[str, Any]:
        cache_dir = self._reference_bank_cache_dir()
        metadata_path = self._reference_bank_metadata_path()
        lock_path = cache_dir.parent / f"{cache_dir.name}.lock"

        with _exclusive_file_lock(lock_path):
            if metadata_path.exists():
                metadata = _read_json(metadata_path)
                if self._cache_metadata_is_compatible(metadata):
                    return metadata

            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            tmp_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{cache_dir.name}.tmp.",
                    dir=str(cache_dir.parent),
                )
            )
            try:
                banks = self._build_reference_banks(self._build_episode_infos())
                metadata = {"version": 2, "tasks": {}}
                for idx, (task_key, bank) in enumerate(sorted(banks.items())):
                    proprio_name = f"bank_{idx:04d}_proprio.npy"
                    action_chunks_name = f"bank_{idx:04d}_action_chunks.npy"
                    action_chunks_norm_name = f"bank_{idx:04d}_action_chunks_norm.npy"
                    np.save(tmp_dir / proprio_name, bank.proprio, allow_pickle=False)
                    np.save(tmp_dir / action_chunks_name, bank.action_chunks, allow_pickle=False)
                    np.save(tmp_dir / action_chunks_norm_name, bank.action_chunks_norm, allow_pickle=False)
                    metadata["tasks"][task_key] = {
                        "proprio": proprio_name,
                        "action_chunks": action_chunks_name,
                        "action_chunks_norm": action_chunks_norm_name,
                    }
                _write_json(tmp_dir / "metadata.json", metadata)
                tmp_dir.replace(cache_dir)
                return metadata
            except Exception:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

    def _load_reference_banks(self) -> dict[str, _RefBank]:
        cache_key = self._reference_bank_cache_key()
        cached = _REFERENCE_BANKS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        metadata_path = self._reference_bank_metadata_path()
        metadata = _read_json(metadata_path) if metadata_path.exists() else self._build_reference_bank_cache()
        if not self._cache_metadata_is_compatible(metadata):
            metadata = self._build_reference_bank_cache()
        banks = self._load_reference_banks_from_cache(metadata)
        _REFERENCE_BANKS_CACHE[cache_key] = banks
        return banks

    def _resolve_reference_banks(self, *, instruction: str | None) -> tuple[list[_RefBank], str]:
        if instruction is None:
            return [bank for bank in self.ref_banks.values() if bank.proprio.shape[0] > 0], "global reference bank"

        instruction_key = self._instruction_key(instruction)
        bank = self.ref_banks.get(instruction_key)
        if bank is None or bank.proprio.shape[0] == 0:
            return [], f"instruction {instruction!r}"
        return [bank], f"instruction {instruction!r}"

    def _collect_candidate_pools(
        self,
        *,
        banks: list[_RefBank],
        query: np.ndarray,
        use_proprio_threshold: bool,
        fallback_to_nearest_if_proprio_empty: bool,
    ) -> list[_CandidatePool]:
        pools: list[_CandidatePool] = []
        best_bank: _RefBank | None = None
        best_idx: int | None = None
        best_similarity: float | None = None

        for bank in banks:
            if bank.proprio.shape[1] != query.shape[0]:
                raise ValueError(
                    f"Proprio dimension mismatch between query ({query.shape[0]}) and "
                    f"reference bank ({bank.proprio.shape[1]})."
                )

            similarities = bank.proprio @ query
            if similarities.size == 0:
                continue

            local_best_idx = int(np.argmax(similarities))
            local_best_similarity = float(similarities[local_best_idx])
            if best_similarity is None or local_best_similarity > best_similarity:
                best_bank = bank
                best_idx = local_best_idx
                best_similarity = local_best_similarity

            if use_proprio_threshold and self.proprio_similarity_threshold is not None:
                valid = similarities >= self.proprio_similarity_threshold
                if not np.any(valid):
                    continue
                candidate_indices = np.nonzero(valid)[0].astype(np.int64, copy=False)
                candidate_similarities = similarities[valid].astype(np.float32, copy=False)
            else:
                candidate_indices = np.arange(bank.proprio.shape[0], dtype=np.int64)
                candidate_similarities = similarities.astype(np.float32, copy=False)

            pools.append(
                _CandidatePool(
                    bank=bank,
                    indices=candidate_indices,
                    similarities=candidate_similarities,
                )
            )

        if pools:
            return pools

        if not fallback_to_nearest_if_proprio_empty or best_bank is None or best_idx is None or best_similarity is None:
            return []

        return [
            _CandidatePool(
                bank=best_bank,
                indices=np.asarray([best_idx], dtype=np.int64),
                similarities=np.asarray([best_similarity], dtype=np.float32),
            )
        ]

    @staticmethod
    def _collect_all_candidate_pools(banks: list[_RefBank]) -> list[_CandidatePool]:
        pools: list[_CandidatePool] = []
        for bank in banks:
            num_candidates = int(bank.action_chunks.shape[0])
            if num_candidates <= 0:
                continue
            pools.append(
                _CandidatePool(
                    bank=bank,
                    indices=np.arange(num_candidates, dtype=np.int64),
                    similarities=np.zeros((num_candidates,), dtype=np.float32),
                )
            )
        return pools

    def _sample_uniform_candidate_batch(
        self,
        banks: list[_RefBank],
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        counts = np.asarray([int(bank.action_chunks.shape[0]) for bank in banks], dtype=np.int64)
        total_candidates = int(np.sum(counts))
        if total_candidates <= 0:
            return None

        cumulative_counts = np.cumsum(counts)
        sampled_offsets = self.rng.integers(total_candidates, size=batch_size, dtype=np.int64)
        bank_indices = np.searchsorted(cumulative_counts, sampled_offsets, side="right").astype(
            np.int64,
            copy=False,
        )
        previous_counts = np.zeros_like(sampled_offsets)
        has_previous_bank = bank_indices > 0
        previous_counts[has_previous_bank] = cumulative_counts[bank_indices[has_previous_bank] - 1]
        local_indices = sampled_offsets - previous_counts
        return bank_indices, local_indices

    def _sample_uniform_candidate_by_action_similarity_rejection(
        self,
        banks: list[_RefBank],
        *,
        action_chunk_size: int,
        query_chunk_norm: np.ndarray,
        action_similarity_min: float | None,
        action_similarity_max: float | None,
    ) -> tuple[_RefBank, int] | None:
        query_vector = np.asarray(query_chunk_norm, dtype=np.float32).reshape(-1)
        remaining_attempts = _UNIFORM_ACTION_FILTER_REJECTION_ATTEMPTS
        while remaining_attempts > 0:
            batch_size = min(_UNIFORM_ACTION_FILTER_REJECTION_BATCH_SIZE, remaining_attempts)
            remaining_attempts -= batch_size
            sampled = self._sample_uniform_candidate_batch(banks, batch_size=batch_size)
            if sampled is None:
                return None
            sampled_bank_indices, sampled_local_indices = sampled
            valid_bank_indices: list[np.ndarray] = []
            valid_local_indices: list[np.ndarray] = []
            for bank_idx in np.unique(sampled_bank_indices):
                bank_mask = sampled_bank_indices == bank_idx
                local_indices = sampled_local_indices[bank_mask].astype(np.int64, copy=False)
                bank = banks[int(bank_idx)]
                candidate_action_chunks_norm = np.asarray(
                    bank.action_chunks_norm[local_indices],
                    dtype=np.float32,
                ).reshape(local_indices.shape[0], -1)
                action_cos = (candidate_action_chunks_norm @ query_vector) / float(action_chunk_size)
                action_sims = np.clip(0.5 + 0.5 * action_cos, 0.0, 1.0)
                action_valid = np.ones_like(action_sims, dtype=np.bool_)
                if action_similarity_min is not None:
                    action_valid &= action_sims >= action_similarity_min
                if action_similarity_max is not None:
                    action_valid &= action_sims <= action_similarity_max
                if not np.any(action_valid):
                    continue
                valid_count = int(np.count_nonzero(action_valid))
                valid_bank_indices.append(np.full((valid_count,), int(bank_idx), dtype=np.int64))
                valid_local_indices.append(local_indices[action_valid])

            if valid_bank_indices:
                selected_valid = int(self.rng.integers(sum(len(indices) for indices in valid_local_indices)))
                for bank_idx_values, local_idx_values in zip(
                    valid_bank_indices,
                    valid_local_indices,
                    strict=True,
                ):
                    if selected_valid < local_idx_values.shape[0]:
                        return banks[int(bank_idx_values[selected_valid])], int(local_idx_values[selected_valid])
                    selected_valid -= int(local_idx_values.shape[0])
                raise RuntimeError("Failed to resolve sampled valid action chunk candidate.")
        return None

    def _filter_candidate_pools_by_action_similarity(
        self,
        pools: list[_CandidatePool],
        *,
        action_chunk_size: int,
        query_chunk_norm: np.ndarray,
        action_similarity_min: float | None,
        action_similarity_max: float | None,
    ) -> list[_CandidatePool]:
        filtered: list[_CandidatePool] = []
        for pool in pools:
            candidate_action_chunks_norm = pool.bank.action_chunks_norm[pool.indices].reshape(pool.indices.shape[0], -1)
            action_cos = (candidate_action_chunks_norm @ query_chunk_norm.T)[:, 0] / float(action_chunk_size)
            action_sims = np.clip(0.5 + 0.5 * action_cos, 0.0, 1.0)
            action_valid = np.ones_like(action_sims, dtype=np.bool_)
            if action_similarity_min is not None:
                action_valid &= action_sims >= action_similarity_min
            if action_similarity_max is not None:
                action_valid &= action_sims <= action_similarity_max
            if not np.any(action_valid):
                continue

            filtered.append(
                _CandidatePool(
                    bank=pool.bank,
                    indices=pool.indices[action_valid],
                    similarities=pool.similarities[action_valid],
                )
            )
        return filtered

    def _sample_from_candidate_pools(
        self,
        pools: list[_CandidatePool],
        *,
        sampling_strategy: str,
    ) -> tuple[_RefBank, int]:
        if sampling_strategy == "uniform":
            total_candidates = sum(int(pool.indices.shape[0]) for pool in pools)
            sampled_offset = int(self.rng.integers(total_candidates))
            for pool in pools:
                pool_size = int(pool.indices.shape[0])
                if sampled_offset < pool_size:
                    return pool.bank, int(pool.indices[sampled_offset])
                sampled_offset -= pool_size
            raise RuntimeError("Failed to resolve sampled candidate offset.")

        max_logit = max(float(np.max(pool.similarities)) / self.similarity_temperature for pool in pools)
        pool_weights: list[np.ndarray] = []
        pool_weight_sums: list[float] = []
        total_weight = 0.0
        for pool in pools:
            logits = pool.similarities / self.similarity_temperature
            weights = np.exp(logits - max_logit)
            weight_sum = float(np.sum(weights))
            pool_weights.append(weights)
            pool_weight_sums.append(weight_sum)
            total_weight += weight_sum

        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError(
                "Non-finite or non-positive probabilities computed for dynamic action chunk sampling. "
                f"pool_weight_sums={pool_weight_sums}."
            )

        sampled_mass = float(self.rng.random()) * total_weight
        for pool, weights, weight_sum in zip(pools, pool_weights, pool_weight_sums, strict=True):
            if sampled_mass < weight_sum:
                cumulative = np.cumsum(weights)
                sampled_local_idx = int(np.searchsorted(cumulative, sampled_mass, side="right"))
                sampled_local_idx = min(sampled_local_idx, pool.indices.shape[0] - 1)
                return pool.bank, int(pool.indices[sampled_local_idx])
            sampled_mass -= weight_sum

        last_pool = pools[-1]
        return last_pool.bank, int(last_pool.indices[-1])

    def sample(
        self,
        *,
        instruction: str | None = None,
        proprio: np.ndarray,
        action_chunk_size: int,
        action_dim: int,
        query_action_chunk: np.ndarray | None = None,
        action_similarity_threshold: float | None = None,
        action_similarity_min: float | None = None,
        action_similarity_max: float | None = None,
        sampling_strategy: str = "weighted",
        use_proprio_threshold: bool = True,
        fallback_to_nearest_if_proprio_empty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if action_similarity_threshold is not None:
            if action_similarity_min is not None:
                raise ValueError(
                    "Specify either `action_similarity_threshold` or `action_similarity_min`, not both."
                )
            action_similarity_min = action_similarity_threshold
        for name, value in (
            ("action_similarity_min", action_similarity_min),
            ("action_similarity_max", action_similarity_max),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"`{name}` must be within [0.0, 1.0], got {value}.")
        if (
            action_similarity_min is not None
            and action_similarity_max is not None
            and action_similarity_min > action_similarity_max
        ):
            raise ValueError(
                "`action_similarity_min` must be <= `action_similarity_max`, "
                f"got {action_similarity_min} > {action_similarity_max}."
            )
        if (action_similarity_min is not None or action_similarity_max is not None) and query_action_chunk is None:
            raise ValueError(
                "`query_action_chunk` must be provided when action similarity filtering is enabled."
            )
        if sampling_strategy not in {"weighted", "uniform"}:
            raise ValueError(
                "`sampling_strategy` must be either 'weighted' or 'uniform', "
                f"got {sampling_strategy!r}."
            )

        banks, bank_label = self._resolve_reference_banks(instruction=instruction)
        if not banks:
            print(
                "[Warning] No dynamic action chunk references found for "
                f"{bank_label} in dataset {self.repo_id!r}; returning masked zeros."
            )
            return self._empty(action_chunk_size, action_dim)

        query_chunk_norm = None
        if action_similarity_min is not None or action_similarity_max is not None:
            query_chunk = np.asarray(query_action_chunk, dtype=np.float32)
            if query_chunk.shape != (action_chunk_size, action_dim):
                raise ValueError(
                    f"`query_action_chunk` must have shape ({action_chunk_size}, {action_dim}), "
                    f"got {query_chunk.shape}."
                )
            query_chunk_norm = _normalize_rows(self._normalize_action_chunk(query_chunk)).reshape(1, -1)

        if (
            query_chunk_norm is not None
            and sampling_strategy == "uniform"
            and not use_proprio_threshold
        ):
            sampled = self._sample_uniform_candidate_by_action_similarity_rejection(
                banks,
                action_chunk_size=action_chunk_size,
                query_chunk_norm=query_chunk_norm,
                action_similarity_min=action_similarity_min,
                action_similarity_max=action_similarity_max,
            )
            if sampled is not None:
                sampled_bank, sampled_idx = sampled
                sampled_chunk = np.asarray(sampled_bank.action_chunks[sampled_idx], dtype=np.float32)
                if sampled_chunk.shape != (action_chunk_size, action_dim):
                    raise ValueError(
                        f"Sampled action chunk has shape {sampled_chunk.shape}, expected "
                        f"({action_chunk_size}, {action_dim})."
                    )
                return sampled_chunk[None, ...], np.ones((1,), dtype=np.bool_)

        if sampling_strategy == "uniform" and not use_proprio_threshold:
            pools = self._collect_all_candidate_pools(banks)
        else:
            query = np.asarray(proprio, dtype=np.float32).reshape(-1)
            query = _normalize_rows(self._normalize_proprio(query))[0]
            pools = self._collect_candidate_pools(
                banks=banks,
                query=query,
                use_proprio_threshold=use_proprio_threshold,
                fallback_to_nearest_if_proprio_empty=fallback_to_nearest_if_proprio_empty,
            )
        if not pools:
            return self._empty(action_chunk_size, action_dim)

        if query_chunk_norm is not None:
            pools = self._filter_candidate_pools_by_action_similarity(
                pools,
                action_chunk_size=action_chunk_size,
                query_chunk_norm=query_chunk_norm,
                action_similarity_min=action_similarity_min,
                action_similarity_max=action_similarity_max,
            )
            if not pools:
                return self._empty(action_chunk_size, action_dim)

        sampled_bank, sampled_idx = self._sample_from_candidate_pools(
            pools,
            sampling_strategy=sampling_strategy,
        )

        sampled_chunk = np.asarray(sampled_bank.action_chunks[sampled_idx], dtype=np.float32)
        if sampled_chunk.shape != (action_chunk_size, action_dim):
            raise ValueError(
                f"Sampled action chunk has shape {sampled_chunk.shape}, expected "
                f"({action_chunk_size}, {action_dim})."
            )
        return sampled_chunk[None, ...], np.ones((1,), dtype=np.bool_)
