from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


DEFAULT_CONFIG_NAME = "pi05_hsr_svr_chunk32_nochunkcond"
DEFAULT_REPO_ID = "local/hsr_task6891011_level12_s2refpa_b020_20260504"
DEFAULT_PARAMS_DIR_NAME = "params"

_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class BundlePaths:
    bundle_root: Path
    checkpoint_dir: Path
    assets_dir: Path
    resolver_bundle: Path
    openpi_root: Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def normalize_instruction(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("pick-up", "pick up")
    text = text.replace("pickup", "pick up")
    text = text.replace("grey", "gray")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def instruction_similarity(query: str, candidate: str) -> float:
    query_norm = normalize_instruction(query)
    candidate_norm = normalize_instruction(candidate)
    if not query_norm and not candidate_norm:
        return 1.0
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    query_tokens = set(query_norm.split())
    candidate_tokens = set(candidate_norm.split())
    union = query_tokens | candidate_tokens
    jaccard = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
    seq = SequenceMatcher(None, query_norm, candidate_norm).ratio()
    contains = 1.0 if query_norm in candidate_norm or candidate_norm in query_norm else 0.0
    return 0.55 * seq + 0.35 * jaccard + 0.10 * contains


class HSRPromptResolver:
    def __init__(self, resolver_bundle_path: str | Path, *, top_k: int = 5) -> None:
        self.path = Path(resolver_bundle_path)
        with self.path.open("r", encoding="utf-8") as f:
            self.bundle: dict[str, Any] = json.load(f)
        self.top_k = top_k
        self._entries: list[dict[str, Any]] = list(self.bundle["entries"])
        self._by_original = {entry["original_instruction"]: entry for entry in self._entries}
        self._training_prompts = {entry["canonical_training_prompt"] for entry in self._entries}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "resolver_bundle": str(self.path),
            "resolver_bundle_type": self.bundle.get("bundle_type", "unknown"),
            "resolver_unique_original_instructions": self.bundle.get("unique_original_instructions", 0),
        }

    def is_training_prompt(self, prompt: str) -> bool:
        return prompt in self._training_prompts or ". Details:" in prompt

    def resolve(self, instruction: str) -> dict[str, Any]:
        instruction = (instruction or "").strip()
        instruction_norm = normalize_instruction(instruction)
        exact_lookup = self.bundle.get("exact_lookup", {})
        if instruction_norm in exact_lookup:
            matched_original = exact_lookup[instruction_norm]
            entry = self._by_original[matched_original]
            return {
                "query_instruction": instruction,
                "normalized_query_instruction": instruction_norm,
                "match_type": "exact_normalized",
                "score": 1.0,
                "matched_original_instruction": entry["original_instruction"],
                "canonical_refined_instruction": entry["canonical_refined_instruction"],
                "canonical_training_prompt": entry["canonical_training_prompt"],
                "canonical_parent_task": entry["canonical_parent_task"],
                "entry": entry,
                "candidates": [
                    {
                        "original_instruction": entry["original_instruction"],
                        "score": 1.0,
                        "canonical_refined_instruction": entry["canonical_refined_instruction"],
                    }
                ],
            }

        scored = sorted(
            (
                {
                    "entry": entry,
                    "score": instruction_similarity(instruction_norm, entry["normalized_original_instruction"]),
                }
                for entry in self._entries
            ),
            key=lambda item: (
                -item["score"],
                -item["entry"]["episode_count"],
                item["entry"]["original_instruction"],
            ),
        )
        best = scored[0]
        return {
            "query_instruction": instruction,
            "normalized_query_instruction": instruction_norm,
            "match_type": "nearest_original_instruction",
            "score": best["score"],
            "matched_original_instruction": best["entry"]["original_instruction"],
            "canonical_refined_instruction": best["entry"]["canonical_refined_instruction"],
            "canonical_training_prompt": best["entry"]["canonical_training_prompt"],
            "canonical_parent_task": best["entry"]["canonical_parent_task"],
            "entry": best["entry"],
            "candidates": [
                {
                    "original_instruction": item["entry"]["original_instruction"],
                    "score": item["score"],
                    "canonical_refined_instruction": item["entry"]["canonical_refined_instruction"],
                }
                for item in scored[: max(1, self.top_k)]
            ],
        }


def resolve_bundle_paths(
    checkpoint_path: str | Path,
    *,
    assets_dir: str | Path | None = None,
    resolver_bundle: str | Path | None = None,
    openpi_root: str | Path | None = None,
) -> BundlePaths:
    root = Path(checkpoint_path).expanduser().resolve()
    if (root / "checkpoint" / "149999" / "params").is_dir():
        bundle_root = root
        checkpoint_dir = root / "checkpoint" / "149999"
    elif (root / "params").is_dir():
        checkpoint_dir = root
        bundle_root = root.parent.parent if root.parent.name == "checkpoint" else root
    elif (root / "149999" / "params").is_dir():
        checkpoint_dir = root / "149999"
        bundle_root = root.parent if root.name == "checkpoint" else root
    else:
        raise FileNotFoundError(
            "Could not find an OpenPI params directory. Set POLICY_CHECKPOINT_PATH "
            "to the portable bundle root or to checkpoint/149999."
        )

    resolved_assets_dir = Path(assets_dir).expanduser().resolve() if assets_dir else None
    if resolved_assets_dir is None:
        for candidate in (bundle_root / "assets_hsr_epstats", checkpoint_dir / "assets"):
            if candidate.is_dir():
                resolved_assets_dir = candidate
                break
    if resolved_assets_dir is None:
        raise FileNotFoundError("Could not find HSR norm stats assets in the portable bundle.")

    resolved_resolver_bundle = Path(resolver_bundle).expanduser().resolve() if resolver_bundle else None
    if resolved_resolver_bundle is None:
        resolved_resolver_bundle = bundle_root / "resolver" / "hsr_pa_refinement_bundle.json"
    if not resolved_resolver_bundle.is_file():
        raise FileNotFoundError(f"Resolver bundle not found: {resolved_resolver_bundle}")

    resolved_openpi_root = Path(openpi_root).expanduser().resolve() if openpi_root else None
    if resolved_openpi_root is None:
        resolved_openpi_root = bundle_root / "openpi"
    if not (resolved_openpi_root / "src" / "openpi").is_dir():
        raise FileNotFoundError(f"Vendored OpenPI source not found under: {resolved_openpi_root}")

    return BundlePaths(
        bundle_root=bundle_root,
        checkpoint_dir=checkpoint_dir,
        assets_dir=resolved_assets_dir,
        resolver_bundle=resolved_resolver_bundle,
        openpi_root=resolved_openpi_root,
    )


def ensure_openpi_import_paths(openpi_root: Path) -> None:
    paths = [
        openpi_root / "src",
        openpi_root / "packages" / "openpi-client" / "src",
    ]
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _build_hsr_aligned_model_config(model_cfg: Any) -> Any:
    return dataclasses.replace(
        model_cfg,
        use_visual_budget_gate=True,
        visual_budget_gate_hidden_dim=256,
        visual_budget_gate_floor=0.1,
        visual_budget_gate_ignore_priors=True,
        visual_budget_gate_use_prior_bias=True,
        visual_budget_gate_inference_mode="off",
        visual_budget_gate_keep_ratio_base=0.2,
        visual_budget_gate_keep_ratio_wrist=0.2,
    )


class HSRPortableOpenPIPolicy:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        config_name: str = DEFAULT_CONFIG_NAME,
        repo_id: str = DEFAULT_REPO_ID,
        params_dir_name: str = DEFAULT_PARAMS_DIR_NAME,
        assets_dir: str | Path | None = None,
        resolver_bundle: str | Path | None = None,
        openpi_root: str | Path | None = None,
        strict_exact_match: bool | None = None,
        resolve_prompt: bool | None = None,
        top_k: int | None = None,
        pytorch_device: str | None = None,
    ) -> None:
        self.paths = resolve_bundle_paths(
            checkpoint_path,
            assets_dir=assets_dir or os.environ.get("POLICY_ASSETS_DIR"),
            resolver_bundle=resolver_bundle or os.environ.get("POLICY_RESOLVER_BUNDLE"),
            openpi_root=openpi_root or os.environ.get("POLICY_OPENPI_ROOT"),
        )
        ensure_openpi_import_paths(self.paths.openpi_root)

        self.config_name = os.environ.get("POLICY_CONFIG_NAME", config_name)
        self.repo_id = os.environ.get("POLICY_REPO_ID", repo_id)
        self.params_dir_name = os.environ.get("POLICY_PARAMS_DIR_NAME", params_dir_name)
        self.strict_exact_match = (
            _env_bool("POLICY_STRICT_EXACT_MATCH", False) if strict_exact_match is None else strict_exact_match
        )
        self.resolve_prompt = _env_bool("POLICY_RESOLVE_PROMPT", True) if resolve_prompt is None else resolve_prompt
        self.top_k = int(os.environ.get("POLICY_RESOLVER_TOP_K", top_k or 5))
        self.pytorch_device = pytorch_device or os.environ.get("POLICY_PYTORCH_DEVICE")

        self.resolver = HSRPromptResolver(self.paths.resolver_bundle, top_k=self.top_k)
        self._base_policy = self._load_policy()
        self.metadata = dict(getattr(self._base_policy, "metadata", {}))
        self.metadata.update(
            {
                "policy": "hsr689_s2_portable_openpi",
                "checkpoint_dir": str(self.paths.checkpoint_dir),
                "assets_dir": str(self.paths.assets_dir),
                "openpi_root": str(self.paths.openpi_root),
                "config_name": self.config_name,
                "repo_id": self.repo_id,
                "params_dir_name": self.params_dir_name,
                "prompt_contract": "raw_pa_prompt_resolved_to_canonical_training_prompt",
                "strict_exact_match": self.strict_exact_match,
                "resolve_prompt": self.resolve_prompt,
                **self.resolver.metadata,
            }
        )

    def _load_policy(self) -> Any:
        from openpi.policies import policy_config as openpi_policy_config
        from openpi.training import checkpoints as openpi_checkpoints
        from openpi.training import config as openpi_config

        logging.info("Loading HSR689 S2 OpenPI policy from %s", self.paths.checkpoint_dir)
        train_cfg = openpi_config.get_config(self.config_name)
        train_cfg = dataclasses.replace(
            train_cfg,
            model=_build_hsr_aligned_model_config(train_cfg.model),
            data=dataclasses.replace(
                train_cfg.data,
                repo_id=self.repo_id,
                adapt_to_pi=True,
                freeze_low_variance_actions=False,
                assets=dataclasses.replace(train_cfg.data.assets, assets_dir=str(self.paths.assets_dir)),
            ),
        )

        data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
        norm_stats = None
        if data_cfg.asset_id:
            norm_stats_dir = self.paths.assets_dir / data_cfg.asset_id
            if norm_stats_dir.exists():
                logging.info("Loading HSR norm stats from %s", norm_stats_dir)
                norm_stats = openpi_checkpoints.load_norm_stats(self.paths.assets_dir, data_cfg.asset_id)
        else:
            norm_stats = {}

        return openpi_policy_config.create_trained_policy(
            train_cfg,
            str(self.paths.checkpoint_dir),
            default_prompt=None,
            norm_stats=norm_stats,
            pytorch_device=self.pytorch_device,
            params_dir_name=self.params_dir_name,
        )

    def prepare_prompt(self, raw_prompt: str) -> tuple[str, dict[str, Any]]:
        raw_prompt = (raw_prompt or "").strip()
        if not raw_prompt:
            raise ValueError("Observation prompt is empty; expected a raw PA instruction.")
        if not self.resolve_prompt or self.resolver.is_training_prompt(raw_prompt):
            return raw_prompt, {"match_type": "explicit_training_prompt", "score": 1.0}

        resolver_info = self.resolver.resolve(raw_prompt)
        if self.strict_exact_match and resolver_info["match_type"] != "exact_normalized":
            raise ValueError(
                f"Instruction {resolver_info['query_instruction']!r} did not exact-match the closed-set HSR PA inventory."
            )
        return resolver_info["canonical_training_prompt"], resolver_info

    def infer(self, prepared_obs: dict[str, Any]) -> dict[str, Any]:
        return self._base_policy.infer(prepared_obs)


def validate_actions(actions: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Policy returned actions with ndim={actions.ndim}; expected shape (T, 11).")
    if actions.shape[0] < 1 or actions.shape[1] != 11:
        raise ValueError(f"Policy returned actions with shape {actions.shape}; expected shape (T, 11).")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned non-finite actions.")
    return actions
