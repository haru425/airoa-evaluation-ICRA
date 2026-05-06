import dataclasses
import logging
import os
import pathlib
from typing import Any

import flax.traverse_util
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
import openpi.policies.policy as _policy
import openpi.shared.array_typing as at
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _has_nested_key(tree: Any, key_path: tuple[str, ...]) -> bool:
    node = tree
    for key in key_path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _checkpoint_uses_split_discriminator(params: Any) -> bool:
    return (
        _has_nested_key(params, ("discriminator_out_proj_instruction",))
        or _has_nested_key(params, ("action_proj_discriminator_instruction",))
        or _has_nested_key(params, ("discriminator_token_instruction",))
        or _has_nested_key(params, ("PaliGemma", "llm", "final_norm_11"))
        or _has_nested_key(params, ("PaliGemma", "llm", "layers", "pre_attention_norm_11"))
    )


def _checkpoint_uses_lora(params: Any) -> bool:
    if not isinstance(params, dict):
        return False
    for key, value in params.items():
        if "lora" in str(key):
            return True
        if _checkpoint_uses_lora(value):
            return True
    return False


def _set_lora_variant(variant: str, *, enabled: bool) -> str:
    if enabled:
        return variant if variant.endswith("_lora") else f"{variant}_lora"
    return variant.removesuffix("_lora")


_IQL_ACTOR_EXPERT_INDEX = 4
_POLICY_ACTION_EXPERT_INDEX = 1
_IQL_SHARED_LORA_ADAPTER_COUNTS = frozenset((4, 7, 8))
_IQL_POLICY_TOP_LEVEL_PREFIXES = (
    ("action_in_proj",),
    ("action_out_proj",),
    ("time_mlp_in",),
    ("time_mlp_out",),
    ("state_proj",),
    ("action_time_mlp_in",),
    ("action_time_mlp_out",),
)
_GEMMA_EXPERT_MODULE_BASES = (
    "pre_attention_norm",
    "pre_ffw_norm",
    "qkv_einsum",
    "q_einsum",
    "kv_einsum",
    "attn_vec_einsum",
    "mlp",
    "final_norm",
)


def _has_path_prefix(key_path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return key_path[: len(prefix)] == prefix


def _gemma_expert_index(path_part: str) -> int | None:
    for base in _GEMMA_EXPERT_MODULE_BASES:
        if path_part == base:
            return 0
        prefix = f"{base}_"
        if path_part.startswith(prefix):
            suffix = path_part[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
    return None


def _rename_gemma_expert_index(path_part: str, new_index: int) -> str:
    for base in _GEMMA_EXPERT_MODULE_BASES:
        if path_part == base:
            return base if new_index == 0 else f"{base}_{new_index}"
        prefix = f"{base}_"
        if path_part.startswith(prefix) and path_part[len(prefix) :].isdigit():
            return base if new_index == 0 else f"{base}_{new_index}"
    return path_part


def _is_lora_param_path(key_path: tuple[str, ...]) -> bool:
    return any("lora" in path_part for path_part in key_path)


def _is_iql_policy_param_key(key_path: tuple[str, ...]) -> bool:
    if any(_has_path_prefix(key_path, prefix) for prefix in _IQL_POLICY_TOP_LEVEL_PREFIXES):
        return True

    if _has_path_prefix(key_path, ("PaliGemma", "img_actor")):
        return True
    if _has_path_prefix(key_path, ("PaliGemma", "img")):
        # Legacy IQL checkpoints used one shared image encoder at the same path as policy-only checkpoints.
        return True

    if not _has_path_prefix(key_path, ("PaliGemma", "llm")):
        return False

    for path_part in key_path:
        expert_index = _gemma_expert_index(path_part)
        if expert_index is not None:
            return expert_index in (0, _IQL_ACTOR_EXPERT_INDEX)

    # Shared LLM params such as the token embedder are policy params.
    return True


def _slice_iql_lora_adapter(key_path: tuple[str, ...], value: Any) -> Any:
    if not _is_lora_param_path(key_path):
        return value
    if not _has_path_prefix(key_path, ("PaliGemma", "llm")):
        return value
    shape = getattr(value, "shape", ())
    if not shape or shape[0] not in _IQL_SHARED_LORA_ADAPTER_COUNTS:
        return value
    return value[0]


def _remap_iql_policy_params(params: at.Params) -> at.Params:
    """Convert IQL actor params to the normal policy-only pi0.5 parameter layout."""

    flat_params = flax.traverse_util.flatten_dict(params)
    remapped: dict[tuple[str, ...], Any] = {}
    for key_path, value in flat_params.items():
        if not _is_iql_policy_param_key(key_path):
            continue

        if _has_path_prefix(key_path, ("PaliGemma", "img_actor")):
            new_key_path = ("PaliGemma", "img", *key_path[2:])
        elif _has_path_prefix(key_path, ("PaliGemma", "llm")):
            new_key_path = tuple(
                _rename_gemma_expert_index(path_part, _POLICY_ACTION_EXPERT_INDEX)
                if _gemma_expert_index(path_part) == _IQL_ACTOR_EXPERT_INDEX
                else path_part
                for path_part in key_path
            )
        else:
            new_key_path = key_path

        # Prefer the explicit actor image encoder when both legacy shared and split image keys exist.
        if new_key_path in remapped and not _has_path_prefix(key_path, ("PaliGemma", "img_actor")):
            continue
        remapped[new_key_path] = _slice_iql_lora_adapter(key_path, value)

    return flax.traverse_util.unflatten_dict(remapped)


def _make_pi0_policy_only_config(train_config: _config.TrainConfig, params: at.Params) -> _config.TrainConfig:
    model_config = train_config.model
    if not isinstance(model_config, _pi0_config.Pi0Config) or not model_config.use_iql:
        return train_config

    checkpoint_uses_lora = _checkpoint_uses_lora(params)
    adjusted_fields: dict[str, Any] = {
        "use_iql": False,
        "split_discriminator_head": False,
        "instruction_discriminator_only_pretrain": False,
        "iql_actor_loss_type": "awr",
        "use_binary_reward": False,
        "action_relabeling_ratio_for_disc_l": 0.0,
        "joint_relabeling_ratio_for_disc_l": 0.0,
        "flipped_image_action_ratio_for_disc_l": 0.0,
        "flipped_image_only_ratio_for_disc_l": 0.0,
        "flipped_action_only_ratio_for_disc_l": 0.0,
        "use_obs_action_similarity_as_weight": False,
        "entropy_regularization_coef": 0.0,
        "actor_instruction_discriminator_topk_ratio": 1.0,
        "treat_instruction_only_as_unlabeled_for_disc_l": False,
        "treat_action_only_as_unlabeled_for_disc_l": False,
        "discriminator_entropy_exclude_unlabeled": False,
        "discriminator_front_camera_zero_probability": 0.0,
        "discriminator_wrist_camera_zero_probability": 0.0,
        "discriminator_proprio_zero_probability": 0.0,
        "rel_disc_reward_weight": 0.5,
    }

    model_uses_lora = "lora" in model_config.paligemma_variant or "lora" in model_config.action_expert_variant
    if checkpoint_uses_lora != model_uses_lora:
        adjusted_fields.update(
            paligemma_variant=_set_lora_variant(model_config.paligemma_variant, enabled=checkpoint_uses_lora),
            action_expert_variant=_set_lora_variant(
                model_config.action_expert_variant,
                enabled=checkpoint_uses_lora,
            ),
        )

    logging.info(
        "Loading IQL checkpoint as policy-only model; critic and discriminator params will not be restored."
    )
    return dataclasses.replace(train_config, model=dataclasses.replace(model_config, **adjusted_fields))


def _maybe_adjust_pi0_model_for_checkpoint(train_config: _config.TrainConfig, params: Any) -> _config.TrainConfig:
    model_config = train_config.model
    if not isinstance(model_config, _pi0_config.Pi0Config):
        return train_config

    adjusted_fields: dict[str, Any] = {}
    checkpoint_split = _checkpoint_uses_split_discriminator(params)
    if checkpoint_split and not model_config.use_iql:
        raise ValueError(
            "Checkpoint contains split discriminator parameters but the selected config has `model.use_iql=False`. "
            "Use an IQL config (for example, `pi05_iql_libero_lora`)."
        )

    if checkpoint_split and not model_config.split_discriminator_head:
        logging.info(
            "Detected split discriminator checkpoint; overriding model config for serving with "
            "`split_discriminator_head=True`."
        )
        adjusted_fields.update(
            split_discriminator_head=True,
            instruction_discriminator_only_pretrain=False,
        )

    if (not checkpoint_split) and model_config.split_discriminator_head:
        logging.info(
            "Detected non-split discriminator checkpoint; overriding model config for serving with "
            "`split_discriminator_head=False`."
        )
        adjusted_fields.update(
            split_discriminator_head=False,
            instruction_discriminator_only_pretrain=False,
            action_relabeling_ratio_for_disc_l=0.0,
            joint_relabeling_ratio_for_disc_l=0.0,
            use_obs_action_similarity_as_weight=False,
            entropy_regularization_coef=0.0,
            actor_instruction_discriminator_topk_ratio=1.0,
            rel_disc_reward_weight=0.5,
            treat_instruction_only_as_unlabeled_for_disc_l=False,
            treat_action_only_as_unlabeled_for_disc_l=False,
            discriminator_entropy_exclude_unlabeled=False,
            discriminator_front_camera_zero_probability=0.0,
            discriminator_wrist_camera_zero_probability=0.0,
            discriminator_proprio_zero_probability=0.0,
        )

    checkpoint_uses_lora = _checkpoint_uses_lora(params)
    model_uses_lora = "lora" in model_config.paligemma_variant or "lora" in model_config.action_expert_variant
    if checkpoint_uses_lora != model_uses_lora:
        logging.info(
            "Detected checkpoint/model LoRA mismatch; overriding model config for serving with "
            "paligemma_variant=%s and action_expert_variant=%s.",
            _set_lora_variant(model_config.paligemma_variant, enabled=checkpoint_uses_lora),
            _set_lora_variant(model_config.action_expert_variant, enabled=checkpoint_uses_lora),
        )
        adjusted_fields.update(
            paligemma_variant=_set_lora_variant(model_config.paligemma_variant, enabled=checkpoint_uses_lora),
            action_expert_variant=_set_lora_variant(
                model_config.action_expert_variant,
                enabled=checkpoint_uses_lora,
            ),
        )

    if adjusted_fields:
        adjusted_model = dataclasses.replace(model_config, **adjusted_fields)
        return dataclasses.replace(train_config, model=adjusted_model)

    return train_config


def apply_checkpoint_normalization_config(
    data_config: _config.DataConfig,
    normalization_config: dict[str, object] | None,
) -> _config.DataConfig:
    if normalization_config is None:
        return data_config

    overrides: dict[str, Any] = {}
    if "action_norm_mode" in normalization_config:
        action_norm_mode = normalization_config["action_norm_mode"]
        if action_norm_mode != data_config.action_norm_mode:
            logging.warning(
                "Overriding data_config.action_norm_mode=%s with checkpoint action_norm_mode=%s.",
                data_config.action_norm_mode,
                action_norm_mode,
            )
        overrides["action_norm_mode"] = action_norm_mode

    if "use_quantile_norm" in normalization_config:
        use_quantile_norm = bool(normalization_config["use_quantile_norm"])
        if use_quantile_norm != data_config.use_quantile_norm:
            logging.warning(
                "Overriding data_config.use_quantile_norm=%s with checkpoint use_quantile_norm=%s.",
                data_config.use_quantile_norm,
                use_quantile_norm,
            )
        overrides["use_quantile_norm"] = use_quantile_norm

    if not overrides:
        return data_config
    return dataclasses.replace(data_config, **overrides)


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    load_policy_only: bool = False,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        load_policy_only: If True and the selected JAX config is an IQL pi0.5 config, restore only the actor
            parameters and instantiate a policy-only model. This avoids loading critic and discriminator branches
            during action-serving deployments.

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        use_policy_only_restore = (
            load_policy_only
            and isinstance(train_config.model, _pi0_config.Pi0Config)
            and train_config.model.use_iql
        )
        if use_policy_only_restore:
            params = _model.restore_params(
                checkpoint_dir / "params",
                dtype=jnp.bfloat16,
                key_filter=_is_iql_policy_param_key,
            )
            train_config = _make_pi0_policy_only_config(train_config, params)
            params = _remap_iql_policy_params(params)
        else:
            params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
            train_config = _maybe_adjust_pi0_model_for_checkpoint(train_config, params)
        model = train_config.model.load(params)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is not None:
        data_config = apply_checkpoint_normalization_config(
            data_config,
            _checkpoints.load_normalization_config(checkpoint_dir / "assets", data_config.asset_id),
        )
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"
    if is_pytorch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    return _policy.Policy(
        model,
        rng=None if is_pytorch else jax.random.key(seed),
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                key_norm_modes=_config.action_norm_key_modes(data_config),
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                key_norm_modes=_config.action_norm_key_modes(data_config),
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
