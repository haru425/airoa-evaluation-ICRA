import math
import logging
from typing import Any

import einops
from flax import traverse_util
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
import optax
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import instruction_discriminator as _instruction_discriminator

logger = logging.getLogger("openpi")

_IQL_IMAGE_ENCODER_NAMES = (
    "img_actor",
    "img_critic",
    "img_target_critic",
    "img_discriminator",
)


def maybe_replicate_shared_iql_image_encoder_params(
    params: at.Params,
    *,
    target_image_encoders: tuple[str, ...] = _IQL_IMAGE_ENCODER_NAMES,
    drop_shared_encoder: bool = True,
) -> at.Params:
    """Replicate legacy shared IQL vision encoder params into split encoder branches.

    Older IQL checkpoints store a single shared encoder under `PaliGemma/img/...`.
    Newer checkpoints store one encoder per branch under `PaliGemma/img_*...`.
    """

    flat_params = traverse_util.flatten_dict(params, sep="/")
    shared_prefix = "PaliGemma/img/"
    shared_items = {key: value for key, value in flat_params.items() if key.startswith(shared_prefix)}
    if not shared_items:
        return params

    upgraded = dict(flat_params)
    for encoder_name in target_image_encoders:
        branch_prefix = f"PaliGemma/{encoder_name}/"
        if any(key.startswith(branch_prefix) for key in upgraded):
            continue
        for key, value in shared_items.items():
            suffix = key[len(shared_prefix) :]
            upgraded[f"{branch_prefix}{suffix}"] = value

    if drop_shared_encoder:
        for key in shared_items:
            upgraded.pop(key, None)

    return traverse_util.unflatten_dict(upgraded, sep="/")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def assert_in_unit_interval(name: str, values: at.Array, eps: float = 1e-6) -> None:
    def _check(v):
        arr = np.asarray(v, dtype=np.float32)
        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
        if min_value < -eps or max_value > 1.0 + eps:
            raise ValueError(
                f"`{name}` must be in [0, 1]. Got min={min_value:.6f}, max={max_value:.6f}."
            )

    jax.debug.callback(_check, values)


def _compute_td3_bc_actor_loss(
    flow_matching_loss: at.Array,
    objective_value: at.Array,
    *,
    alpha: float,
    eps: float = 1e-6,
) -> tuple[at.Array, dict[str, at.Array]]:
    bc_loss = jnp.mean(jnp.asarray(flow_matching_loss, dtype=jnp.float32))
    objective_value = jnp.asarray(objective_value, dtype=jnp.float32)
    q_scale = jnp.mean(jnp.abs(jax.lax.stop_gradient(objective_value)))
    q_weight = jnp.asarray(alpha, dtype=objective_value.dtype) / jnp.maximum(
        q_scale,
        jnp.asarray(eps, dtype=objective_value.dtype),
    )
    q_loss = -q_weight * jnp.mean(objective_value)
    return bc_loss + q_loss, {
        "bc_loss": bc_loss,
        "q_loss": q_loss,
        "q_weight": q_weight,
    }


def _select_td3_bc_actor_q_value(q1_value: at.Array, q2_value: at.Array) -> at.Array:
    del q2_value
    return jnp.asarray(q1_value, dtype=jnp.float32)


def _combine_discriminator_branch_values(
    primary_value: at.Array,
    secondary_value: at.Array | None = None,
    *,
    secondary_weight: float | at.Array | None = None,
) -> at.Array:
    primary_value = jnp.asarray(primary_value, dtype=jnp.float32)
    if secondary_value is None:
        return primary_value
    secondary_value = jnp.asarray(secondary_value, dtype=jnp.float32)
    if secondary_weight is None:
        return primary_value + secondary_value
    secondary_weight = jnp.asarray(secondary_weight, dtype=jnp.float32)
    return (1.0 - secondary_weight) * primary_value + secondary_weight * secondary_value


def _discriminator_logits_to_reward(logits: at.Array) -> at.Array:
    return jax.nn.log_sigmoid(jnp.asarray(logits, dtype=jnp.float32))


def _steps_to_episode_end_to_binary_reward(
    steps_to_episode_end: at.Array,
    *,
    positive_reward_last_step_num: int,
) -> at.Array:
    steps_to_episode_end = jnp.asarray(steps_to_episode_end, dtype=jnp.int32)
    reward = (steps_to_episode_end < positive_reward_last_step_num).astype(jnp.float32)
    return reward.reshape(reward.shape + (1, 1))


def _compute_rwr_weights(
    reward: at.Array,
    *,
    temperature: float,
    max_weight: float | None,
) -> at.Array:
    reward = jnp.asarray(reward, dtype=jnp.float32)
    exp_reward = jnp.exp(reward * temperature)
    if max_weight is not None:
        exp_reward = jnp.minimum(exp_reward, max_weight)
    return exp_reward


def _project_actions_for_td3_bc_disc_actor(
    actions: at.Array,
    *,
    gripper_tanh_scale: float = 5.0,
) -> at.Array:
    """Project actor actions toward discriminator train-time support while preserving gradients."""
    projected_actions = jnp.clip(jnp.asarray(actions, dtype=jnp.float32), a_min=-1.0, a_max=1.0)
    if projected_actions.shape[-1] > 7:
        projected_actions = projected_actions.at[:, :, 7:].set(0.0)
    if projected_actions.shape[-1] > 6:
        projected_actions = projected_actions.at[:, :, 6].set(
            jnp.tanh(gripper_tanh_scale * projected_actions[:, :, 6])
        )
    return projected_actions


def _actor_instruction_filter_stats_default(batch_size: int) -> dict[str, at.Array]:
    return {
        "actor_disc_l_selected_ratio": jnp.array(1.0, dtype=jnp.float32),
        "actor_disc_l_selected_count": jnp.asarray(batch_size, dtype=jnp.float32),
        "actor_disc_l_score_threshold": jnp.array(jnp.nan, dtype=jnp.float32),
        "actor_disc_l_score_mean_all": jnp.array(jnp.nan, dtype=jnp.float32),
        "actor_disc_l_score_mean_selected": jnp.array(jnp.nan, dtype=jnp.float32),
        "actor_disc_l_score_mean_rejected": jnp.array(jnp.nan, dtype=jnp.float32),
    }


def _actor_instruction_filter_stats_nan() -> dict[str, at.Array]:
    nan = jnp.array(jnp.nan, dtype=jnp.float32)
    return {
        "actor_disc_l_selected_ratio": nan,
        "actor_disc_l_selected_count": nan,
        "actor_disc_l_score_threshold": nan,
        "actor_disc_l_score_mean_all": nan,
        "actor_disc_l_score_mean_selected": nan,
        "actor_disc_l_score_mean_rejected": nan,
    }


def _masked_mean_or_nan(values: at.Array, mask: at.Array) -> at.Array:
    values = jnp.asarray(values)
    mask = jnp.asarray(mask, dtype=values.dtype)
    count = jnp.sum(mask)
    safe_mean = jnp.sum(values * mask) / jnp.maximum(count, jnp.array(1.0, dtype=values.dtype))
    return jnp.where(
        count > 0,
        safe_mean,
        jnp.array(jnp.nan, dtype=values.dtype),
    )


def _take_optional_array(value: at.Array | None, indices: at.Array) -> at.Array | None:
    if value is None:
        return None
    return jnp.take(value, indices, axis=0)


def _take_optional_dict(value: dict[str, at.Array] | None, indices: at.Array) -> dict[str, at.Array] | None:
    if value is None:
        return None
    return {key: jnp.take(array, indices, axis=0) for key, array in value.items()}


def _take_observation_batch(observation: _model.Observation, indices: at.Array) -> _model.Observation:
    return _model.Observation(
        images={key: jnp.take(value, indices, axis=0) for key, value in observation.images.items()},
        image_masks={key: jnp.take(value, indices, axis=0) for key, value in observation.image_masks.items()},
        state=jnp.take(observation.state, indices, axis=0),
        tokenized_prompt=_take_optional_array(observation.tokenized_prompt, indices),
        tokenized_prompt_mask=_take_optional_array(observation.tokenized_prompt_mask, indices),
        token_ar_mask=_take_optional_array(observation.token_ar_mask, indices),
        token_loss_mask=_take_optional_array(observation.token_loss_mask, indices),
        next_images=_take_optional_dict(observation.next_images, indices),
        next_image_masks=_take_optional_dict(observation.next_image_masks, indices),
        next_state=_take_optional_array(observation.next_state, indices),
        done=_take_optional_array(observation.done, indices),
        steps_to_episode_end=_take_optional_array(observation.steps_to_episode_end, indices),
        relabeled_instruction=_take_optional_array(observation.relabeled_instruction, indices),
        relabeled_action=_take_optional_array(observation.relabeled_action, indices),
        left_right_flipped_image=_take_optional_array(observation.left_right_flipped_image, indices),
        left_right_flipped_action=_take_optional_array(observation.left_right_flipped_action, indices),
        relabeled_instruction_similarity=_take_optional_array(observation.relabeled_instruction_similarity, indices),
        relabeled_instruction_similarity_weight=_take_optional_array(
            observation.relabeled_instruction_similarity_weight,
            indices,
        ),
        actor_action_chunk_relabeled=_take_optional_array(observation.actor_action_chunk_relabeled, indices),
    )


def _filter_actor_batch_with_instruction_discriminator_scores(
    observation: _model.Observation,
    actions: _model.Actions,
    instruction_logits: at.Array,
    *,
    topk_ratio: float,
) -> tuple[_model.Observation, _model.Actions, at.Array, dict[str, at.Array]]:
    batch_size = actions.shape[0]
    selected_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
    filter_stats = _actor_instruction_filter_stats_default(batch_size)

    if topk_ratio >= 1.0:
        return observation, actions, selected_mask, filter_stats

    instruction_scores = jax.nn.sigmoid(jnp.asarray(instruction_logits, dtype=jnp.float32).reshape((batch_size,)))
    selected_count = max(1, math.ceil(batch_size * topk_ratio))
    topk_scores, topk_indices = jax.lax.top_k(instruction_scores, selected_count)
    selected_indices = jnp.sort(topk_indices)
    selected_mask = jnp.zeros((batch_size,), dtype=jnp.bool_).at[selected_indices].set(True)
    rejected_mask = jnp.logical_not(selected_mask)
    selected_scores = jnp.take(instruction_scores, selected_indices, axis=0)

    filtered_observation = _take_observation_batch(observation, selected_indices)
    filtered_actions = jnp.take(actions, selected_indices, axis=0)

    filter_stats = {
        "actor_disc_l_selected_ratio": jnp.asarray(selected_count / batch_size, dtype=jnp.float32),
        "actor_disc_l_selected_count": jnp.asarray(selected_count, dtype=jnp.float32),
        "actor_disc_l_score_threshold": jnp.min(topk_scores),
        "actor_disc_l_score_mean_all": jnp.mean(instruction_scores),
        "actor_disc_l_score_mean_selected": jnp.mean(selected_scores),
        "actor_disc_l_score_mean_rejected": _masked_mean_or_nan(
            instruction_scores,
            rejected_mask,
        ),
    }
    return filtered_observation, filtered_actions, selected_mask, filter_stats


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_iql = config.use_iql
        self.split_discriminator_head = bool(config.split_discriminator_head) if self.use_iql else False
        self.actor_instruction_discriminator_topk_ratio = (
            config.actor_instruction_discriminator_topk_ratio if self.use_iql else 1.0
        )
        self.iql_actor_loss_type = config.iql_actor_loss_type if self.use_iql else "awr"
        self.iql_td3_bc_alpha = config.iql_td3_bc_alpha
        self.use_binary_reward = bool(config.use_binary_reward) if self.use_iql else False
        self.positive_reward_last_step_num = int(config.positive_reward_last_step_num)
        self.rel_disc_reward_weight = float(config.rel_disc_reward_weight) if self.use_iql else 0.5

        print(f"paligemma variant: {config.paligemma_variant}")
        print(f"action expert variant: {config.action_expert_variant}")

        if self.use_iql:
            # Shared VLM (2B) experts.
            self.llm_idx_policy_prefix = 0
            self.llm_idx_critic_prefix = 1
            self.llm_idx_target_critic_prefix = 2
            self.llm_idx_discriminator_prefix = 3

            # Action-expert (300M) experts.
            self.llm_idx_actor_expert = 4
            self.llm_idx_q1_expert = 5
            self.llm_idx_q2_expert = 6
            self.llm_idx_q1_target_expert = 7
            self.llm_idx_q2_target_expert = 8
            self.llm_idx_value_expert = 9
            self.llm_idx_discriminator_expert = 10
            self.llm_idx_discriminator_instruction_expert = 11 if self.split_discriminator_head else None
            self.llm_num_experts = 12 if self.split_discriminator_head else 11
            share_2b_expert_weights, share_300m_expert_weights = config.get_iql_expert_weight_sharing()

            paligemma_config = _gemma.get_config(config.paligemma_variant)
            critic_paligemma_config = _gemma.get_config(config.paligemma_variant)
            target_critic_paligemma_config = _gemma.get_config(config.paligemma_variant)
            discriminator_paligemma_config = _gemma.get_config(config.paligemma_variant)
            action_expert_config = _gemma.get_config(config.action_expert_variant)
            q1_value_expert_config = _gemma.get_config(config.action_expert_variant)
            q2_value_expert_config = _gemma.get_config(config.action_expert_variant)
            q1t_value_expert_config = _gemma.get_config(config.action_expert_variant)
            q2t_value_expert_config = _gemma.get_config(config.action_expert_variant)
            value_expert_config = _gemma.get_config(config.action_expert_variant)  # TODO (kondoh) should be shared with Qs?
            discriminator_expert_config = _gemma.get_config(config.action_expert_variant)
            iql_configs = [
                paligemma_config,
                critic_paligemma_config,
                target_critic_paligemma_config,
                discriminator_paligemma_config,
                action_expert_config,
                q1_value_expert_config,
                q2_value_expert_config,
                q1t_value_expert_config,
                q2t_value_expert_config,
                value_expert_config,
                discriminator_expert_config,
            ]
            if self.split_discriminator_head:
                discriminator_instruction_expert_config = _gemma.get_config(config.action_expert_variant)
                iql_configs.append(discriminator_instruction_expert_config)

            # TODO: rewrite gemma in NNX. For now, use bridge.
            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=iql_configs,
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                    share_expert_weights=(share_2b_expert_weights, share_300m_expert_weights),
                )
            )
            use_adarms = [False] * len(iql_configs)
            use_adarms[self.llm_idx_actor_expert] = True
            llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
        else:
            paligemma_config = _gemma.get_config(config.paligemma_variant)
            action_expert_config = _gemma.get_config(config.action_expert_variant)
            # TODO: rewrite gemma in NNX. For now, use bridge.
            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                )
            )
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        def _make_image_encoder():
            img = nnx_bridge.ToNNX(
                _siglip.Module(
                    num_classes=paligemma_config.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                )
            )
            img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
            return img

        if self.use_iql:
            self.PaliGemma = nnx.Dict(
                llm=llm,
                img_actor=_make_image_encoder(),
                img_critic=_make_image_encoder(),
                img_target_critic=_make_image_encoder(),
                img_discriminator=_make_image_encoder(),
            )
        else:
            self.PaliGemma = nnx.Dict(llm=llm, img=_make_image_encoder())
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            assert not config.use_iql, "IQL is only supported with pi0.5"

            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True
                
        if self.use_iql:
            self.q1_out_proj = nnx.Linear(action_expert_config.width, 1, rngs=rngs)
            self.q2_out_proj = nnx.Linear(action_expert_config.width, 1, rngs=rngs)
            self.q1t_out_proj = nnx.Linear(action_expert_config.width, 1, rngs=rngs)
            self.q2t_out_proj = nnx.Linear(action_expert_config.width, 1, rngs=rngs)
            self.v_out_proj = nnx.Linear(action_expert_config.width, 1, rngs=rngs)
            
            self.discriminator_out_proj = nnx.Linear(action_expert_config.width, 1, rngs=rngs)
            if self.split_discriminator_head:
                self.discriminator_out_proj_instruction = nnx.Linear(action_expert_config.width, 1, rngs=rngs)

            self.action_proj_discriminator = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            if self.split_discriminator_head:
                self.action_proj_discriminator_instruction = nnx.Linear(
                    config.action_dim, action_expert_config.width, rngs=rngs
                )
            self.action_proj_q1 = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_proj_q2 = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_proj_q1t = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_proj_q2t = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

            self.q1_token = nnx.Param(jnp.zeros((action_expert_config.width,), dtype=config.dtype), name="q1_token")
            self.q2_token = nnx.Param(jnp.zeros((action_expert_config.width,), dtype=config.dtype), name="q2_token")
            self.q1t_token = nnx.Param(jnp.zeros((action_expert_config.width,), dtype=config.dtype), name="q1t_token")
            self.q2t_token = nnx.Param(jnp.zeros((action_expert_config.width,), dtype=config.dtype), name="q2t_token")
            self.v_token = nnx.Param(jnp.zeros((action_expert_config.width,), dtype=config.dtype), name="v_token")
            self.discriminator_token = nnx.Param(jnp.zeros((action_expert_config.width,), dtype=config.dtype), name="discriminator_token")
            if self.split_discriminator_head:
                self.discriminator_token_instruction = nnx.Param(
                    jnp.zeros((action_expert_config.width,), dtype=config.dtype), name="discriminator_token_instruction"
                )

            self.iql_critic_tau = config.iql_critic_tau
            self.iql_actor_temperature = config.iql_actor_temperature
            self.iql_exp_adv_max = config.iql_exp_adv_max
            self.critic_gamma = config.critic_gamma

            self.diffusion_num_steps = 10

            self.action_relabeling_ratio_for_disc_l = config.action_relabeling_ratio_for_disc_l
            self.joint_relabeling_ratio_for_disc_l = config.joint_relabeling_ratio_for_disc_l
            self.flipped_image_action_ratio_for_disc_l = config.flipped_image_action_ratio_for_disc_l
            self.flipped_image_only_ratio_for_disc_l = config.flipped_image_only_ratio_for_disc_l
            self.flipped_action_only_ratio_for_disc_l = config.flipped_action_only_ratio_for_disc_l
            self.use_obs_action_similarity_as_weight = config.use_obs_action_similarity_as_weight
            self.entropy_regularization_coef = config.entropy_regularization_coef
            self.treat_instruction_only_as_unlabeled_for_disc_l = (
                config.treat_instruction_only_as_unlabeled_for_disc_l
            )
            self.treat_action_only_as_unlabeled_for_disc_l = config.treat_action_only_as_unlabeled_for_disc_l
            self.discriminator_entropy_exclude_unlabeled = config.discriminator_entropy_exclude_unlabeled
            self.discriminator_front_camera_zero_probability = config.discriminator_front_camera_zero_probability
            self.discriminator_wrist_camera_zero_probability = config.discriminator_wrist_camera_zero_probability
            self.discriminator_proprio_zero_probability = config.discriminator_proprio_zero_probability

    def _make_iql_llm_inputs(self, updates: dict[int, at.Array]) -> list[at.Array | None]:
        if not self.use_iql:
            raise ValueError("IQL LLM inputs are only available when `use_iql=True`.")
        inputs: list[at.Array | None] = [None] * self.llm_num_experts
        for idx, value in updates.items():
            inputs[idx] = value
        return inputs

    def _make_iql_adarms_cond(self, updates: dict[int, at.Array] | None = None) -> list[at.Array | None]:
        if not self.use_iql:
            raise ValueError("IQL adarms conditions are only available when `use_iql=True`.")
        cond: list[at.Array | None] = [None] * self.llm_num_experts
        if updates is not None:
            for idx, value in updates.items():
                cond[idx] = value
        return cond

    def _image_encoder_for_model_type(self, model_type: str):
        if not self.use_iql:
            return self.PaliGemma.img
        if model_type in ("actor",):
            return self.PaliGemma.img_actor
        if model_type in ("discriminator", "discriminator_instruction"):
            return self.PaliGemma.img_discriminator
        if model_type in ("q1", "q2", "q1q2", "v"):
            return self.PaliGemma.img_critic
        if model_type in ("q1t", "q2t", "q1tq2t"):
            return self.PaliGemma.img_target_critic
        raise ValueError(f"Unsupported model_type for image encoder selection: {model_type}")

    @at.typecheck
    def embed_prefix(
        self,
        obs: _model.Observation,
        next_or_current: str = "current",
        model_type: str = "actor",
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        image_encoder = self._image_encoder_for_model_type(model_type)
        image_source = obs.images if next_or_current == "current" else obs.next_images
        image_mask_source = (
            obs.image_masks
            if next_or_current == "current" or obs.next_image_masks is None
            else obs.next_image_masks
        )
        # embed images
        for name in obs.images:
            image_tokens, _ = image_encoder(image_source[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    image_mask_source[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: at.Float[Any, "b _ _"] | None = None,
        timestep: at.Float[Any, " b"] | None = None,
        model_type: str = "actor",
        next_or_current: str = "current",
        actions: at.Float[Any, "b ah ad"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(
                obs.state if next_or_current == "current" else obs.next_state
            )[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]
        
        if self.use_iql and model_type in ["discriminator", "discriminator_instruction", "q1", "q2", "q1t", "q2t"]:
            assert actions is not None, "actions must be provided when using IQL for discriminator or Q functions"
            if model_type == "discriminator":
                action_token = self.action_proj_discriminator(actions)
            elif model_type == "discriminator_instruction":
                if not self.split_discriminator_head:
                    raise ValueError("`model_type='discriminator_instruction'` requires `split_discriminator_head=True`.")
                action_token = self.action_proj_discriminator_instruction(actions)
            elif model_type == "q1":
                action_token = self.action_proj_q1(actions)
            elif model_type == "q2":
                action_token = self.action_proj_q2(actions)
            elif model_type == "q1t":
                action_token = self.action_proj_q1t(actions)
            elif model_type == "q2t":
                action_token = self.action_proj_q2t(actions)
            else:
                raise Exception(f"model_type: {model_type}")
            
            tokens.append(action_token)
            input_mask.append(jnp.ones(actions.shape[:2], dtype=jnp.bool_))
            # image/language/state inputs do not attend to action tokens
            ar_mask += [True] * action_token.shape[1]
        else:
            assert actions is None, "actions should not be provided when not using IQL for discriminator or Q functions"

        batch_size = obs.state.shape[0]
        
        if model_type == "actor":
            action_tokens = self.action_in_proj(noisy_actions)
            out_features = self.action_in_proj.out_features
        elif model_type == "discriminator":
            action_tokens = jnp.broadcast_to(
                self.discriminator_token.value[None, None, :],
                (batch_size, 1, self.discriminator_token.value.shape[0]),
            )
            out_features = self.discriminator_token.value.shape[0]
        elif model_type == "discriminator_instruction":
            if not self.split_discriminator_head:
                raise ValueError("`model_type='discriminator_instruction'` requires `split_discriminator_head=True`.")
            action_tokens = jnp.broadcast_to(
                self.discriminator_token_instruction.value[None, None, :],
                (batch_size, 1, self.discriminator_token_instruction.value.shape[0]),
            )
            out_features = self.discriminator_token_instruction.value.shape[0]
        elif model_type == "q1":
            action_tokens = jnp.broadcast_to(
                self.q1_token.value[None, None, :],
                (batch_size, 1, self.q1_token.value.shape[0]),
            )
            out_features = self.q1_token.value.shape[0]
        elif model_type == "q2":
            action_tokens = jnp.broadcast_to(
                self.q2_token.value[None, None, :],
                (batch_size, 1, self.q2_token.value.shape[0]),
            )
            out_features = self.q2_token.value.shape[0]
        elif model_type == "q1t":
            action_tokens = jnp.broadcast_to(
                self.q1t_token.value[None, None, :],
                (batch_size, 1, self.q1t_token.value.shape[0]),
            )
            out_features = self.q1t_token.value.shape[0]
        elif model_type == "q2t":
            action_tokens = jnp.broadcast_to(
                self.q2t_token.value[None, None, :],
                (batch_size, 1, self.q2t_token.value.shape[0]),
            )
            out_features = self.q2t_token.value.shape[0]
        elif model_type == "v":
            action_tokens = jnp.broadcast_to(
                self.v_token.value[None, None, :],
                (batch_size, 1, self.v_token.value.shape[0]),
            )
            out_features = self.v_token.value.shape[0]
        else:
            raise Exception(f"model_type: {model_type}")
        s = action_tokens.shape[1]
        
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=s)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (s - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @staticmethod
    def _count_per_sample_true(value: at.Array | None, batch_size: int) -> at.Array:
        if value is None:
            return jnp.array(-1, dtype=jnp.int32)
        mask = jnp.asarray(value).astype(jnp.bool_)
        mask = mask.reshape((batch_size, -1))
        mask = jnp.any(mask, axis=1)
        return jnp.sum(mask.astype(jnp.int32))

    def _sample_actions_for_actor_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        preprocessed_observation: bool = False,
    ) -> _model.Actions:
        if not preprocessed_observation:
            observation = _model.preprocess_observation(None, observation, train=False)

        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, model_type="actor")
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        if self.use_iql:
            prefix_llm_inputs = self._make_iql_llm_inputs({self.llm_idx_policy_prefix: prefix_tokens})
            _, kv_cache = self.PaliGemma.llm(prefix_llm_inputs, mask=prefix_attn_mask, positions=positions)
        else:
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        dt = -1.0 / num_steps

        def step(x_t, step_idx):
            time = jnp.asarray(1.0 + dt * step_idx, dtype=noise.dtype)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_step_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_step_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            if self.use_iql:
                step_inputs = self._make_iql_llm_inputs({self.llm_idx_actor_expert: suffix_tokens})
                step_adarms_cond = self._make_iql_adarms_cond({self.llm_idx_actor_expert: adarms_cond})
                outputs, _ = self.PaliGemma.llm(
                    step_inputs,
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=step_adarms_cond,
                )
                prefix_out = outputs[self.llm_idx_policy_prefix]
                suffix_out = outputs[self.llm_idx_actor_expert]
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )

            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, None

        step_indices = jnp.arange(int(num_steps), dtype=jnp.int32)
        x_0, _ = jax.lax.scan(step, noise, step_indices)
        return x_0

    def _actor_q_value_grad_block_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(
            nnx.Param,
            nnx.Any(
                nnx_utils.PathRegex(".*img_critic.*"),
                nnx_utils.PathRegex(r".*llm.*(_1(?![0-9])|_5(?![0-9])|_6(?![0-9])).*"),
                nnx_utils.PathRegex(".*llm.*embedder.*"),
                nnx_utils.PathRegex(".*action_proj_q1.*"),
                nnx_utils.PathRegex(".*action_proj_q2.*"),
                nnx_utils.PathRegex(".*q1_out_proj.*"),
                nnx_utils.PathRegex(".*q2_out_proj.*"),
                nnx_utils.PathRegex(".*q1_token.*"),
                nnx_utils.PathRegex(".*q2_token.*"),
            ),
        )

    def _actor_discriminator_grad_block_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(
            nnx.Param,
            nnx.Any(
                nnx_utils.PathRegex(".*img_discriminator.*"),
                nnx_utils.PathRegex(r".*llm.*(_3(?![0-9])|_10(?![0-9])|_11(?![0-9])).*"),
                nnx_utils.PathRegex(".*llm.*embedder.*"),
                nnx_utils.PathRegex(".*action_proj_discriminator.*"),
                nnx_utils.PathRegex(".*action_proj_discriminator_instruction.*"),
                nnx_utils.PathRegex(".*discriminator_out_proj.*"),
                nnx_utils.PathRegex(".*discriminator_out_proj_instruction.*"),
                nnx_utils.PathRegex(".*discriminator_token.*"),
                nnx_utils.PathRegex(".*discriminator_token_instruction.*"),
            ),
        )

    def _forward_q_values_for_actor_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        num_steps: int,
        preprocessed_observation: bool = False,
    ) -> tuple[at.Array, at.Array]:
        graphdef, state = nnx.split(self)
        frozen_state = nnx_utils.state_map(
            state,
            self._actor_q_value_grad_block_filter(),
            lambda p: p.replace(jax.lax.stop_gradient(p.value)),
        )
        frozen_model = nnx.merge(graphdef, frozen_state)
        return frozen_model.forward_values(
            rng,
            observation,
            actions,
            num_steps=num_steps,
            model_type="q1q2",
            preprocessed_observation=preprocessed_observation,
        )

    def _forward_discriminator_values_for_actor_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        num_steps: int,
        preprocessed_observation: bool = False,
    ) -> tuple[at.Array, at.Array | None]:
        graphdef, state = nnx.split(self)
        frozen_state = nnx_utils.state_map(
            state,
            self._actor_discriminator_grad_block_filter(),
            lambda p: p.replace(jax.lax.stop_gradient(p.value)),
        )
        frozen_model = nnx.merge(graphdef, frozen_state)
        discriminator_value_action = frozen_model.forward_values(
            rng,
            observation,
            actions,
            num_steps=num_steps,
            model_type="discriminator",
            preprocessed_observation=preprocessed_observation,
        )
        discriminator_value_instruction = None
        if self.split_discriminator_head:
            discriminator_value_instruction = frozen_model.forward_values(
                rng,
                observation,
                actions,
                num_steps=num_steps,
                model_type="discriminator_instruction",
                preprocessed_observation=preprocessed_observation,
            )
        return discriminator_value_action, discriminator_value_instruction

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_size = actions.shape[0]
        actor_instruction_filter_stats = _actor_instruction_filter_stats_default(batch_size)

        if self.use_iql and self.actor_instruction_discriminator_topk_ratio < 1.0:
            instruction_logits = self.forward_values(
                rng,
                observation,
                actions,
                num_steps=self.diffusion_num_steps,
                model_type="discriminator_instruction",
                preprocessed_observation=True,
            )
            instruction_logits = jax.lax.stop_gradient(instruction_logits)
            observation, actions, _, actor_instruction_filter_stats = (
                _filter_actor_batch_with_instruction_discriminator_scores(
                    observation,
                    actions,
                    instruction_logits,
                    topk_ratio=self.actor_instruction_discriminator_topk_ratio,
                )
            )
            batch_size = actions.shape[0]

        # if train and self.use_iql:
        #     relabeled_instruction_count = self._count_per_sample_true(observation.relabeled_instruction, batch_size)
        #     actor_action_chunk_relabeled_count = self._count_per_sample_true(
        #         observation.actor_action_chunk_relabeled, batch_size
        #     )
        #     jax.debug.print(
        #         "[IQL DEBUG][actor] batch={batch} relabeled_instruction={instruction_relabeled}/{batch} "
        #         "actor_action_chunk_relabeled={actor_action_relabeled}/{batch}",
        #         batch=batch_size,
        #         instruction_relabeled=relabeled_instruction_count,
        #         actor_action_relabeled=actor_action_chunk_relabeled_count,
        #         ordered=True,
        #     )

        batch_shape = actions.shape[:-2] # actions: (b, h, d)
        noise = jax.random.normal(noise_rng, actions.shape) # (b, h, d)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001 # (b,)
        time_expanded = time[..., None, None] # (b, 1, 1)
        x_t = time_expanded * noise + (1 - time_expanded) * actions # (b, h, d)
        u_t = noise - actions # (b, h, d)

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, model_type="actor")
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        if self.use_iql:
            llm_inputs = self._make_iql_llm_inputs(
                {
                    self.llm_idx_policy_prefix: prefix_tokens,
                    self.llm_idx_actor_expert: suffix_tokens,
                }
            )
            adarms_cond_list = self._make_iql_adarms_cond({self.llm_idx_actor_expert: adarms_cond})
            outputs, _ = self.PaliGemma.llm(
                llm_inputs,
                mask=attn_mask,
                positions=positions,
                adarms_cond=adarms_cond_list,
            )
            prefix_out = outputs[self.llm_idx_policy_prefix]
            suffix_out = outputs[self.llm_idx_actor_expert]
        else:
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
        # prefix_out: (b, 968, 2048), suffix_out: (b, h, 1024)
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :]) # (b, h, d)

        if not self.use_iql:
            return jnp.mean(jnp.square(v_t - u_t)), {}
        else:
            flow_matching_loss = jnp.square(v_t - u_t).mean(axis=(1,2))

            if self.iql_actor_loss_type in ("td3_bc", "td3_bc_disc"):
                policy_rng = jax.random.fold_in(rng, 1)
                policy_actions = self._sample_actions_for_actor_loss(
                    policy_rng,
                    observation,
                    num_steps=self.diffusion_num_steps,
                    preprocessed_observation=True,
                )
                q1_value = jnp.array(jnp.nan, dtype=jnp.float32)
                q2_value = jnp.array(jnp.nan, dtype=jnp.float32)
                td3_bc_q_value = jnp.array(jnp.nan, dtype=jnp.float32)
                td3_bc_disc_value = jnp.array(jnp.nan, dtype=jnp.float32)

                if self.iql_actor_loss_type == "td3_bc":
                    q1_value_arr, q2_value_arr = self._forward_q_values_for_actor_loss(
                        rng,
                        observation,
                        policy_actions,
                        num_steps=self.diffusion_num_steps,
                        preprocessed_observation=True,
                    )
                    q1_value = q1_value_arr.squeeze(axis=(1, 2))
                    q2_value = q2_value_arr.squeeze(axis=(1, 2))
                    actor_objective_value = _select_td3_bc_actor_q_value(q1_value, q2_value)
                    td3_bc_q_value = jnp.mean(actor_objective_value)
                else:
                    policy_actions = _project_actions_for_td3_bc_disc_actor(policy_actions)
                    discriminator_logits_action, discriminator_logits_instruction = (
                        self._forward_discriminator_values_for_actor_loss(
                            rng,
                            observation,
                            policy_actions,
                            num_steps=self.diffusion_num_steps,
                            preprocessed_observation=True,
                        )
                    )
                    discriminator_value_action = _discriminator_logits_to_reward(discriminator_logits_action).squeeze(
                        axis=(1, 2)
                    )
                    discriminator_value_instruction = None
                    if discriminator_logits_instruction is not None:
                        discriminator_value_instruction = _discriminator_logits_to_reward(
                            discriminator_logits_instruction
                        ).squeeze(axis=(1, 2))
                    actor_objective_value = _combine_discriminator_branch_values(
                        discriminator_value_action,
                        discriminator_value_instruction,
                    )
                    td3_bc_disc_value = jnp.mean(actor_objective_value)

                loss, td3_bc_stats = _compute_td3_bc_actor_loss(
                    flow_matching_loss,
                    actor_objective_value,
                    alpha=self.iql_td3_bc_alpha,
                )

                info = {
                    "exp_adv": jnp.nan,
                    "loss": td3_bc_stats["bc_loss"],
                    "q1_t_value": jnp.nan,
                    "q2_t_value": jnp.nan,
                    "q_t_value": jnp.nan,
                    "value": jnp.nan,
                    "gt_action": jnp.mean(actions),

                    "loss_value": jnp.nan,
                    "loss_q_value": jnp.nan,
                    "q1_value": jnp.mean(q1_value),
                    "q2_value": jnp.mean(q2_value),
                    "q_value_relabeled_instruction": jnp.nan,
                    "q_value_non_relabeled_instruction": jnp.nan,
                    "target_q_value": jnp.nan,
                    "next_value": jnp.nan,
                    "rewards": jnp.nan,
                    "predicted_actions": jnp.mean(policy_actions),
                    "discriminator_predicted_probs": jnp.nan,
                    "discriminator_predicted_probs_action": jnp.nan,
                    "discriminator_predicted_probs_instruction": jnp.nan,
                    "discriminator_target": jnp.nan,
                    "discriminator_accuracy": jnp.nan,
                    "discriminator_accuracy_action": jnp.nan,
                    "discriminator_accuracy_instruction": jnp.nan,
                    "discriminator_loss_action": jnp.nan,
                    "discriminator_loss_instruction": jnp.nan,
                    "discriminator_entropy": jnp.nan,
                    "discriminator_entropy_bonus": jnp.nan,
                    "td3_bc_bc_loss": td3_bc_stats["bc_loss"],
                    "td3_bc_q_loss": td3_bc_stats["q_loss"],
                    "td3_bc_lambda": td3_bc_stats["q_weight"],
                    "td3_bc_q_value": td3_bc_q_value,
                    "td3_bc_disc_value": td3_bc_disc_value,
                    **actor_instruction_filter_stats,
                }
                return loss, info

            if self.iql_actor_loss_type == "rwr":
                predicted_logits_action = self.forward_values(
                    rng,
                    observation,
                    actions,
                    num_steps=self.diffusion_num_steps,
                    model_type="discriminator",
                    preprocessed_observation=True,
                )
                predicted_logits_action = jax.lax.stop_gradient(predicted_logits_action)
                reward_action = _discriminator_logits_to_reward(predicted_logits_action)
                reward_instruction = None
                predicted_probs = jax.nn.sigmoid(predicted_logits_action)
                predicted_probs_action_mean = jnp.nan
                predicted_probs_instruction_mean = jnp.nan

                if self.split_discriminator_head:
                    predicted_logits_instruction = self.forward_values(
                        rng,
                        observation,
                        actions,
                        num_steps=self.diffusion_num_steps,
                        model_type="discriminator_instruction",
                        preprocessed_observation=True,
                    )
                    predicted_logits_instruction = jax.lax.stop_gradient(predicted_logits_instruction)
                    reward_instruction = _discriminator_logits_to_reward(predicted_logits_instruction)
                    reward = _combine_discriminator_branch_values(reward_action, reward_instruction)
                    predicted_probs = jnp.nan
                    predicted_probs_action_mean = jnp.mean(jax.nn.sigmoid(predicted_logits_action))
                    predicted_probs_instruction_mean = jnp.mean(jax.nn.sigmoid(predicted_logits_instruction))
                else:
                    reward = reward_action
                    predicted_probs = jnp.mean(predicted_probs)

                reward = reward.squeeze(axis=(1, 2))
                exp_reward = _compute_rwr_weights(
                    reward,
                    temperature=self.iql_actor_temperature,
                    max_weight=self.iql_exp_adv_max,
                )
                loss = jnp.mean(exp_reward * flow_matching_loss)

                info = {
                    "exp_adv": jnp.mean(exp_reward),
                    "loss": jnp.mean(flow_matching_loss),
                    "q1_t_value": jnp.nan,
                    "q2_t_value": jnp.nan,
                    "q_t_value": jnp.nan,
                    "value": jnp.nan,
                    "gt_action": jnp.mean(actions),

                    "loss_value": jnp.nan,
                    "loss_q_value": jnp.nan,
                    "q1_value": jnp.nan,
                    "q2_value": jnp.nan,
                    "q_value_relabeled_instruction": jnp.nan,
                    "q_value_non_relabeled_instruction": jnp.nan,
                    "target_q_value": jnp.nan,
                    "next_value": jnp.nan,
                    "rewards": jnp.mean(reward),
                    "predicted_actions": jnp.nan,
                    "discriminator_predicted_probs": predicted_probs,
                    "discriminator_predicted_probs_action": predicted_probs_action_mean,
                    "discriminator_predicted_probs_instruction": predicted_probs_instruction_mean,
                    "discriminator_target": jnp.nan,
                    "discriminator_accuracy": jnp.nan,
                    "discriminator_accuracy_action": jnp.nan,
                    "discriminator_accuracy_instruction": jnp.nan,
                    "discriminator_loss_action": jnp.nan,
                    "discriminator_loss_instruction": jnp.nan,
                    "discriminator_entropy": jnp.nan,
                    "discriminator_entropy_bonus": jnp.nan,
                    "td3_bc_bc_loss": jnp.nan,
                    "td3_bc_q_loss": jnp.nan,
                    "td3_bc_lambda": jnp.nan,
                    "td3_bc_q_value": jnp.nan,
                    "td3_bc_disc_value": jnp.nan,
                    **actor_instruction_filter_stats,
                }
                return loss, info

            if self.iql_actor_loss_type != "awr":
                raise ValueError(f"Unsupported `iql_actor_loss_type`: {self.iql_actor_loss_type}")

            q1_t_value, q2_t_value = self.forward_values(
                rng,
                observation,
                actions,
                num_steps=self.diffusion_num_steps,
                model_type="q1tq2t",
                preprocessed_observation=True,
            )
            q1_t_value = jax.lax.stop_gradient(q1_t_value)
            q2_t_value = jax.lax.stop_gradient(q2_t_value)
            q_value = jnp.minimum(q1_t_value, q2_t_value)

            value = self.forward_values(
                rng,
                observation,
                actions,
                num_steps=self.diffusion_num_steps,
                model_type="v",
                preprocessed_observation=True,
            )
            value = jax.lax.stop_gradient(value)

            exp_adv = jnp.exp((q_value - value) * self.iql_actor_temperature)
            if self.iql_exp_adv_max is not None:
                exp_adv = jnp.minimum(exp_adv, self.iql_exp_adv_max)

            exp_adv = exp_adv.squeeze(axis=(1,2))

            loss = jnp.mean(exp_adv * flow_matching_loss)

            info = {
                "exp_adv": jnp.mean(exp_adv),
                "loss": jnp.mean(flow_matching_loss),
                "q1_t_value": jnp.mean(q1_t_value),
                "q2_t_value": jnp.mean(q2_t_value),
                "q_t_value": jnp.mean(q_value),
                "value": jnp.mean(value),
                "gt_action": jnp.mean(actions),

                "loss_value": jnp.nan,
                "loss_q_value": jnp.nan,
                "q1_value": jnp.nan,
                "q2_value": jnp.nan,
                "q_value_relabeled_instruction": jnp.nan,
                "q_value_non_relabeled_instruction": jnp.nan,
                "target_q_value": jnp.nan,
                "next_value": jnp.nan,
                "rewards": jnp.nan,
                "predicted_actions": jnp.nan,
                "discriminator_predicted_probs": jnp.nan,
                "discriminator_predicted_probs_action": jnp.nan,
                "discriminator_predicted_probs_instruction": jnp.nan,
                "discriminator_target": jnp.nan,
                "discriminator_accuracy": jnp.nan,
                "discriminator_accuracy_action": jnp.nan,
                "discriminator_accuracy_instruction": jnp.nan,
                "discriminator_loss_action": jnp.nan,
                "discriminator_loss_instruction": jnp.nan,
                "discriminator_entropy": jnp.nan,
                "discriminator_entropy_bonus": jnp.nan,
                "td3_bc_bc_loss": jnp.nan,
                "td3_bc_q_loss": jnp.nan,
                "td3_bc_lambda": jnp.nan,
                "td3_bc_q_value": jnp.nan,
                "td3_bc_disc_value": jnp.nan,
                **actor_instruction_filter_stats,
            }

            return loss, info

    def compute_loss_critic(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_size = actions.shape[0]

        # Calculate value loss
        q1_t_value, q2_t_value = self.forward_values(
            rng,
            observation,
            actions,
            num_steps=self.diffusion_num_steps,
            model_type="q1tq2t",
            preprocessed_observation=True,
        )
        q1_t_value = jax.lax.stop_gradient(q1_t_value)
        q2_t_value = jax.lax.stop_gradient(q2_t_value)
        target_value = jnp.minimum(q1_t_value, q2_t_value)

        predicted_value = self.forward_values(
            rng,
            observation,
            actions,
            num_steps=self.diffusion_num_steps,
            model_type="v",
            preprocessed_observation=True,
        )

        error = (target_value - predicted_value)
        
        weight = jnp.where(error > 0, self.iql_critic_tau, 1 - self.iql_critic_tau)
        loss_value = jnp.mean(weight * jnp.square(error))

        # Calculate Q loss
        next_value = self.forward_values(
            rng,
            observation,
            actions,
            num_steps=self.diffusion_num_steps,
            model_type="v",
            next_or_current="next",
            preprocessed_observation=True,
        )
        next_value = jax.lax.stop_gradient(next_value)
        
        dones = observation.done
        dones = dones.reshape(dones.shape + (1, 1))

        if getattr(self, "use_binary_reward", False):
            if observation.steps_to_episode_end is None:
                raise ValueError("`steps_to_episode_end` is required when `use_binary_reward=True`.")
            rewards = _steps_to_episode_end_to_binary_reward(
                observation.steps_to_episode_end,
                positive_reward_last_step_num=self.positive_reward_last_step_num,
            )
            predicted_probs = jnp.nan
            predicted_probs_action_mean = jnp.nan
            predicted_probs_instruction_mean = jnp.nan
        elif self.split_discriminator_head:
            predicted_logits_action = self.forward_values(
                rng,
                observation,
                actions,
                num_steps=self.diffusion_num_steps,
                model_type="discriminator",
                preprocessed_observation=True,
            )
            predicted_logits_instruction = self.forward_values(
                rng,
                observation,
                actions,
                num_steps=self.diffusion_num_steps,
                model_type="discriminator_instruction",
                preprocessed_observation=True,
            )
            predicted_logits_action = jax.lax.stop_gradient(predicted_logits_action)
            predicted_logits_instruction = jax.lax.stop_gradient(predicted_logits_instruction)
            rewards = _combine_discriminator_branch_values(
                _discriminator_logits_to_reward(predicted_logits_action),
                _discriminator_logits_to_reward(predicted_logits_instruction),
                secondary_weight=getattr(self, "rel_disc_reward_weight", 0.5),
            )
            predicted_probs_action = jax.nn.sigmoid(predicted_logits_action)
            predicted_probs_instruction = jax.nn.sigmoid(predicted_logits_instruction)
            predicted_probs = jnp.nan
            predicted_probs_action_mean = jnp.mean(predicted_probs_action)
            predicted_probs_instruction_mean = jnp.mean(predicted_probs_instruction)
        else:
            predicted_logits = self.forward_values(
                rng,
                observation,
                actions,
                num_steps=self.diffusion_num_steps,
                model_type="discriminator",
                preprocessed_observation=True,
            )
            predicted_logits = jax.lax.stop_gradient(predicted_logits)
            predicted_probs = jax.nn.sigmoid(predicted_logits)
            rewards = _discriminator_logits_to_reward(predicted_logits)
            predicted_probs_action_mean = jnp.nan
            predicted_probs_instruction_mean = jnp.nan
        target_q_value = rewards + self.critic_gamma * (1 - dones) * next_value
        q1_value, q2_value = self.forward_values(
            rng,
            observation,
            actions,
            num_steps=self.diffusion_num_steps,
            model_type="q1q2",
            preprocessed_observation=True,
        )
        loss_q_value = jnp.mean(jnp.square(q1_value - target_q_value)) + jnp.mean(jnp.square(q2_value - target_q_value))
        q_value = jnp.minimum(q1_value, q2_value)

        relabeled_instruction_q_value = jnp.nan
        non_relabeled_instruction_q_value = jnp.nan
        relabeled_instruction_count = jnp.array(-1, dtype=jnp.int32)
        if observation.relabeled_instruction is not None:
            relabeled_instruction = jnp.asarray(observation.relabeled_instruction).astype(jnp.bool_)
            relabeled_instruction = relabeled_instruction.reshape((batch_size, -1))
            relabeled_instruction = jnp.any(relabeled_instruction, axis=1)
            relabeled_instruction_count = jnp.sum(relabeled_instruction.astype(jnp.int32))

            q_value_per_sample = q_value.reshape((batch_size, -1)).mean(axis=1)
            relabeled_instruction_q_value = _masked_mean_or_nan(q_value_per_sample, relabeled_instruction)
            non_relabeled_instruction_q_value = _masked_mean_or_nan(q_value_per_sample, ~relabeled_instruction)

        actor_action_chunk_relabeled_count = self._count_per_sample_true(
            observation.actor_action_chunk_relabeled, batch_size
        )
        # if train and self.use_iql:
        #     jax.debug.print(
        #         "[IQL DEBUG][critic] batch={batch} relabeled_instruction={instruction_relabeled}/{batch} "
        #         "actor_action_chunk_relabeled={actor_action_relabeled}/{batch}",
        #         batch=batch_size,
        #         instruction_relabeled=relabeled_instruction_count,
        #         actor_action_relabeled=actor_action_chunk_relabeled_count,
        #         ordered=True,
        #     )

        info = {
            "loss_value": loss_value,
            "loss_q_value": loss_q_value,
            "q1_t_value": jnp.mean(q1_t_value),
            "q2_t_value": jnp.mean(q2_t_value),
            "q_t_value": jnp.mean(target_value),
            "q1_value": jnp.mean(q1_value),
            "q2_value": jnp.mean(q2_value),
            "q_value_relabeled_instruction": relabeled_instruction_q_value,
            "q_value_non_relabeled_instruction": non_relabeled_instruction_q_value,
            "target_q_value": jnp.mean(target_q_value),
            "value": jnp.mean(predicted_value),
            "next_value": jnp.mean(next_value),
            "rewards": jnp.mean(rewards),
            "discriminator_predicted_probs": jnp.mean(predicted_probs),
            "discriminator_predicted_probs_action": predicted_probs_action_mean,
            "discriminator_predicted_probs_instruction": predicted_probs_instruction_mean,
            "gt_action": jnp.mean(actions),

            "exp_adv": jnp.nan,
            "loss": jnp.nan,
            "predicted_actions": jnp.nan,
            "discriminator_target": jnp.nan,
            "discriminator_accuracy": jnp.nan,
            "discriminator_accuracy_action": jnp.nan,
            "discriminator_accuracy_instruction": jnp.nan,
            "discriminator_loss_action": jnp.nan,
            "discriminator_loss_instruction": jnp.nan,
            "discriminator_entropy": jnp.nan,
            "discriminator_entropy_bonus": jnp.nan,
            "td3_bc_bc_loss": jnp.nan,
            "td3_bc_q_loss": jnp.nan,
            "td3_bc_lambda": jnp.nan,
            "td3_bc_q_value": jnp.nan,
            "td3_bc_disc_value": jnp.nan,
            **_actor_instruction_filter_stats_nan(),
        }

        return loss_value + loss_q_value, info

    def compute_loss_discriminator(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        preprocess_rng, sample_rng, modality_zero_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        ground_truth_actions = actions
        predicted_actions = self.sample_actions(sample_rng, observation, num_steps=self.diffusion_num_steps)
        predicted_actions = jax.lax.stop_gradient(predicted_actions)

        relabeled_instruction = observation.relabeled_instruction
        if relabeled_instruction is None:
            raise ValueError("relabeled_instruction is required for IQL discriminator.")

        batch_size = ground_truth_actions.shape[0]
        original_batch_size = batch_size
        relabeled_instruction = jnp.asarray(relabeled_instruction).astype(jnp.bool_)
        relabeled_instruction = relabeled_instruction.reshape((batch_size, -1))
        relabeled_instruction = jnp.any(relabeled_instruction, axis=1)
        if observation.relabeled_action is None:
            relabeled_action_from_data = jnp.zeros((batch_size,), dtype=jnp.bool_)
        else:
            relabeled_action_from_data = jnp.asarray(observation.relabeled_action).astype(jnp.bool_)
            relabeled_action_from_data = relabeled_action_from_data.reshape((batch_size, -1))
            relabeled_action_from_data = jnp.any(relabeled_action_from_data, axis=1)
        relabeled_instruction_count = jnp.sum(relabeled_instruction.astype(jnp.int32))
        actor_action_chunk_relabeled_count = self._count_per_sample_true(
            observation.actor_action_chunk_relabeled, batch_size
        )
        discriminator_action_relabel_count = jnp.array(-1, dtype=jnp.int32)

        # Normalize optional scalar-like fields to per-sample vectors.
        def _to_per_sample_vector(value: at.Array | None, *, default: float) -> at.Array:
            if value is None:
                return jnp.full((batch_size,), default, dtype=jnp.float32)
            value = jnp.asarray(value, dtype=jnp.float32)
            if value.ndim == 1 and value.shape == (batch_size,):
                return value
            if value.ndim == 2 and value.shape == (batch_size, 1):
                return value[:, 0]
            if value.ndim == 3 and value.shape == (batch_size, 1, 1):
                return value[:, 0, 0]
            raise ValueError(
                "`relabeled_instruction_similarity` must have shape "
                f"[{batch_size}], [{batch_size}, 1], or [{batch_size}, 1, 1], got {value.shape}."
            )

        def _to_per_sample_weight(value: at.Array | None) -> at.Array:
            if value is None:
                return jnp.ones((batch_size,), dtype=jnp.float32)

            value = jnp.asarray(value, dtype=jnp.float32)
            if value.ndim == 1 and value.shape == (batch_size,):
                return value
            if value.ndim == 2 and value.shape == (batch_size, 1):
                return value[:, 0]
            if value.ndim == 3 and value.shape == (batch_size, 1, 1):
                return value[:, 0, 0]
            raise ValueError(
                "`relabeled_instruction_similarity_weight` must have shape "
                f"[{batch_size}], [{batch_size}, 1], or [{batch_size}, 1, 1], got {value.shape}."
            )

        if self.use_obs_action_similarity_as_weight and observation.relabeled_instruction_similarity_weight is None:
            raise ValueError(
                "`relabeled_instruction_similarity_weight` must be provided when "
                "`use_obs_action_similarity_as_weight=True`."
            )

        relabeled_instruction_similarity = _to_per_sample_vector(
            observation.relabeled_instruction_similarity,
            default=1.0,
        )
        relabeled_instruction_similarity_weight = _to_per_sample_weight(observation.relabeled_instruction_similarity_weight)

        # assert_in_unit_interval("relabeled_instruction_similarity", relabeled_instruction_similarity)
        # assert_in_unit_interval("relabeled_instruction_similarity_weight", relabeled_instruction_similarity_weight)

        predicted_actions = predicted_actions.at[:, :, 7:].set(0.0)
        predicted_actions = jnp.clip(predicted_actions, a_min=-1.0, a_max=1.0)
        predicted_actions = predicted_actions.at[:, :, 6].set(
            jnp.where(
                predicted_actions[:, :, 6] < 0.0,
                -1.0,
                1.0,
            )
        )

        def _duplicate_optional_array(value: at.Array | None) -> at.Array | None:
            if value is None:
                return None
            return jnp.concatenate((value, value), axis=0)

        def _duplicate_optional_dict(value: dict[str, at.Array] | None) -> dict[str, at.Array] | None:
            if value is None:
                return None
            return {key: jnp.concatenate((array, array), axis=0) for key, array in value.items()}

        def _duplicate_observation_batch(value: _model.Observation) -> _model.Observation:
            return _model.Observation(
                images={key: jnp.concatenate((array, array), axis=0) for key, array in value.images.items()},
                image_masks={key: jnp.concatenate((array, array), axis=0) for key, array in value.image_masks.items()},
                state=jnp.concatenate((value.state, value.state), axis=0),
                tokenized_prompt=_duplicate_optional_array(value.tokenized_prompt),
                tokenized_prompt_mask=_duplicate_optional_array(value.tokenized_prompt_mask),
                token_ar_mask=_duplicate_optional_array(value.token_ar_mask),
                token_loss_mask=_duplicate_optional_array(value.token_loss_mask),
                next_images=_duplicate_optional_dict(value.next_images),
                next_image_masks=_duplicate_optional_dict(value.next_image_masks),
                next_state=_duplicate_optional_array(value.next_state),
                done=_duplicate_optional_array(value.done),
                steps_to_episode_end=_duplicate_optional_array(value.steps_to_episode_end),
                relabeled_instruction=_duplicate_optional_array(value.relabeled_instruction),
                relabeled_action=_duplicate_optional_array(value.relabeled_action),
                left_right_flipped_image=_duplicate_optional_array(value.left_right_flipped_image),
                left_right_flipped_action=_duplicate_optional_array(value.left_right_flipped_action),
                relabeled_instruction_similarity=_duplicate_optional_array(value.relabeled_instruction_similarity),
                relabeled_instruction_similarity_weight=_duplicate_optional_array(
                    value.relabeled_instruction_similarity_weight
                ),
                actor_action_chunk_relabeled=_duplicate_optional_array(value.actor_action_chunk_relabeled),
            )

        if self.split_discriminator_head:
            relabeled_instruction_original = relabeled_instruction
            relabeled_action_original = relabeled_action_from_data
            if observation.left_right_flipped_image is None:
                left_right_flipped_image_original = jnp.zeros((batch_size,), dtype=jnp.bool_)
            else:
                left_right_flipped_image_original = jnp.asarray(observation.left_right_flipped_image).astype(jnp.bool_)
                left_right_flipped_image_original = left_right_flipped_image_original.reshape((batch_size, -1))
                left_right_flipped_image_original = jnp.any(left_right_flipped_image_original, axis=1)
            if observation.left_right_flipped_action is None:
                left_right_flipped_action_original = jnp.zeros((batch_size,), dtype=jnp.bool_)
            else:
                left_right_flipped_action_original = jnp.asarray(observation.left_right_flipped_action).astype(jnp.bool_)
                left_right_flipped_action_original = left_right_flipped_action_original.reshape((batch_size, -1))
                left_right_flipped_action_original = jnp.any(left_right_flipped_action_original, axis=1)
            instruction_branch_customized = jnp.logical_or(
                jnp.logical_or(relabeled_instruction_original, relabeled_action_original),
                jnp.logical_or(left_right_flipped_image_original, left_right_flipped_action_original),
            )
            not_relabeled = jnp.logical_not(instruction_branch_customized)
            discriminator_action_relabel_count = jnp.sum(relabeled_action_original.astype(jnp.int32))

            duplicated_actions = jnp.where(
                not_relabeled.reshape((batch_size, 1, 1)),
                predicted_actions,
                ground_truth_actions,
            )
            action_branch_observation = _duplicate_observation_batch(observation)

            action_branch_actions = jnp.concatenate((ground_truth_actions, duplicated_actions), axis=0)
            action_mask = jnp.concatenate((not_relabeled, not_relabeled), axis=0).astype(jnp.float32).reshape((-1, 1, 1))
            target_action = jnp.concatenate(
                (
                    jnp.ones((batch_size,), dtype=jnp.float32),
                    jnp.zeros((batch_size,), dtype=jnp.float32),
                ),
                axis=0,
            ).reshape((-1, 1, 1))

            predicted_logits_action = self.forward_values(
                rng,
                action_branch_observation,
                action_branch_actions,
                num_steps=self.diffusion_num_steps,
                model_type="discriminator",
                preprocessed_observation=True,
            )
            instruction_observation = _instruction_discriminator.maybe_zero_discriminator_modalities(
                modality_zero_rng,
                observation,
                front_camera_zero_probability=self.discriminator_front_camera_zero_probability,
                wrist_camera_zero_probability=self.discriminator_wrist_camera_zero_probability,
                proprio_zero_probability=self.discriminator_proprio_zero_probability,
            )
            predicted_logits_instruction = self.forward_values(
                rng,
                instruction_observation,
                ground_truth_actions,
                num_steps=self.diffusion_num_steps,
                model_type="discriminator_instruction",
                preprocessed_observation=True,
            )

            predicted_probs_action = jax.nn.sigmoid(predicted_logits_action)
            predicted_probs_instruction = jax.nn.sigmoid(predicted_logits_instruction)
            predicted_classes_action = (predicted_probs_action >= 0.5).astype(jnp.int32)
            predicted_classes_instruction = (predicted_probs_instruction >= 0.5).astype(jnp.int32)

            action_mask_total = jnp.sum(action_mask)
            accuracy_action = jnp.where(
                action_mask_total > 0,
                jnp.sum((predicted_classes_action == target_action.astype(jnp.int32)).astype(jnp.float32) * action_mask)
                / action_mask_total,
                jnp.array(jnp.nan, dtype=jnp.float32),
            )

            relabeled_instruction_mask = relabeled_instruction_original.reshape((batch_size, 1, 1))
            relabeled_action_mask = relabeled_action_original.reshape((batch_size, 1, 1))
            if self.use_obs_action_similarity_as_weight:
                instruction_loss_weights = relabeled_instruction_similarity_weight.reshape((batch_size, 1, 1))
            else:
                instruction_loss_weights = None
            instruction_loss_stats = _instruction_discriminator.compute_instruction_pnu_loss(
                predicted_logits_instruction,
                relabeled_instruction_mask,
                relabeled_action_mask,
                left_right_flipped_images=left_right_flipped_image_original.reshape((batch_size, 1, 1)),
                left_right_flipped_actions=left_right_flipped_action_original.reshape((batch_size, 1, 1)),
                treat_instruction_only_as_unlabeled=self.treat_instruction_only_as_unlabeled_for_disc_l,
                treat_action_only_as_unlabeled=self.treat_action_only_as_unlabeled_for_disc_l,
                similarity_weights=instruction_loss_weights,
                use_similarity_weights=self.use_obs_action_similarity_as_weight,
            )
            labeled_mask_instruction = instruction_loss_stats["labeled_mask"].astype(jnp.float32)
            labeled_mask_instruction_bool = instruction_loss_stats["labeled_mask"].astype(jnp.bool_)
            has_labeled_instruction = jnp.sum(labeled_mask_instruction) > 0
            accuracy_instr = jnp.where(
                has_labeled_instruction,
                jnp.sum(
                    (predicted_classes_instruction == instruction_loss_stats["target"].astype(jnp.int32)).astype(jnp.float32)
                    * labeled_mask_instruction
                )
                / jnp.maximum(jnp.sum(labeled_mask_instruction), 1.0),
                jnp.array(jnp.nan, dtype=jnp.float32),
            )
            has_action_branch = action_mask_total > 0
            accuracy = jnp.where(
                has_action_branch,
                jnp.where(has_labeled_instruction, 0.5 * (accuracy_action + accuracy_instr), accuracy_action),
                accuracy_instr,
            )

            loss_bce_action = optax.sigmoid_binary_cross_entropy(predicted_logits_action, target_action)
            loss_bce_action = jnp.where(
                action_mask_total > 0,
                jnp.sum(loss_bce_action * action_mask) / action_mask_total,
                jnp.array(0.0, dtype=jnp.float32),
            )
            loss_bce_instruction = instruction_loss_stats["loss_pnu"]
            loss_bce = loss_bce_action + loss_bce_instruction

            entropy_per_instruction = _instruction_discriminator.binary_entropy_from_logits(predicted_logits_instruction)
            entropy_selected_mask = (
                instruction_loss_stats["labeled_mask"] if self.discriminator_entropy_exclude_unlabeled else None
            )
            if self.use_obs_action_similarity_as_weight:
                entropy_instruction = jnp.where(
                    instruction_loss_stats["unlabeled_count"] > 0,
                    _instruction_discriminator.reduce_binary_entropy_loss(
                        entropy_per_instruction,
                        selected_mask=entropy_selected_mask,
                    ),
                    _instruction_discriminator.reduce_binary_entropy_loss(
                        entropy_per_instruction,
                        similarity_weights=instruction_loss_weights,
                        use_similarity_weights=True,
                        selected_mask=entropy_selected_mask,
                    ),
                )
            else:
                entropy_instruction = _instruction_discriminator.reduce_binary_entropy_loss(
                    entropy_per_instruction,
                    selected_mask=entropy_selected_mask,
                )
            entropy_bonus = self.entropy_regularization_coef * entropy_instruction
            loss = loss_bce - entropy_bonus
            discriminator_entropy = entropy_instruction
            discriminator_entropy_bonus = entropy_bonus

            mean_predicted_probs_action = jnp.where(
                action_mask_total > 0,
                jnp.sum(predicted_probs_action * action_mask) / action_mask_total,
                jnp.array(jnp.nan, dtype=jnp.float32),
            )
            mean_predicted_probs_instruction = _masked_mean_or_nan(
                predicted_probs_instruction.reshape((batch_size,)),
                labeled_mask_instruction_bool.reshape((batch_size,)),
            )
            mean_target_action = jnp.where(
                action_mask_total > 0,
                jnp.sum(target_action * action_mask) / action_mask_total,
                jnp.array(jnp.nan, dtype=jnp.float32),
            )
            mean_target_instruction = _masked_mean_or_nan(
                instruction_loss_stats["target"].reshape((batch_size,)),
                labeled_mask_instruction_bool.reshape((batch_size,)),
            )
            discriminator_predicted_probs = jnp.nan
            discriminator_target = jnp.where(
                has_action_branch,
                jnp.where(has_labeled_instruction, 0.5 * (mean_target_action + mean_target_instruction), mean_target_action),
                mean_target_instruction,
            )
            discriminator_predicted_probs_action = mean_predicted_probs_action
            discriminator_predicted_probs_instruction = mean_predicted_probs_instruction
        else:
            relabeled_action = jnp.logical_not(relabeled_instruction)
            discriminator_action_relabel_count = jnp.sum(relabeled_action.astype(jnp.int32))
            has_relabeled_instruction = jnp.sum(relabeled_instruction) > 0

            # Keep the duplicated branch shape static under jit and mask out policy copies for
            # instruction-relabeled samples, which should only contribute instruction negatives.
            duplicated_observation = _duplicate_observation_batch(observation)
            duplicated_actions = jnp.concatenate((ground_truth_actions, predicted_actions), axis=0)
            target = jnp.concatenate(
                (
                    relabeled_action.astype(jnp.float32),
                    jnp.zeros((batch_size,), dtype=jnp.float32),
                ),
                axis=0,
            ).reshape((-1, 1, 1))
            selected_mask = jnp.concatenate(
                (
                    jnp.ones((batch_size,), dtype=jnp.float32),
                    relabeled_action.astype(jnp.float32),
                ),
                axis=0,
            ).reshape((-1, 1, 1))
            predicted_logits = self.forward_values(
                rng,
                duplicated_observation,
                duplicated_actions,
                num_steps=self.diffusion_num_steps,
                model_type="discriminator",
                preprocessed_observation=True,
            )

            # Calculate accuracy
            predicted_probs = jax.nn.sigmoid(predicted_logits)
            predicted_classes = (predicted_probs >= 0.5).astype(jnp.int32)
            accuracy_values = (predicted_classes == target.astype(jnp.int32)).astype(jnp.float32)
            num_selected = jnp.sum(selected_mask)
            accuracy = jnp.where(
                num_selected > 0,
                jnp.sum(accuracy_values * selected_mask) / num_selected,
                jnp.array(jnp.nan, dtype=jnp.float32),
            )

            instruction_mask = jnp.concatenate(
                (
                    relabeled_instruction.astype(jnp.float32),
                    jnp.zeros((batch_size,), dtype=jnp.float32),
                ),
                axis=0,
            ).reshape((-1, 1, 1))
            num_instruction = jnp.sum(instruction_mask)
            accuracy_instr = jnp.where(
                has_relabeled_instruction,
                jnp.sum(accuracy_values * instruction_mask) / jnp.maximum(num_instruction, 1.0),
                jnp.array(jnp.nan, dtype=jnp.float32),
            )

            action_mask = jnp.concatenate((relabeled_action, relabeled_action), axis=0).astype(jnp.float32).reshape((-1, 1, 1))
            num_action = jnp.sum(action_mask)
            accuracy_action = jnp.where(
                num_action > 0,
                jnp.sum(accuracy_values * action_mask) / num_action,
                jnp.array(jnp.nan, dtype=jnp.float32),
            )

            # Calculate BECWithLogitsLoss
            loss_bce = optax.sigmoid_binary_cross_entropy(predicted_logits, target.astype(jnp.float32))
            loss_bce = jnp.where(
                num_selected > 0,
                jnp.sum(loss_bce * selected_mask) / num_selected,
                jnp.array(0.0, dtype=jnp.float32),
            )
            loss = loss_bce
            discriminator_entropy = jnp.nan
            discriminator_entropy_bonus = jnp.nan
            loss_bce_action = loss_bce
            loss_bce_instruction = jnp.nan
            discriminator_target = jnp.where(
                num_selected > 0,
                jnp.sum(target * selected_mask) / num_selected,
                jnp.array(jnp.nan, dtype=jnp.float32),
            )
            discriminator_predicted_probs = jnp.where(
                num_selected > 0,
                jnp.sum(predicted_probs * selected_mask) / num_selected,
                jnp.array(jnp.nan, dtype=jnp.float32),
            )
            discriminator_predicted_probs_action = jnp.nan
            discriminator_predicted_probs_instruction = jnp.nan

        # if train and self.use_iql:
        #     jax.debug.print(
        #         "[IQL DEBUG][discriminator] split={split} batch(orig/eff)={orig_batch}/{effective_batch} "
        #         "relabeled_instruction={instruction_relabeled}/{orig_batch} "
        #         "actor_action_chunk_relabeled={actor_action_relabeled}/{orig_batch} "
        #         "discriminator_action_relabel={disc_action_relabeled}/{orig_batch}",
        #         split=self.split_discriminator_head,
        #         orig_batch=original_batch_size,
        #         effective_batch=batch_size,
        #         instruction_relabeled=relabeled_instruction_count,
        #         actor_action_relabeled=actor_action_chunk_relabeled_count,
        #         disc_action_relabeled=discriminator_action_relabel_count,
        #         ordered=True,
        #     )

        info = {
            "predicted_actions": jnp.mean(predicted_actions),
            "discriminator_target": discriminator_target,
            "discriminator_predicted_probs": discriminator_predicted_probs,
            "discriminator_predicted_probs_action": discriminator_predicted_probs_action,
            "discriminator_predicted_probs_instruction": discriminator_predicted_probs_instruction,
            "discriminator_accuracy": accuracy,
            "discriminator_accuracy_action": accuracy_action,
            "discriminator_accuracy_instruction": accuracy_instr,
            "discriminator_loss_action": loss_bce_action,
            "discriminator_loss_instruction": loss_bce_instruction,
            "discriminator_entropy": discriminator_entropy,
            "discriminator_entropy_bonus": discriminator_entropy_bonus,
            "gt_action": jnp.mean(ground_truth_actions),

            "exp_adv": jnp.nan,
            "loss": jnp.nan,
            "q1_t_value": jnp.nan,
            "q2_t_value": jnp.nan,
            "q_t_value": jnp.nan,
            "value": jnp.nan,
            "loss_value": jnp.nan,
            "loss_q_value": jnp.nan,
            "q1_value": jnp.nan,
            "q2_value": jnp.nan,
            "q_value_relabeled_instruction": jnp.nan,
            "q_value_non_relabeled_instruction": jnp.nan,
            "target_q_value": jnp.nan,
            "next_value": jnp.nan,
            "rewards": jnp.nan,
            "td3_bc_bc_loss": jnp.nan,
            "td3_bc_q_loss": jnp.nan,
            "td3_bc_lambda": jnp.nan,
            "td3_bc_q_value": jnp.nan,
            "td3_bc_disc_value": jnp.nan,
            **_actor_instruction_filter_stats_nan(),
        }

        return loss, info
    
    def embed_suffix_and_positions_for_values(
        self,
        observation: _model.Observation,
        batch_size: int,
        noise: at.Float[at.Array, "b 1 1"],
        model_type: str,
        next_or_current : str,
        prefix_tokens: at.Float[at.Array, "b p emb"],
        prefix_mask: at.Bool[at.Array, "b p"],
        actions: _model.Actions | None = None,
    ):
        time = 1.0 # dummy value
        suffix_tokens, suffix_mask, suffix_ar_mask, _ = self.embed_suffix(
            observation, noise, jnp.broadcast_to(time, batch_size), model_type=model_type, next_or_current=next_or_current, actions=actions,
        )
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        return suffix_tokens, full_attn_mask, positions

    def forward_values(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions | None = None,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b 1 1"] | None = None,
        model_type: str = None,
        next_or_current : str = "current",
        preprocessed_observation: bool = False,
    ) -> at.Float[at.Array, "b 1 1"]:
        if not preprocessed_observation:
            observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        assert noise is None

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation,
            next_or_current=next_or_current,
            model_type=model_type,
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1    

        if model_type in ["discriminator", "discriminator_instruction"]:
            prefix_llm_inputs = self._make_iql_llm_inputs({self.llm_idx_discriminator_prefix: prefix_tokens})
            _, kv_cache = self.PaliGemma.llm(prefix_llm_inputs, mask=prefix_attn_mask, positions=positions)
            suffix_tokens, full_attn_mask, positions = self.embed_suffix_and_positions_for_values(
                observation, batch_size, noise, model_type=model_type, next_or_current=next_or_current, prefix_tokens=prefix_tokens, prefix_mask=prefix_mask, actions=actions,
            )

            if model_type == "discriminator_instruction":
                if not self.split_discriminator_head or self.llm_idx_discriminator_instruction_expert is None:
                    raise ValueError("`model_type='discriminator_instruction'` requires `split_discriminator_head=True`.")
                discriminator_idx = self.llm_idx_discriminator_instruction_expert
            else:
                discriminator_idx = self.llm_idx_discriminator_expert

            suffix_llm_inputs = self._make_iql_llm_inputs({discriminator_idx: suffix_tokens})
            outputs, _ = self.PaliGemma.llm(
                suffix_llm_inputs,
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._make_iql_adarms_cond(),
            )
            suffix_out = outputs[discriminator_idx]
            assert outputs[self.llm_idx_policy_prefix] is None
            assert outputs[self.llm_idx_critic_prefix] is None
            assert outputs[self.llm_idx_target_critic_prefix] is None
            assert outputs[self.llm_idx_discriminator_prefix] is None
            if model_type == "discriminator_instruction":
                value = self.discriminator_out_proj_instruction(suffix_out[:, -1 :])
            else:
                value = self.discriminator_out_proj(suffix_out[:, -1 :])
            return value

        elif model_type == "q1q2":
            prefix_llm_inputs = self._make_iql_llm_inputs({self.llm_idx_critic_prefix: prefix_tokens})
            _, kv_cache = self.PaliGemma.llm(prefix_llm_inputs, mask=prefix_attn_mask, positions=positions)
            suffix_tokens, full_attn_mask, positions = self.embed_suffix_and_positions_for_values(
                observation, batch_size, noise, model_type="q1", next_or_current=next_or_current, prefix_tokens=prefix_tokens, prefix_mask=prefix_mask, actions=actions,
            )
            q1_inputs = self._make_iql_llm_inputs({self.llm_idx_q1_expert: suffix_tokens})
            outputs, _ = self.PaliGemma.llm(
                q1_inputs,
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._make_iql_adarms_cond(),
            )
            suffix_out = outputs[self.llm_idx_q1_expert]
            assert outputs[self.llm_idx_policy_prefix] is None
            assert outputs[self.llm_idx_critic_prefix] is None
            assert outputs[self.llm_idx_target_critic_prefix] is None
            assert outputs[self.llm_idx_discriminator_prefix] is None
            value_q1 = self.q1_out_proj(suffix_out[:, -1 :])

            suffix_tokens, full_attn_mask, positions = self.embed_suffix_and_positions_for_values(
                observation, batch_size, noise, model_type="q2", next_or_current=next_or_current, prefix_tokens=prefix_tokens, prefix_mask=prefix_mask, actions=actions,
            )
            q2_inputs = self._make_iql_llm_inputs({self.llm_idx_q2_expert: suffix_tokens})
            outputs, _ = self.PaliGemma.llm(
                q2_inputs,
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._make_iql_adarms_cond(),
            )
            suffix_out = outputs[self.llm_idx_q2_expert]
            assert outputs[self.llm_idx_policy_prefix] is None
            assert outputs[self.llm_idx_critic_prefix] is None
            assert outputs[self.llm_idx_target_critic_prefix] is None
            assert outputs[self.llm_idx_discriminator_prefix] is None
            value_q2 = self.q2_out_proj(suffix_out[:, -1 :])

            return value_q1, value_q2

        elif model_type == "q1tq2t":
            prefix_llm_inputs = self._make_iql_llm_inputs({self.llm_idx_target_critic_prefix: prefix_tokens})
            _, kv_cache = self.PaliGemma.llm(prefix_llm_inputs, mask=prefix_attn_mask, positions=positions)
            suffix_tokens, full_attn_mask, positions = self.embed_suffix_and_positions_for_values(
                observation, batch_size, noise, model_type="q1t", next_or_current=next_or_current, prefix_tokens=prefix_tokens, prefix_mask=prefix_mask, actions=actions,
            )
            q1t_inputs = self._make_iql_llm_inputs({self.llm_idx_q1_target_expert: suffix_tokens})
            outputs, _ = self.PaliGemma.llm(
                q1t_inputs,
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._make_iql_adarms_cond(),
            )
            suffix_out = outputs[self.llm_idx_q1_target_expert]
            assert outputs[self.llm_idx_policy_prefix] is None
            assert outputs[self.llm_idx_critic_prefix] is None
            assert outputs[self.llm_idx_target_critic_prefix] is None
            assert outputs[self.llm_idx_discriminator_prefix] is None
            value_q1t = self.q1t_out_proj(suffix_out[:, -1 :])

            suffix_tokens, full_attn_mask, positions = self.embed_suffix_and_positions_for_values(
                observation, batch_size, noise, model_type="q2t", next_or_current=next_or_current, prefix_tokens=prefix_tokens, prefix_mask=prefix_mask, actions=actions,
            )
            q2t_inputs = self._make_iql_llm_inputs({self.llm_idx_q2_target_expert: suffix_tokens})
            outputs, _ = self.PaliGemma.llm(
                q2t_inputs,
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._make_iql_adarms_cond(),
            )
            suffix_out = outputs[self.llm_idx_q2_target_expert]
            assert outputs[self.llm_idx_policy_prefix] is None
            assert outputs[self.llm_idx_critic_prefix] is None
            assert outputs[self.llm_idx_target_critic_prefix] is None
            assert outputs[self.llm_idx_discriminator_prefix] is None
            value_q2t = self.q2t_out_proj(suffix_out[:, -1 :])

            return value_q1t, value_q2t
        elif model_type == "v":
            prefix_llm_inputs = self._make_iql_llm_inputs({self.llm_idx_critic_prefix: prefix_tokens})
            _, kv_cache = self.PaliGemma.llm(prefix_llm_inputs, mask=prefix_attn_mask, positions=positions)

            suffix_tokens, full_attn_mask, positions = self.embed_suffix_and_positions_for_values(
                observation, batch_size, noise, model_type="v", next_or_current=next_or_current, prefix_tokens=prefix_tokens, prefix_mask=prefix_mask
            )
            value_inputs = self._make_iql_llm_inputs({self.llm_idx_value_expert: suffix_tokens})
            outputs, _ = self.PaliGemma.llm(
                value_inputs,
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._make_iql_adarms_cond(),
            )

            suffix_out = outputs[self.llm_idx_value_expert]
            assert outputs[self.llm_idx_policy_prefix] is None
            assert outputs[self.llm_idx_critic_prefix] is None
            assert outputs[self.llm_idx_target_critic_prefix] is None
            assert outputs[self.llm_idx_discriminator_prefix] is None
            value = self.v_out_proj(suffix_out[:, -1 :])
            return value
        else:
            raise Exception(f"model_type: {model_type} is not supported")

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, model_type="actor")
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        if self.use_iql:
            prefix_llm_inputs = self._make_iql_llm_inputs({self.llm_idx_policy_prefix: prefix_tokens})
            _, kv_cache = self.PaliGemma.llm(prefix_llm_inputs, mask=prefix_attn_mask, positions=positions)
        else:
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            if self.use_iql:
                step_inputs = self._make_iql_llm_inputs(
                    {self.llm_idx_actor_expert: suffix_tokens}
                )
                step_adarms_cond = self._make_iql_adarms_cond({self.llm_idx_actor_expert: adarms_cond})
                outputs, _ = self.PaliGemma.llm(
                    step_inputs,
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=step_adarms_cond,
                )
                prefix_out = outputs[self.llm_idx_policy_prefix]
                suffix_out = outputs[self.llm_idx_actor_expert]
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
