import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # IQL+GAIL
    use_iql: bool = False
    iql_critic_tau: float = 0.7
    iql_actor_temperature: float = 0.5
    iql_exp_adv_max: float = 2.0
    iql_actor_loss_type: Literal["awr", "rwr", "td3_bc", "td3_bc_disc"] = "awr"
    iql_td3_bc_alpha: float = 2.5
    iql_tau: float = 0.05
    critic_gamma: float = 0.7
    use_binary_reward: bool = False
    positive_reward_last_step_num: int = 1

    gail_discriminator_update_steps: int = 1
    gail_critic_update_steps: int = 2
    gail_actor_update_steps: int = 2
    action_relabeling_ratio_for_disc_l: float = 0.0
    joint_relabeling_ratio_for_disc_l: float = 0.0
    flipped_image_action_ratio_for_disc_l: float = 0.0
    flipped_image_only_ratio_for_disc_l: float = 0.0
    flipped_action_only_ratio_for_disc_l: float = 0.0
    split_discriminator_head: bool = False
    instruction_discriminator_only_pretrain: bool = False
    use_obs_action_similarity_as_weight: bool = False
    entropy_regularization_coef: float = 0.0
    actor_instruction_discriminator_topk_ratio: float = 1.0
    treat_instruction_only_as_unlabeled_for_disc_l: bool = False
    treat_action_only_as_unlabeled_for_disc_l: bool = False
    discriminator_entropy_exclude_unlabeled: bool = False
    discriminator_front_camera_zero_probability: float = 0.0
    discriminator_wrist_camera_zero_probability: float = 0.0
    discriminator_proprio_zero_probability: float = 0.0
    rel_disc_reward_weight: float = 0.5

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        paligemma_uses_lora = "lora" in self.paligemma_variant
        action_expert_uses_lora = "lora" in self.action_expert_variant
        for ratio_name, ratio_value in (
            ("action_relabeling_ratio_for_disc_l", self.action_relabeling_ratio_for_disc_l),
            ("joint_relabeling_ratio_for_disc_l", self.joint_relabeling_ratio_for_disc_l),
            ("flipped_image_action_ratio_for_disc_l", self.flipped_image_action_ratio_for_disc_l),
            ("flipped_image_only_ratio_for_disc_l", self.flipped_image_only_ratio_for_disc_l),
            ("flipped_action_only_ratio_for_disc_l", self.flipped_action_only_ratio_for_disc_l),
        ):
            if ratio_value < 0.0 or ratio_value > 1.0:
                raise ValueError(f"`{ratio_name}` must be in [0, 1], got {ratio_value}.")
        total_discriminator_instruction_ratio = (
            self.action_relabeling_ratio_for_disc_l
            + self.joint_relabeling_ratio_for_disc_l
            + self.flipped_image_action_ratio_for_disc_l
            + self.flipped_image_only_ratio_for_disc_l
            + self.flipped_action_only_ratio_for_disc_l
        )
        if total_discriminator_instruction_ratio > 1.0:
            raise ValueError(
                "Instruction discriminator ratios must sum to <= 1.0 across "
                "`action_relabeling_ratio_for_disc_l`, `joint_relabeling_ratio_for_disc_l`, "
                "`flipped_image_action_ratio_for_disc_l`, `flipped_image_only_ratio_for_disc_l`, and "
                f"`flipped_action_only_ratio_for_disc_l`, got {total_discriminator_instruction_ratio}."
            )
        for ratio_name, ratio_value in (
            ("discriminator_front_camera_zero_probability", self.discriminator_front_camera_zero_probability),
            ("discriminator_wrist_camera_zero_probability", self.discriminator_wrist_camera_zero_probability),
            ("discriminator_proprio_zero_probability", self.discriminator_proprio_zero_probability),
            ("rel_disc_reward_weight", self.rel_disc_reward_weight),
        ):
            if ratio_value < 0.0 or ratio_value > 1.0:
                raise ValueError(f"`{ratio_name}` must be in [0, 1], got {ratio_value}.")
        if self.use_iql and paligemma_uses_lora != action_expert_uses_lora:
            raise ValueError(
                "Mixed LoRA/non-LoRA IQL variants are not supported. "
                f"Got paligemma_variant={self.paligemma_variant!r}, "
                f"action_expert_variant={self.action_expert_variant!r}."
            )
        if self.split_discriminator_head and not self.use_iql:
            raise ValueError("`split_discriminator_head=True` requires `use_iql=True`.")
        if self.instruction_discriminator_only_pretrain and not self.split_discriminator_head:
            raise ValueError(
                "`instruction_discriminator_only_pretrain=True` requires `split_discriminator_head=True`."
            )
        has_instruction_branch_customization = any(
            (
                self.action_relabeling_ratio_for_disc_l > 0.0,
                self.joint_relabeling_ratio_for_disc_l > 0.0,
                self.flipped_image_action_ratio_for_disc_l > 0.0,
                self.flipped_image_only_ratio_for_disc_l > 0.0,
                self.flipped_action_only_ratio_for_disc_l > 0.0,
                self.treat_instruction_only_as_unlabeled_for_disc_l,
                self.treat_action_only_as_unlabeled_for_disc_l,
                self.discriminator_entropy_exclude_unlabeled,
                self.discriminator_front_camera_zero_probability > 0.0,
                self.discriminator_wrist_camera_zero_probability > 0.0,
                self.discriminator_proprio_zero_probability > 0.0,
            )
        )
        if has_instruction_branch_customization and not self.use_iql:
            raise ValueError("Instruction discriminator settings require `use_iql=True`.")
        if has_instruction_branch_customization and not self.split_discriminator_head:
            raise ValueError("Instruction discriminator settings require `split_discriminator_head=True`.")
        flip_ratio_enabled = any(
            (
                self.flipped_image_action_ratio_for_disc_l > 0.0,
                self.flipped_image_only_ratio_for_disc_l > 0.0,
                self.flipped_action_only_ratio_for_disc_l > 0.0,
            )
        )
        continuous_proprio_enabled = not self.pi05 and self.discriminator_proprio_zero_probability < 1.0
        discrete_proprio_enabled = self.pi05 and self.discrete_state_input
        if flip_ratio_enabled and (continuous_proprio_enabled or discrete_proprio_enabled):
            raise ValueError(
                "Left-right flip instruction discriminator ratios require proprio to be disabled. "
                "Set `discriminator_proprio_zero_probability=1.0` for continuous-state models or "
                "`discrete_state_input=False` for pi05, or keep all flip ratios at 0."
            )
        if self.use_obs_action_similarity_as_weight and not self.use_iql:
            raise ValueError("`use_obs_action_similarity_as_weight=True` requires `use_iql=True`.")
        if self.use_obs_action_similarity_as_weight and not self.split_discriminator_head:
            raise ValueError(
                "`use_obs_action_similarity_as_weight=True` requires `split_discriminator_head=True`."
            )
        if self.iql_actor_loss_type not in ("awr", "rwr", "td3_bc", "td3_bc_disc"):
            raise ValueError(
                "`iql_actor_loss_type` must be one of ('awr', 'rwr', 'td3_bc', 'td3_bc_disc'), "
                f"got {self.iql_actor_loss_type!r}."
            )
        if self.iql_actor_loss_type in ("rwr", "td3_bc", "td3_bc_disc") and not self.use_iql:
            raise ValueError(f"`iql_actor_loss_type='{self.iql_actor_loss_type}'` requires `use_iql=True`.")
        if self.iql_actor_loss_type in ("td3_bc", "td3_bc_disc") and self.iql_td3_bc_alpha <= 0.0:
            raise ValueError(f"`iql_td3_bc_alpha` must be > 0, got {self.iql_td3_bc_alpha}.")
        if self.use_binary_reward and not self.use_iql:
            raise ValueError("`use_binary_reward=True` requires `use_iql=True`.")
        if self.positive_reward_last_step_num < 1:
            raise ValueError(
                "`positive_reward_last_step_num` must be >= 1, "
                f"got {self.positive_reward_last_step_num}."
            )
        if self.use_binary_reward and self.iql_actor_loss_type in ("rwr", "td3_bc_disc"):
            raise ValueError(
                "`use_binary_reward=True` supports only `iql_actor_loss_type` values "
                "`awr` and `td3_bc`."
            )
        if self.entropy_regularization_coef < 0.0:
            raise ValueError(
                f"`entropy_regularization_coef` must be >= 0, got {self.entropy_regularization_coef}."
            )
        if self.entropy_regularization_coef > 0.0 and not self.split_discriminator_head:
            raise ValueError(
                "`entropy_regularization_coef>0` requires `split_discriminator_head=True`."
            )
        if not 0.0 < self.actor_instruction_discriminator_topk_ratio <= 1.0:
            raise ValueError(
                "`actor_instruction_discriminator_topk_ratio` must be within (0.0, 1.0], "
                f"got {self.actor_instruction_discriminator_topk_ratio}."
            )
        if self.actor_instruction_discriminator_topk_ratio < 1.0 and not self.use_iql:
            raise ValueError(
                "`actor_instruction_discriminator_topk_ratio < 1.0` requires `use_iql=True`."
            )
        if self.actor_instruction_discriminator_topk_ratio < 1.0 and not self.split_discriminator_head:
            raise ValueError(
                "`actor_instruction_discriminator_topk_ratio < 1.0` requires `split_discriminator_head=True`."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    def get_iql_expert_weight_sharing(self) -> tuple[bool, bool]:
        """Returns whether the 2B and 300M IQL expert groups share base LLM weights."""
        return "lora" in self.paligemma_variant, "lora" in self.action_expert_variant

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "Pi0":
        if self.use_iql:
            from openpi.models import pi0 as _pi0

            params = _pi0.maybe_replicate_shared_iql_image_encoder_params(params)
        return super().load(params, remove_extra_params=remove_extra_params)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                next_images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                next_image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                next_state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                done=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                steps_to_episode_end=jax.ShapeDtypeStruct([batch_size], jnp.int32),
                relabeled_instruction=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                relabeled_action=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                left_right_flipped_image=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                left_right_flipped_action=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                relabeled_instruction_similarity=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                relabeled_instruction_similarity_weight=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                actor_action_chunk_relabeled=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        if self.use_iql:
            action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_4.*")
        else:
            action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
            if self.use_iql:
                max_norm_idx = 12 if self.split_discriminator_head else 11
                for i in range(1, max_norm_idx):
                    if i != 4:
                        filters.append(
                            nnx.Not(nnx_utils.PathRegex(f".*_norm_{i}.*")),
                        )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
