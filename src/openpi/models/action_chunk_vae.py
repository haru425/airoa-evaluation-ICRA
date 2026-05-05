import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
import openpi.shared.array_typing as at


def _as_jnp_dtype(dtype: str | jnp.dtype) -> jnp.dtype:
    if isinstance(dtype, str):
        return jnp.dtype(dtype)
    return jnp.dtype(dtype)


def _default_rng(rng: at.KeyArrayLike | None) -> at.KeyArrayLike:
    return jax.random.key(0) if rng is None else rng


def _init_param(
    rngs: nnx.Rngs,
    shape: tuple[int, ...],
    *,
    dtype: jnp.dtype,
    scale: float = 0.02,
    name: str | None = None,
) -> nnx.Param:
    value = scale * jax.random.normal(rngs(), shape, dtype=dtype)
    if name is None:
        return nnx.Param(value)
    return nnx.Param(value, name=name)


def compute_action_chunk_vae_losses(
    reconstruction: at.Array,
    target_actions: at.Array,
    mu: at.Array,
    logvar: at.Array,
    *,
    beta: float,
    valid_action_dim: int,
) -> dict[str, at.Array]:
    reconstruction = jnp.asarray(reconstruction, dtype=jnp.float32)
    target_actions = jnp.asarray(target_actions, dtype=jnp.float32)
    mu = jnp.asarray(mu, dtype=jnp.float32)
    logvar = jnp.asarray(logvar, dtype=jnp.float32)

    action_dim = int(target_actions.shape[-1])
    if not 0 < int(valid_action_dim) <= action_dim:
        raise ValueError(
            f"`valid_action_dim` must be in [1, {action_dim}], got {valid_action_dim}."
        )

    squared_error = jnp.square(reconstruction - target_actions)
    valid_mask = (jnp.arange(action_dim) < int(valid_action_dim)).astype(jnp.float32).reshape((1, 1, action_dim))
    padded_mask = 1.0 - valid_mask

    valid_denominator = jnp.maximum(jnp.sum(valid_mask) * target_actions.shape[0] * target_actions.shape[1], 1.0)
    valid_scalars_per_sample = jnp.maximum(jnp.sum(valid_mask) * target_actions.shape[1], 1.0)
    padded_denominator = jnp.sum(padded_mask) * target_actions.shape[0] * target_actions.shape[1]

    reconstruction_mse_valid_dims = jnp.sum(squared_error * valid_mask) / valid_denominator
    reconstruction_mse_padded_dims = jnp.where(
        padded_denominator > 0,
        jnp.sum(squared_error * padded_mask) / padded_denominator,
        jnp.array(0.0, dtype=jnp.float32),
    )

    kl_per_sample = -0.5 * jnp.sum(1.0 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)
    kl_loss = jnp.mean(kl_per_sample)
    # Keep the reconstruction and KL terms on comparable scales so `beta`
    # does not implicitly grow with action horizon or valid action dim.
    kl_loss_normalized = kl_loss / valid_scalars_per_sample

    beta_value = jnp.asarray(beta, dtype=jnp.float32)
    loss = reconstruction_mse_valid_dims + beta_value * kl_loss_normalized

    return {
        "loss": loss,
        "reconstruction_loss": reconstruction_mse_valid_dims,
        "kl_loss": kl_loss,
        "kl_loss_normalized": kl_loss_normalized,
        "beta": beta_value,
        "reconstruction_mse_valid_dims": reconstruction_mse_valid_dims,
        "reconstruction_mse_padded_dims": reconstruction_mse_padded_dims,
    }


class TransformerEncoderLayer(nnx.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            dropout_rate=dropout,
            dtype=dtype,
            decode=False,
            rngs=rngs,
        )
        self.norm_attn = nnx.LayerNorm(hidden_dim, dtype=dtype, rngs=rngs)
        self.norm_ffn = nnx.LayerNorm(hidden_dim, dtype=dtype, rngs=rngs)
        self.ffn_in = nnx.Linear(hidden_dim, mlp_dim, dtype=dtype, rngs=rngs)
        self.ffn_out = nnx.Linear(mlp_dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: at.Array, *, deterministic: bool, rngs: nnx.Rngs) -> at.Array:
        attn_input = self.norm_attn(x)
        attn_output = self.self_attn(attn_input, deterministic=deterministic, rngs=rngs)
        attn_output = self.dropout(attn_output, deterministic=deterministic, rngs=rngs)
        x = x + attn_output

        ffn_input = self.norm_ffn(x)
        ffn_output = self.ffn_out(jax.nn.gelu(self.ffn_in(ffn_input)))
        ffn_output = self.dropout(ffn_output, deterministic=deterministic, rngs=rngs)
        return x + ffn_output


class TransformerDecoderLayer(nnx.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            dropout_rate=dropout,
            dtype=dtype,
            decode=False,
            rngs=rngs,
        )
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            dropout_rate=dropout,
            dtype=dtype,
            decode=False,
            rngs=rngs,
        )
        self.norm_self = nnx.LayerNorm(hidden_dim, dtype=dtype, rngs=rngs)
        self.norm_cross = nnx.LayerNorm(hidden_dim, dtype=dtype, rngs=rngs)
        self.norm_ffn = nnx.LayerNorm(hidden_dim, dtype=dtype, rngs=rngs)
        self.ffn_in = nnx.Linear(hidden_dim, mlp_dim, dtype=dtype, rngs=rngs)
        self.ffn_out = nnx.Linear(mlp_dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        x: at.Array,
        memory: at.Array,
        *,
        deterministic: bool,
        rngs: nnx.Rngs,
    ) -> at.Array:
        self_attn_input = self.norm_self(x)
        self_attn_output = self.self_attn(self_attn_input, deterministic=deterministic, rngs=rngs)
        self_attn_output = self.dropout(self_attn_output, deterministic=deterministic, rngs=rngs)
        x = x + self_attn_output

        cross_attn_input = self.norm_cross(x)
        cross_attn_output = self.cross_attn(
            cross_attn_input,
            memory,
            memory,
            deterministic=deterministic,
            rngs=rngs,
        )
        cross_attn_output = self.dropout(cross_attn_output, deterministic=deterministic, rngs=rngs)
        x = x + cross_attn_output

        ffn_input = self.norm_ffn(x)
        ffn_output = self.ffn_out(jax.nn.gelu(self.ffn_in(ffn_input)))
        ffn_output = self.dropout(ffn_output, deterministic=deterministic, rngs=rngs)
        return x + ffn_output


@dataclasses.dataclass(frozen=True)
class ActionChunkVAEConfig(_model.BaseModelConfig):
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 1

    latent_dim: int = 256
    hidden_dim: int = 256
    num_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    mlp_dim: int = 1024
    dropout: float = 0.1
    dtype: str = "float32"
    valid_action_dim: int | None = None

    # Keep these attributes to remain compatible with the common train-state initializer.
    use_iql: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"`hidden_dim` must be divisible by `num_heads`, got {self.hidden_dim=} and {self.num_heads=}."
            )
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be > 0, got {self.action_horizon}.")
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be > 0, got {self.action_dim}.")
        if self.latent_dim <= 0:
            raise ValueError(f"`latent_dim` must be > 0, got {self.latent_dim}.")
        if self.encoder_layers <= 0 or self.decoder_layers <= 0:
            raise ValueError(
                f"`encoder_layers` and `decoder_layers` must be > 0, got {self.encoder_layers=} and {self.decoder_layers=}."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {self.dropout}.")
        if self.valid_action_dim is not None and not 0 < self.valid_action_dim <= self.action_dim:
            raise ValueError(
                f"`valid_action_dim` must be within [1, {self.action_dim}], got {self.valid_action_dim}."
            )

    @property
    def model_type(self) -> _model.ModelType:
        # Reuse PI05 normalization defaults so the VAE sees the same normalized action space.
        return _model.ModelType.PI05

    def create(self, rng: at.KeyArrayLike) -> "ActionChunkVAEModel":
        return ActionChunkVAEModel(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, 1, 1, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        token_spec = jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
        token_mask_spec = jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_)
        state_spec = jax.ShapeDtypeStruct([batch_size, 1], jnp.float32)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={key: image_spec for key in _model.IMAGE_KEYS},
                image_masks={key: image_mask_spec for key in _model.IMAGE_KEYS},
                state=state_spec,
                tokenized_prompt=token_spec,
                tokenized_prompt_mask=token_mask_spec,
                next_images={key: image_spec for key in _model.IMAGE_KEYS},
                next_image_masks={key: image_mask_spec for key in _model.IMAGE_KEYS},
                next_state=state_spec,
                done=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
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


class ActionChunkVAEModel(_model.BaseModel):
    def __init__(self, config: ActionChunkVAEConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.valid_action_dim = config.valid_action_dim or config.action_dim
        self.dropout_rate = config.dropout
        self.compute_dtype = _as_jnp_dtype(config.dtype)

        self.action_in_proj = nnx.Linear(config.action_dim, config.hidden_dim, dtype=self.compute_dtype, rngs=rngs)
        self.decoder_latent_proj = nnx.Linear(
            config.latent_dim,
            config.hidden_dim,
            dtype=self.compute_dtype,
            rngs=rngs,
        )
        self.mu_proj = nnx.Linear(config.hidden_dim, config.latent_dim, dtype=self.compute_dtype, rngs=rngs)
        self.logvar_proj = nnx.Linear(config.hidden_dim, config.latent_dim, dtype=self.compute_dtype, rngs=rngs)
        self.action_out_proj = nnx.Linear(config.hidden_dim, config.action_dim, dtype=self.compute_dtype, rngs=rngs)

        self.encoder_latent_token = _init_param(
            rngs,
            (config.hidden_dim,),
            dtype=self.compute_dtype,
            name="encoder_latent_token",
        )
        self.encoder_pos_embedding = _init_param(
            rngs,
            (config.action_horizon + 1, config.hidden_dim),
            dtype=self.compute_dtype,
            name="encoder_pos_embedding",
        )
        self.decoder_query_tokens = _init_param(
            rngs,
            (config.action_horizon, config.hidden_dim),
            dtype=self.compute_dtype,
            name="decoder_query_tokens",
        )
        self.decoder_pos_embedding = _init_param(
            rngs,
            (config.action_horizon, config.hidden_dim),
            dtype=self.compute_dtype,
            name="decoder_pos_embedding",
        )

        self.encoder_norm = nnx.LayerNorm(config.hidden_dim, dtype=self.compute_dtype, rngs=rngs)
        self.decoder_norm = nnx.LayerNorm(config.hidden_dim, dtype=self.compute_dtype, rngs=rngs)

        self.encoder_layer_names = tuple(f"encoder_layer_{idx}" for idx in range(config.encoder_layers))
        for layer_name in self.encoder_layer_names:
            setattr(
                self,
                layer_name,
                TransformerEncoderLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    mlp_dim=config.mlp_dim,
                    dropout=config.dropout,
                    dtype=self.compute_dtype,
                    rngs=rngs,
                ),
            )

        self.decoder_layer_names = tuple(f"decoder_layer_{idx}" for idx in range(config.decoder_layers))
        for layer_name in self.decoder_layer_names:
            setattr(
                self,
                layer_name,
                TransformerDecoderLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    mlp_dim=config.mlp_dim,
                    dropout=config.dropout,
                    dtype=self.compute_dtype,
                    rngs=rngs,
                ),
            )

        # This attribute is toggled by model.train() / model.eval().
        self.deterministic = True

    def _encoder_forward(self, actions: at.Array, *, deterministic: bool, rng: at.KeyArrayLike | None) -> at.Array:
        actions = jnp.asarray(actions, dtype=self.compute_dtype)
        rngs = nnx.Rngs(_default_rng(rng))

        action_tokens = self.action_in_proj(actions)
        batch_size = action_tokens.shape[0]
        latent_token = jnp.broadcast_to(
            self.encoder_latent_token.value.reshape((1, 1, self.hidden_dim)),
            (batch_size, 1, self.hidden_dim),
        )
        hidden = jnp.concatenate((latent_token, action_tokens), axis=1)
        hidden = hidden + self.encoder_pos_embedding.value.reshape((1, self.action_horizon + 1, self.hidden_dim))

        for layer_name in self.encoder_layer_names:
            hidden = getattr(self, layer_name)(hidden, deterministic=deterministic, rngs=rngs)

        hidden = self.encoder_norm(hidden)
        return hidden[:, 0, :]

    def _sample_latent(
        self,
        mu: at.Array,
        logvar: at.Array,
        *,
        deterministic: bool,
        rng: at.KeyArrayLike | None,
    ) -> at.Array:
        if deterministic:
            return mu
        sample_rng = _default_rng(rng)
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(sample_rng, mu.shape, dtype=mu.dtype)
        return mu + std * eps

    def encode_actions(
        self,
        actions: at.Array,
        *,
        deterministic: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Array, at.Array, at.Array]:
        encoder_rng, sample_rng = jax.random.split(_default_rng(rng))
        latent_hidden = self._encoder_forward(actions, deterministic=deterministic, rng=encoder_rng)
        mu = self.mu_proj(latent_hidden).astype(jnp.float32)
        logvar = self.logvar_proj(latent_hidden).astype(jnp.float32)
        z = self._sample_latent(mu, logvar, deterministic=deterministic, rng=sample_rng)
        return z, mu, logvar

    def decode_latent(
        self,
        z: at.Array,
        *,
        deterministic: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Array:
        z = jnp.asarray(z, dtype=jnp.float32)
        rngs = nnx.Rngs(_default_rng(rng))

        memory = self.decoder_latent_proj(z.astype(self.compute_dtype))[:, None, :]
        batch_size = z.shape[0]
        query_tokens = jnp.broadcast_to(
            self.decoder_query_tokens.value.reshape((1, self.action_horizon, self.hidden_dim)),
            (batch_size, self.action_horizon, self.hidden_dim),
        )
        hidden = query_tokens + self.decoder_pos_embedding.value.reshape((1, self.action_horizon, self.hidden_dim))

        for layer_name in self.decoder_layer_names:
            hidden = getattr(self, layer_name)(hidden, memory, deterministic=deterministic, rngs=rngs)

        hidden = self.decoder_norm(hidden)
        return self.action_out_proj(hidden).astype(jnp.float32)

    def reconstruct(
        self,
        actions: at.Array,
        *,
        deterministic: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Array, at.Array, at.Array, at.Array]:
        encode_rng, decode_rng = jax.random.split(_default_rng(rng))
        z, mu, logvar = self.encode_actions(actions, deterministic=deterministic, rng=encode_rng)
        reconstruction = self.decode_latent(z, deterministic=deterministic, rng=decode_rng)
        return reconstruction, z, mu, logvar

    def compute_vae_loss(
        self,
        rng: at.KeyArrayLike,
        actions: at.Array,
        *,
        beta: float = 1.0,
        train: bool = False,
    ) -> tuple[at.Array, dict[str, at.Array]]:
        reconstruction, z, mu, logvar = self.reconstruct(actions, deterministic=not train, rng=rng)
        info = compute_action_chunk_vae_losses(
            reconstruction,
            actions,
            mu,
            logvar,
            beta=beta,
            valid_action_dim=self.valid_action_dim,
        )
        info.update(
            {
                "latent_mu_mean": jnp.mean(mu),
                "latent_mu_std": jnp.std(mu),
                "latent_logvar_mean": jnp.mean(logvar),
                "latent_std_mean": jnp.mean(jnp.exp(0.5 * logvar)),
                "latent_z_mean": jnp.mean(z),
            }
        )
        return info["loss"], info

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Array, dict[str, at.Array]]:
        del observation
        return self.compute_vae_loss(rng, actions, train=train)

    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, **kwargs) -> _model.Actions:
        batch_size = observation.state.shape[0]
        z = jax.random.normal(rng, (batch_size, self.latent_dim), dtype=jnp.float32)
        deterministic = bool(kwargs.pop("deterministic", True))
        return self.decode_latent(z, deterministic=deterministic, rng=rng)
