import abc
from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
from openpi.shared import path_utils
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    next_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    next_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    next_state: at.Float[ArrayT, "*b s"] | None = None
    done: at.Bool[ArrayT, "*b"] | None = None
    steps_to_episode_end: at.Int[ArrayT, "*b"] | None = None
    relabeled_instruction: at.Bool[ArrayT, "*b"] | None = None
    relabeled_action: at.Bool[ArrayT, "*b"] | None = None
    left_right_flipped_image: at.Bool[ArrayT, "*b"] | None = None
    left_right_flipped_action: at.Bool[ArrayT, "*b"] | None = None
    relabeled_instruction_similarity: at.Float[ArrayT, "*b"] | None = None
    relabeled_instruction_similarity_weight: at.Float[ArrayT, "*b"] | None = None
    actor_action_chunk_relabeled: at.Bool[ArrayT, "*b"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        
        if "next_image" in data:
            for key in data["next_image"]:
                if data["next_image"][key].dtype == np.uint8:
                    data["next_image"][key] = data["next_image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                elif hasattr(data["next_image"][key], "dtype") and data["next_image"][key].dtype == torch.uint8:
                    data["next_image"][key] = data["next_image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            next_images=data["next_image"] if "next_image" in data else None,
            next_image_masks=data["next_image_mask"] if "next_image_mask" in data else None,
            next_state=data["next_state"] if "next_state" in data else None,
            done=data["done"] if "done" in data else None,
            steps_to_episode_end=data["steps_to_episode_end"] if "steps_to_episode_end" in data else None,
            relabeled_instruction=data["relabeled_instruction"] if "relabeled_instruction" in data else None,
            relabeled_action=data["relabeled_action"] if "relabeled_action" in data else None,
            left_right_flipped_image=(
                data["left_right_flipped_image"] if "left_right_flipped_image" in data else None
            ),
            left_right_flipped_action=(
                data["left_right_flipped_action"] if "left_right_flipped_action" in data else None
            ),
            relabeled_instruction_similarity=(
                data["relabeled_instruction_similarity"] if "relabeled_instruction_similarity" in data else None
            ),
            relabeled_instruction_similarity_weight=(
                data["relabeled_instruction_similarity_weight"]
                if "relabeled_instruction_similarity_weight" in data
                else None
            ),
            actor_action_chunk_relabeled=(
                data["actor_action_chunk_relabeled"] if "actor_action_chunk_relabeled" in data else None
            ),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        next_images = result.pop("next_images")
        next_image_masks = result.pop("next_image_masks")
        if next_images is not None or next_image_masks is not None:
            if next_images is None or next_image_masks is None:
                raise ValueError("next_images and next_image_masks must be provided together.")
            result["next_image"] = next_images
            result["next_image_mask"] = next_image_masks
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])
    
    out_next_images = {}
    out_next_masks = {}
    if observation.next_images is not None:
        for key in image_keys:
            image = observation.next_images[key]
            if image.shape[1:3] != image_resolution:
                logger.info(f"Resizing next_image {key} from {image.shape[1:3]} to {image_resolution}")
                image = image_tools.resize_with_pad(image, *image_resolution)

            if train:
                # Convert from [-1, 1] to [0, 1] for augmax.
                image = image / 2.0 + 0.5

                transforms = []
                if "wrist" not in key:
                    height, width = image.shape[1:3]
                    transforms += [
                        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                        augmax.Resize(width, height),
                        augmax.Rotate((-5, 5)),
                    ]
                transforms += [
                    augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                ]
                sub_rngs = jax.random.split(rng, image.shape[0])
                image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

                # Back to [-1, 1].
                image = image * 2.0 - 1.0

            out_next_images[key] = image
    
        # obtain next_image mask
        for key in out_next_images:
            if key not in observation.next_image_masks:
                # do not mask by default
                out_next_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
            else:
                out_next_masks[key] = jnp.asarray(observation.next_image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        next_images=out_next_images,
        next_image_masks=out_next_masks,
        next_state=observation.next_state,
        done=jnp.asarray(observation.done) if observation.done is not None else None,
        steps_to_episode_end=(
            jnp.asarray(observation.steps_to_episode_end)
            if observation.steps_to_episode_end is not None
            else None
        ),
        relabeled_instruction=jnp.asarray(observation.relabeled_instruction) if observation.relabeled_instruction is not None else None,
        relabeled_action=jnp.asarray(observation.relabeled_action) if observation.relabeled_action is not None else None,
        left_right_flipped_image=(
            jnp.asarray(observation.left_right_flipped_image)
            if observation.left_right_flipped_image is not None
            else None
        ),
        left_right_flipped_action=(
            jnp.asarray(observation.left_right_flipped_action)
            if observation.left_right_flipped_action is not None
            else None
        ),
        relabeled_instruction_similarity=(
            jnp.asarray(observation.relabeled_instruction_similarity)
            if observation.relabeled_instruction_similarity is not None
            else None
        ),
        relabeled_instruction_similarity_weight=(
            jnp.asarray(observation.relabeled_instruction_similarity_weight)
            if observation.relabeled_instruction_similarity_weight is not None
            else None
        ),
        actor_action_chunk_relabeled=(
            jnp.asarray(observation.actor_action_chunk_relabeled)
            if observation.actor_action_chunk_relabeled is not None
            else None
        ),
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
    key_filter: Callable[[tuple[str, ...]], bool] | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.
        key_filter: Optional predicate that receives normalized parameter key paths and returns True for keys to
            restore. If the checkpoint was saved from an NNX state, the trailing "value" path component is removed
            before the predicate is called.

    Returns:
        The restored params.
    """
    if not str(params_path).startswith("gs://"):
        params_path = path_utils.epath_safe_local_path(params_path)

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        params_metadata = metadata["params"]
        if key_filter is not None:
            flat_metadata = traverse_util.flatten_dict(params_metadata)
            filtered_metadata = {
                key_path: value
                for key_path, value in flat_metadata.items()
                if key_filter(key_path[:-1] if key_path and key_path[-1] == "value" else key_path)
            }
            if not filtered_metadata:
                raise ValueError(f"No checkpoint params matched key_filter for {params_path}.")
            params_metadata = traverse_util.unflatten_dict(filtered_metadata)
        item = {"params": params_metadata}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
