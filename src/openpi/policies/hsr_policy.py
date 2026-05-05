import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_hsr_example() -> dict:
    """Creates a random input example for the HSR policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class HSRInputs(transforms.DataTransformFn):
    """Converts HSR observations into the common model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_wrist_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "next_observation/state" in data:
            inputs["next_state"] = data["next_observation/state"]
            inputs["next_image"] = {
                "base_0_rgb": _parse_image(data["next_observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["next_observation/wrist_image"]),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            }
            inputs["next_image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_wrist_mask,
            }

        if "done" in data:
            inputs["done"] = data["done"]

        if "steps_to_episode_end" in data:
            inputs["steps_to_episode_end"] = data["steps_to_episode_end"]

        if "relabeled_instruction" in data:
            inputs["relabeled_instruction"] = data["relabeled_instruction"]

        if "relabeled_action" in data:
            inputs["relabeled_action"] = data["relabeled_action"]

        if "left_right_flipped_image" in data:
            inputs["left_right_flipped_image"] = data["left_right_flipped_image"]

        if "left_right_flipped_action" in data:
            inputs["left_right_flipped_action"] = data["left_right_flipped_action"]

        if "relabeled_instruction_similarity" in data:
            inputs["relabeled_instruction_similarity"] = data["relabeled_instruction_similarity"]

        if "relabeled_instruction_similarity_weight" in data:
            inputs["relabeled_instruction_similarity_weight"] = data["relabeled_instruction_similarity_weight"]

        if "actor_action_chunk_relabeled" in data:
            inputs["actor_action_chunk_relabeled"] = data["actor_action_chunk_relabeled"]

        return inputs


@dataclasses.dataclass(frozen=True)
class HSROutputs(transforms.DataTransformFn):
    """Converts model outputs back into the 11D HSR relative-action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :11])}
