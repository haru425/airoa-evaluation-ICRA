import numpy as np


def to_calvin_action_chunk(actions) -> np.ndarray:
    """Converts model outputs into CALVIN's 7D action space with discrete gripper commands."""
    action_chunk = np.asarray(actions, dtype=np.float32)
    if action_chunk.ndim == 0:
        raise ValueError("Expected CALVIN actions to have at least 1 dimension.")
    if action_chunk.shape[-1] < 7:
        raise ValueError(f"Expected CALVIN actions with final dimension >= 7, got {action_chunk.shape}.")

    calvin_actions = np.array(action_chunk[..., :7], dtype=np.float32, copy=True)
    # CALVIN's robot implementation asserts the gripper command is exactly -1 or 1.
    calvin_actions[..., 6] = np.where(calvin_actions[..., 6] >= 0.0, 1.0, -1.0)
    return calvin_actions
