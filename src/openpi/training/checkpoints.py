from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
import typing
from typing import Protocol, TYPE_CHECKING

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils
import openpi.transforms as _transforms

if TYPE_CHECKING:
    import openpi.training.config as _config


_NORMALIZATION_CONFIG_FILENAME = "normalization_config.json"
_VALID_ACTION_NORM_MODES = {None, *typing.get_args(_transforms.NormMode)}


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    save_train_state: bool = True,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if data_config.asset_id is not None:
            assets_dir = directory / data_config.asset_id
            if norm_stats is not None:
                _normalize.save(assets_dir, norm_stats)
            save_normalization_config(assets_dir, data_config)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "params": {"params": params},
    }
    if save_train_state:
        items["train_state"] = train_state
    else:
        logging.info(
            f"Saving params-only checkpoint at step {step} (train_state omitted to reduce CPU memory usage)."
        )
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    actual_step = step if step is not None else checkpoint_manager.latest_step()

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)

        if _checkpoint_has_train_state(checkpoint_manager, actual_step):
            restored = checkpoint_manager.restore(
                actual_step,
                items={
                    "train_state": train_state,
                    "params": {"params": params},
                },
            )
            return _merge_params(restored["train_state"], restored["params"])
        else:
            logging.warning(
                f"Checkpoint at step {actual_step} does not contain train_state "
                "(params-only checkpoint). Restoring inference params only; "
                "optimizer state will be reinitialized."
            )
            restored = checkpoint_manager.restore(
                actual_step,
                items={
                    "params": {"params": params},
                },
            )
            return _merge_params(train_state, restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


def save_normalization_config(asset_dir: epath.Path | str, data_config: _config.DataConfig) -> None:
    payload = {
        "action_norm_mode": data_config.action_norm_mode,
        "use_quantile_norm": bool(data_config.use_quantile_norm),
    }
    path = epath.Path(asset_dir) / _NORMALIZATION_CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_normalization_config_path(path: epath.Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    action_norm_mode = payload.get("action_norm_mode")
    if action_norm_mode not in _VALID_ACTION_NORM_MODES:
        raise ValueError(
            f"Invalid action_norm_mode in {path}: {action_norm_mode!r}. "
            "Expected one of: quantile, mean_std, mean_only, or null."
        )
    if "use_quantile_norm" in payload and not isinstance(payload["use_quantile_norm"], bool):
        raise ValueError(f"Invalid use_quantile_norm in {path}: expected a boolean.")
    logging.info("Loaded normalization config from %s", path)
    return payload


def load_normalization_config(assets_dir: epath.Path | str, asset_id: str) -> dict[str, object] | None:
    return _load_normalization_config_path(epath.Path(assets_dir) / asset_id / _NORMALIZATION_CONFIG_FILENAME)


def load_normalization_config_from_asset_dir(asset_dir: epath.Path | str) -> dict[str, object] | None:
    return _load_normalization_config_path(epath.Path(asset_dir) / _NORMALIZATION_CONFIG_FILENAME)


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _checkpoint_has_train_state(checkpoint_manager: ocp.CheckpointManager, step: int) -> bool:
    """Return True if the checkpoint at the given step includes train_state."""
    train_state_dir = epath.Path(checkpoint_manager.directory) / str(step) / "train_state"
    return train_state_dir.exists()


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
