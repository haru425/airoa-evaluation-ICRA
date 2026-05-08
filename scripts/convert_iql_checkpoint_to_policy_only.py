#!/usr/bin/env python3
"""Convert a pi0.5 IQL checkpoint into a policy-only checkpoint.

The resulting checkpoint directory has the deployment shape expected by
``policy_config.create_trained_policy``:

    <output_checkpoint_dir>/
      params/
      assets/        # copied from the source checkpoint by default

Only actor/policy parameters are restored from the source checkpoint. Critic and
discriminator branches are not loaded or saved.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import shutil
import sys
from typing import Any


logger = logging.getLogger(__name__)


def _import_checkpoint_modules():
    return (
        importlib.import_module("flax.traverse_util"),
        importlib.import_module("numpy"),
        importlib.import_module("orbax.checkpoint"),
    )


def _import_openpi_modules():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir.is_dir() and src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)

    return (
        importlib.import_module("openpi.models.model"),
        importlib.import_module("openpi.policies.policy_config"),
    )


def _default_output_dir(source_checkpoint_dir: Path) -> Path:
    return source_checkpoint_dir.with_name(f"{source_checkpoint_dir.name}_policy_only")


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_output_dir(source_checkpoint_dir: Path, output_checkpoint_dir: Path | None) -> Path:
    source = source_checkpoint_dir.expanduser().resolve()
    output = (output_checkpoint_dir or _default_output_dir(source_checkpoint_dir)).expanduser().resolve()

    if output == source:
        raise ValueError("Output checkpoint directory must be different from the source checkpoint directory.")
    if _is_relative_to(output, source):
        raise ValueError("Output checkpoint directory must not be inside the source checkpoint directory.")
    return output


def _normalized_metadata_keys(params_dir: Path) -> set[tuple[str, ...]]:
    traverse_util, _, ocp = _import_checkpoint_modules()
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(params_dir)["params"]

    keys = set()
    for key_path in traverse_util.flatten_dict(metadata):
        if key_path and key_path[-1] == "value":
            key_path = key_path[:-1]
        keys.add(key_path)
    return keys


def _format_key(key_path: tuple[str, ...]) -> str:
    return "/".join(key_path)


def _check_reference_key_match(params: dict[str, Any], reference_checkpoint_dir: Path | None) -> None:
    if reference_checkpoint_dir is None:
        return

    reference_params_dir = reference_checkpoint_dir.expanduser().resolve() / "params"
    if not reference_params_dir.is_dir():
        raise FileNotFoundError(f"Reference checkpoint is missing params/: {reference_params_dir}")

    traverse_util, _, _ = _import_checkpoint_modules()
    converted_keys = set(traverse_util.flatten_dict(params))
    reference_keys = _normalized_metadata_keys(reference_params_dir)

    missing = sorted(reference_keys - converted_keys)
    extra = sorted(converted_keys - reference_keys)
    if missing or extra:
        lines = [
            "Converted policy parameter keys do not match the reference policy checkpoint.",
            f"missing_from_converted={len(missing)}",
            f"extra_in_converted={len(extra)}",
        ]
        if missing:
            lines.append("first_missing=" + ", ".join(_format_key(key) for key in missing[:10]))
        if extra:
            lines.append("first_extra=" + ", ".join(_format_key(key) for key in extra[:10]))
        raise ValueError("\n".join(lines))

    logger.info("Reference key check passed against %s.", reference_checkpoint_dir)


def _assert_policy_only_params(params: dict[str, Any]) -> None:
    traverse_util, _, _ = _import_checkpoint_modules()
    flat = traverse_util.flatten_dict(params)
    forbidden_prefixes = (
        "action_proj_discriminator",
        "action_proj_q",
        "discriminator",
        "q1",
        "q2",
        "v_",
    )
    forbidden = [
        key_path
        for key_path in flat
        if key_path[0] in {"PaliGemma"} and any(part in {"img_critic", "img_discriminator"} for part in key_path)
        or any(key_path[0].startswith(prefix) for prefix in forbidden_prefixes)
    ]
    if forbidden:
        preview = ", ".join(_format_key(key) for key in sorted(forbidden)[:10])
        raise ValueError(f"Converted params still contain critic/discriminator keys: {preview}")


def _copy_assets(source_checkpoint_dir: Path, output_checkpoint_dir: Path) -> None:
    source_assets = source_checkpoint_dir / "assets"
    output_assets = output_checkpoint_dir / "assets"
    if not source_assets.exists():
        logger.warning("Source checkpoint has no assets/ directory: %s", source_assets)
        return
    shutil.copytree(source_assets, output_assets)
    logger.info("Copied assets: %s -> %s", source_assets, output_assets)


def convert_checkpoint(
    source_checkpoint_dir: Path,
    output_checkpoint_dir: Path,
    *,
    copy_assets: bool,
    overwrite: bool,
    reference_checkpoint_dir: Path | None,
) -> None:
    source_checkpoint_dir = source_checkpoint_dir.expanduser().resolve()
    output_checkpoint_dir = _resolve_output_dir(source_checkpoint_dir, output_checkpoint_dir)
    source_params_dir = source_checkpoint_dir / "params"
    if not source_params_dir.is_dir():
        raise FileNotFoundError(f"Source checkpoint is missing params/: {source_params_dir}")

    if output_checkpoint_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output checkpoint directory already exists: {output_checkpoint_dir}. "
            "Pass --overwrite to replace it."
        )

    traverse_util, np, ocp = _import_checkpoint_modules()
    model_lib, policy_config = _import_openpi_modules()

    logger.info("Restoring policy params from %s.", source_params_dir)
    iql_policy_params = model_lib.restore_params(
        source_params_dir,
        restore_type=np.ndarray,
        key_filter=policy_config._is_iql_policy_param_key,
    )
    policy_params = policy_config._remap_iql_policy_params(iql_policy_params)
    _assert_policy_only_params(policy_params)
    _check_reference_key_match(policy_params, reference_checkpoint_dir)

    if output_checkpoint_dir.exists():
        shutil.rmtree(output_checkpoint_dir)
    output_checkpoint_dir.mkdir(parents=True)

    output_params_dir = output_checkpoint_dir / "params"
    logger.info("Saving policy-only params to %s.", output_params_dir)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(output_params_dir, {"params": policy_params})

    if copy_assets:
        _copy_assets(source_checkpoint_dir, output_checkpoint_dir)

    num_params = len(traverse_util.flatten_dict(policy_params))
    logger.info("Wrote %d policy parameter leaves to %s.", num_params, output_checkpoint_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a pi05_iql_hsr checkpoint into a policy-only pi05_hsr-style checkpoint "
            "without modifying the source checkpoint."
        )
    )
    parser.add_argument(
        "source_checkpoint_dir",
        type=Path,
        help="Source checkpoint step directory, for example .../pi05_iql_hsr/.../100000.",
    )
    parser.add_argument(
        "output_checkpoint_dir",
        nargs="?",
        type=Path,
        help="Output checkpoint step directory. Defaults to a sibling named '<source>_policy_only'.",
    )
    parser.add_argument(
        "--reference-checkpoint-dir",
        type=Path,
        help="Optional pi05_hsr checkpoint step directory used to verify the converted parameter key set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output_checkpoint_dir if it already exists. The source checkpoint is never removed.",
    )
    parser.add_argument(
        "--no-copy-assets",
        action="store_true",
        help="Do not copy assets/ from the source checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    source = args.source_checkpoint_dir.expanduser().resolve()
    output = _resolve_output_dir(source, args.output_checkpoint_dir)
    convert_checkpoint(
        source,
        output,
        copy_assets=not args.no_copy_assets,
        overwrite=args.overwrite,
        reference_checkpoint_dir=args.reference_checkpoint_dir,
    )


if __name__ == "__main__":
    main()
