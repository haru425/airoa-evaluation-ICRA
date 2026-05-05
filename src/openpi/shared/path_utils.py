import hashlib
import os
import pathlib
import tempfile


_EPATH_RESERVED_LOCAL_PREFIXES = ("/gs/",)
_EPATH_ALIAS_DIRNAME = "openpi_epath_aliases"


def absolute_local_path(path: pathlib.Path | str) -> pathlib.Path:
    """Return an absolute local path without resolving symlinks."""
    local_path = pathlib.Path(path).expanduser()
    if local_path.is_absolute():
        return local_path
    return pathlib.Path.cwd() / local_path


def epath_safe_local_path(path: pathlib.Path | str) -> pathlib.Path:
    """Return a local path that is safe to pass to etils.epath / Orbax.

    etils.epath treats absolute paths under `/gs/` as a GCS shorthand. On systems where
    shared filesystems are mounted at `/gs/...`, we expose the same local target through a
    symlink rooted outside the reserved prefix.
    """
    local_path = absolute_local_path(path)
    if not _needs_epath_local_alias(local_path):
        return local_path

    alias_root = pathlib.Path(tempfile.gettempdir()) / _EPATH_ALIAS_DIRNAME / _current_user_tag()
    alias_root.mkdir(parents=True, exist_ok=True)

    suffix = local_path.name or "root"
    digest = hashlib.sha256(os.fspath(local_path).encode("utf-8")).hexdigest()[:16]
    alias_path = alias_root / f"{digest}-{suffix}"

    _ensure_alias(alias_path, local_path)
    return alias_path


def _needs_epath_local_alias(path: pathlib.Path) -> bool:
    path_str = os.fspath(path)
    return any(path_str.startswith(prefix) for prefix in _EPATH_RESERVED_LOCAL_PREFIXES)


def _current_user_tag() -> str:
    if hasattr(os, "getuid"):
        return str(os.getuid())
    return "default"


def _ensure_alias(alias_path: pathlib.Path, target_path: pathlib.Path) -> None:
    if alias_path.is_symlink():
        if alias_path.resolve() == target_path.resolve():
            return
        alias_path.unlink()
    elif alias_path.exists():
        raise FileExistsError(f"Cannot create epath-safe alias at {alias_path}: path already exists.")

    alias_path.symlink_to(target_path, target_is_directory=target_path.is_dir())
