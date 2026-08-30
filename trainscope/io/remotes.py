"""Remote run-tree access for the CLI.

``trainscope ui --runs s3://bucket/runs`` and ``--run s3://...`` accept
URI-backed paths via fsspec. A remote object store cannot be browsed with
``pathlib.Path``, so this module *materializes* a remote run tree into a
local temporary directory (only the files the UI reads: arrows, JSON, and
checkpoint/RNG sidecars) and returns a local ``Path`` the rest of the code
already understands.

Materialization is a one-shot sync, not a live mount: a remote run being
written *right now* is a snapshot, not a streaming view (object stores
offer no cheap append). For post-mortem inspection of completed runs — the
primary multi-run use case — this is sufficient.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

try:
    import fsspec
    import fsspec.core
except Exception:  # pragma: no cover
    fsspec = None

# Extensions we pull down. Everything else in a run dir (e.g. user data) is
# left behind; the UI never reads it.
_READABLE_SUFFIXES = {".arrow", ".json", ".pt", ".pkl"}


def _is_uri(path: str) -> bool:
    """True when ``path`` looks like a URI (``s3://``, ``gs://``, ...)."""
    return "://" in path


def materialize_run_path(path: str | Path) -> Path:
    """Return a local ``Path`` for ``path``.

    Local paths pass through unchanged. URI paths are materialized into a
    fresh temporary directory and the local copy is returned. The temp dir
    is intentionally not cleaned up here: the caller (the CLI) keeps it alive
    for the lifetime of the UI process.
    """
    if isinstance(path, Path):
        return path
    if not _is_uri(path):
        return Path(path)
    if fsspec is None:
        raise ImportError("Remote storage requires 'fsspec'. Install it with: pip install fsspec")

    fs, root = fsspec.core.url_to_fs(path)
    local_root = Path(tempfile.mkdtemp(prefix="trainscope_remote_"))
    _copy_tree(fs, root, local_root)
    return local_root


def _copy_tree(fs: Any, remote_root: str, local_root: Path) -> None:
    """Recursively copy readable files under ``remote_root`` to ``local_root``."""
    try:
        entries = fs.find(remote_root)
    except Exception:
        try:
            entries = fs.ls(remote_root, detail=True)
            entries = [
                e.get("name") if isinstance(e, dict) else e
                for e in entries
                if not (isinstance(e, dict) and e.get("type") == "directory")
            ]
        except Exception:
            entries = []

    for remote_path in entries:
        if remote_path == remote_root:
            continue
        rel = remote_path[len(remote_root) :].lstrip("/")
        if not rel:
            continue
        if Path(rel).suffix not in _READABLE_SUFFIXES:
            continue
        dest = local_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with fs.open(remote_path, "rb") as src, open(dest, "wb") as out:
                out.write(src.read())
        except Exception:
            # A file written concurrently may be momentarily unreadable; skip
            # it rather than failing the whole materialization.
            continue


__all__ = ["materialize_run_path"]
