import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def dataframe_sha256(data: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 digest for a dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Ordered experiment input data.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest including schema and values.
    """
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in data.dtypes.items()]
    digest.update(json.dumps(schema, separators=(",", ":")).encode())
    hashes = np.asarray(pd.util.hash_pandas_object(data, index=True), dtype=np.uint64)
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str | None:
    """Return a file SHA-256 digest or ``None`` when it is absent."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worktree_path_sha256(path: Path) -> str | None:
    """Hash a worktree path without following symbolic links."""
    if path.is_symlink():
        digest = hashlib.sha256(b"symlink\0")
        digest.update(str(path.readlink()).encode())
        return digest.hexdigest()
    return file_sha256(path)


def git_commit() -> str | None:
    """Return the current Git commit without failing outside a repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def git_worktree_state() -> tuple[bool | None, str | None]:
    """Return dirty state and a digest of tracked and untracked changes.

    Returns
    -------
    tuple
        Dirty state and SHA-256 digest, or two ``None`` values outside Git.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        return None, None
    if not status.stdout:
        return False, None

    digest = hashlib.sha256(status.stdout)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        capture_output=True,
        check=False,
    )
    if diff.returncode != 0:
        return True, None
    digest.update(diff.stdout)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=False,
    )
    if untracked.returncode != 0:
        return True, None
    for raw_path in untracked.stdout.split(b"\0"):
        if not raw_path:
            continue
        digest.update(raw_path)
        path = Path(raw_path.decode())
        file_digest = _worktree_path_sha256(path)
        if file_digest is not None:
            digest.update(file_digest.encode())
    return True, digest.hexdigest()


def build_experiment_manifest(
    experiment: str,
    config: dict[str, object],
    datasets: dict[str, pd.DataFrame],
    *,
    features: list[str],
    boundaries: dict[str, str],
    environment: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build an auditable experiment manifest.

    Parameters
    ----------
    experiment : str
        Stable experiment identifier.
    config : dict
        Fully resolved experiment configuration.
    datasets : dict
        Named ordered input dataframes.
    features : list of str
        Model feature names used by the experiment.
    boundaries : dict
        Chronological split boundary timestamps.
    environment : dict, optional
        Fully resolved runtime environment relevant to the experiment.

    Returns
    -------
    dict
        JSON-compatible provenance manifest.
    """
    dirty, worktree_sha256 = git_worktree_state()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment": experiment,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "git_worktree_sha256": worktree_sha256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uv_lock_sha256": file_sha256(Path("uv.lock")),
        "dataset_sha256": {
            name: dataframe_sha256(data) for name, data in datasets.items()
        },
        "config": config,
        "features": features,
        "boundaries": boundaries,
    }
    if environment is not None:
        manifest["environment"] = environment
    return manifest


def write_manifest(manifest: dict[str, object], path: Path) -> None:
    """Write an experiment manifest as stable formatted JSON."""
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
