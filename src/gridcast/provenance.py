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
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    """Return the current Git commit without failing outside a repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def build_experiment_manifest(
    experiment: str,
    config: dict[str, object],
    datasets: dict[str, pd.DataFrame],
    *,
    features: list[str],
    boundaries: dict[str, str],
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

    Returns
    -------
    dict
        JSON-compatible provenance manifest.
    """
    return {
        "schema_version": 1,
        "experiment": experiment,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
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


def write_manifest(manifest: dict[str, object], path: Path) -> None:
    """Write an experiment manifest as stable formatted JSON."""
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
