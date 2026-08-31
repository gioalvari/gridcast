import json
from pathlib import Path

import pandas as pd

from gridcast.provenance import (
    build_experiment_manifest,
    dataframe_sha256,
    file_sha256,
    write_manifest,
)


def test_dataframe_hash_is_deterministic_and_value_sensitive() -> None:
    first = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    second = first.copy()
    changed = first.assign(value=[1, 3])

    assert dataframe_sha256(first) == dataframe_sha256(second)
    assert dataframe_sha256(first) != dataframe_sha256(changed)


def test_file_hash_and_manifest_are_auditable(tmp_path: Path) -> None:
    file = tmp_path / "input.txt"
    file.write_text("gridcast", encoding="utf-8")
    data = pd.DataFrame({"value": [1.0, 2.0]})

    manifest = build_experiment_manifest(
        "test-experiment",
        {"seed": 42},
        {"load": data},
        features=["lag_168h"],
        boundaries={"holdout_start": "2024-01-01T00:00:00"},
    )
    output = tmp_path / "manifest.json"
    write_manifest(manifest, output)

    assert file_sha256(file) is not None
    assert file_sha256(tmp_path / "missing") is None
    parsed = json.loads(output.read_text())
    assert parsed["schema_version"] == 1
    assert parsed["dataset_sha256"]["load"] == dataframe_sha256(data)
    assert parsed["features"] == ["lag_168h"]
