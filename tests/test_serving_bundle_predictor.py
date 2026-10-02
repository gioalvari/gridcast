import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gridcast.columns import Col
from gridcast.data import generate_synthetic_load
from gridcast.serving.bundle import (
    BundleIntegrityError,
    DriftReference,
    ModelManifest,
    load_bundle,
    train_bundle,
    write_bundle,
)
from gridcast.serving.predictor import (
    InvalidHistoryError,
    Predictor,
    fallback_forecast,
    population_stability_index,
    validate_history,
)


@pytest.mark.parametrize(
    "edges, proportions",
    [
        ([0.0], []),
        ([0.0, 1.0], [0.4]),
        ([0.0, float("inf")], [1.0]),
        ([1.0, 0.0], [1.0]),
    ],
)
def test_drift_reference_validation(
    edges: list[float], proportions: list[float]
) -> None:
    with pytest.raises(ValidationError):
        DriftReference(feature="lag", bin_edges=edges, proportions=proportions)


def test_bundle_failure_paths(bundle_directory: Path, tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        write_bundle(
            train_bundle(
                generate_synthetic_load(periods=1681),
                "1.0.0",
                n_estimators=1,
                calibration_hours=168,
            ),
            bundle_directory,
        )
    with pytest.raises(BundleIntegrityError, match="does not exist"):
        load_bundle(tmp_path / "missing")
    unexpected = tmp_path / "unexpected"
    shutil.copytree(bundle_directory, unexpected)
    (unexpected / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="missing or unexpected"):
        load_bundle(unexpected)
    invalid = tmp_path / "invalid"
    shutil.copytree(bundle_directory, invalid)
    manifest = json.loads((invalid / "manifest.json").read_text())
    manifest["bundle_format"] = 2
    (invalid / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="unsupported bundle format"):
        load_bundle(invalid)
    manifest = load_bundle(bundle_directory).manifest.model_dump()
    manifest["feature_columns"] = ["wrong"]
    with pytest.raises(ValidationError, match="feature_columns"):
        ModelManifest.model_validate(manifest)
    for field, value, message in (
        ("model_name", "wrong", "model_name"),
        ("quantiles", [0.1], "quantiles"),
        ("files", {"point.txt": "0" * 64}, "files"),
        ("files", dict.fromkeys(manifest["files"], "bad"), "hashes"),
    ):
        invalid_manifest = load_bundle(bundle_directory).manifest.model_dump()
        invalid_manifest[field] = value
        with pytest.raises(ValidationError, match=message):
            ModelManifest.model_validate(invalid_manifest)
    with pytest.raises(ValueError, match="insufficient"):
        train_bundle(
            generate_synthetic_load(periods=500),
            "1.0.0",
            n_estimators=1,
            calibration_hours=168,
        )
    with pytest.raises(ValueError, match="positive"):
        train_bundle(generate_synthetic_load(periods=1681), "1.0.0", n_estimators=0)
    corrupt_model = tmp_path / "corrupt-model"
    shutil.copytree(bundle_directory, corrupt_model)
    manifest = json.loads((corrupt_model / "manifest.json").read_text())
    (corrupt_model / "point.txt").write_text("not a booster", encoding="utf-8")
    from gridcast.provenance import file_sha256

    manifest["files"]["point.txt"] = file_sha256(corrupt_model / "point.txt")
    (corrupt_model / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="could not load native model"):
        load_bundle(corrupt_model)


def test_predictor_and_fallback_error_paths(bundle_directory: Path) -> None:
    predictor = Predictor(load_bundle(bundle_directory))
    history = generate_synthetic_load(periods=336)
    with pytest.raises(ValueError, match="horizon_hours"):
        predictor.predict(history, 0)
    with pytest.raises(ValueError, match="at least"):
        predictor.predict(history.iloc[:10], 1)
    irregular = history.copy()
    irregular.loc[1, Col.TIMESTAMP] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="regular hourly"):
        predictor.predict(irregular, 1)
    with pytest.raises(ValueError, match="horizon_hours"):
        fallback_forecast(history, 169, "0.1.0", "test")
    with pytest.raises(ValueError, match="values"):
        population_stability_index(np.array([]), [0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="reference"):
        population_stability_index(np.array([1.0]), [0.0, 1.0], [0.5])


def test_validate_history_raises_dedicated_client_error() -> None:
    history = generate_synthetic_load(periods=336)
    history.loc[1, Col.TIMESTAMP] += pd.Timedelta(minutes=30)
    with pytest.raises(InvalidHistoryError, match="regular hourly"):
        validate_history(history, 336)
