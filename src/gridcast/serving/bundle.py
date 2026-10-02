"""Versioned, integrity-checked native LightGBM model bundles."""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import Booster
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import gridcast
from gridcast.columns import Col
from gridcast.features import BASE_FEATURE_COLUMNS, build_forecast_features
from gridcast.metrics import interval_coverage, mean_absolute_error
from gridcast.models import LightGBMLoadForecaster, LightGBMQuantileForecaster
from gridcast.pjm import validate_hourly_load
from gridcast.probabilistic import (
    LOWER_QUANTILE,
    MEDIAN_QUANTILE,
    UPPER_QUANTILE,
    conformal_correction,
)
from gridcast.provenance import dataframe_sha256, file_sha256

BUNDLE_FORMAT_VERSION = 1
MIN_HISTORY_HOURS = 336
MAX_HORIZON_HOURS = 168
MODEL_FILENAMES = ("point.txt", "p10.txt", "p50.txt", "p90.txt")


class BundleIntegrityError(ValueError):
    """Raised when a model bundle is incomplete, unexpected, or tampered with."""


class TrainingManifest(BaseModel):
    """Training-frame provenance captured in a serving manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    start: datetime
    end: datetime
    rows: int = Field(gt=0)
    data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ValidationManifest(BaseModel):
    """Holdout calibration diagnostics captured at package time."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    calibration_hours: int = Field(gt=0)
    point_mae_mw: float = Field(ge=0)
    raw_interval_coverage: float = Field(ge=0, le=1)
    calibrated_interval_coverage: float = Field(ge=0, le=1)


class FallbackManifest(BaseModel):
    """Fallback strategy configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    seasonal_period: int = Field(gt=0)


class DriftReference(BaseModel):
    """Reference distribution used for online feature-drift diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    feature: str
    bin_edges: list[float]
    proportions: list[float]

    @model_validator(mode="after")
    def validate_bins(self) -> "DriftReference":
        """Validate a non-empty, finite histogram reference."""
        if len(self.bin_edges) != len(self.proportions) + 1:
            raise ValueError(
                "bin_edges must have exactly one more value than proportions"
            )
        if len(self.proportions) == 0 or not np.isclose(sum(self.proportions), 1.0):
            raise ValueError("proportions must be non-empty and sum to one")
        if (
            not np.isfinite(self.bin_edges).all()
            or not np.isfinite(self.proportions).all()
        ):
            raise ValueError("drift reference values must be finite")
        if any(
            self.bin_edges[index + 1] <= self.bin_edges[index]
            for index in range(len(self.bin_edges) - 1)
        ):
            raise ValueError("bin_edges must be strictly increasing")
        return self


class RuntimeManifest(BaseModel):
    """Runtime versions recorded for reproducibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    python: str
    lightgbm: str
    numpy: str
    pandas: str
    gridcast: str
    git_commit: str | None


class ModelManifest(BaseModel):
    """Strict JSON manifest for an immutable GridCast serving bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    bundle_format: int = BUNDLE_FORMAT_VERSION
    model_name: str = "gridcast-load"
    model_version: Annotated[
        str, Field(pattern=r"^\d+\.\d+\.\d+([-+][0-9A-Za-z.-]+)?$")
    ]
    created_at: datetime
    feature_columns: list[str]
    horizon_hours: int = Field(ge=1, le=MAX_HORIZON_HOURS)
    min_history_hours: int = Field(ge=MIN_HISTORY_HOURS)
    quantiles: list[float]
    conformal_correction_mw: float = Field(ge=0)
    training: TrainingManifest
    validation: ValidationManifest
    fallback: FallbackManifest
    drift_reference: DriftReference
    runtime: RuntimeManifest
    files: dict[str, str]

    @field_validator("created_at")
    @classmethod
    def validate_aware_datetime(cls, value: datetime) -> datetime:
        """Reject naive creation times."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "ModelManifest":
        """Validate bundle invariants that span manifest fields."""
        if self.bundle_format != BUNDLE_FORMAT_VERSION:
            raise ValueError(f"unsupported bundle format {self.bundle_format}")
        if self.model_name != "gridcast-load":
            raise ValueError("model_name must be gridcast-load")
        if self.feature_columns != BASE_FEATURE_COLUMNS:
            raise ValueError("feature_columns must equal BASE_FEATURE_COLUMNS")
        if self.quantiles != [LOWER_QUANTILE, MEDIAN_QUANTILE, UPPER_QUANTILE]:
            raise ValueError("quantiles must equal [0.1, 0.5, 0.9]")
        if set(self.files) != set(MODEL_FILENAMES):
            raise ValueError("files must contain exactly the required model files")
        if any(not _is_sha256(value) for value in self.files.values()):
            raise ValueError("all file hashes must be SHA-256 hex digests")
        return self


class TrainedBundle(BaseModel):
    """Fitted native boosters and immutable metadata ready to be written."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    manifest: ModelManifest
    point: Booster
    p10: Booster
    p50: Booster
    p90: Booster


class LoadedBundle(TrainedBundle):
    """Verified bundle loaded from a model directory."""

    directory: Path


def train_bundle(
    data: pd.DataFrame,
    model_version: str,
    *,
    n_estimators: int = 300,
    calibration_hours: int = 24 * 7 * 4,
    git_commit: str | None = None,
) -> TrainedBundle:
    """Train, calibrate, and refit the four models used by the serving runtime.

    Calibration metrics and conformal correction are measured with a chronological
    trailing holdout. After measurement, all four estimators are refit on every
    complete feature row so the packaged serving model uses all available data.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical regular hourly training data.
    model_version : str
        Semantic version assigned to this immutable model artifact.
    n_estimators : int, default=300
        LightGBM boosting iterations per model.
    calibration_hours : int, default=672
        Trailing observations held out to estimate calibration diagnostics.
    git_commit : str, optional
        Source revision recorded in the runtime manifest.

    Returns
    -------
    TrainedBundle
        Fitted native boosters and a strict manifest.
    """
    validate_hourly_load(data)
    if n_estimators < 1 or calibration_hours < 1:
        raise ValueError("n_estimators and calibration_hours must be positive")
    features = build_forecast_features(data)
    complete = features.notna().all(axis=1)
    usable_features = features.loc[complete, BASE_FEATURE_COLUMNS]
    usable_target = data.loc[complete, Col.TARGET].astype(float)
    if len(usable_features) <= calibration_hours + 48:
        raise ValueError("insufficient complete rows for training and calibration")
    train_features = usable_features.iloc[:-calibration_hours]
    train_target = usable_target.iloc[:-calibration_hours]
    calibration_features = usable_features.iloc[-calibration_hours:]
    calibration_target = cast(
        np.ndarray[tuple[Any, ...], np.dtype[np.float64]],
        usable_target.iloc[-calibration_hours:].to_numpy(dtype=np.float64),
    )
    point_model = LightGBMLoadForecaster(n_estimators=n_estimators).fit(
        train_features, train_target
    )
    quantile_models = [
        LightGBMQuantileForecaster(quantile, n_estimators=n_estimators).fit(
            train_features, train_target
        )
        for quantile in (LOWER_QUANTILE, MEDIAN_QUANTILE, UPPER_QUANTILE)
    ]
    calibration_quantiles = np.sort(
        np.column_stack(
            [model.predict(calibration_features) for model in quantile_models]
        ),
        axis=1,
    )
    correction = conformal_correction(
        calibration_target, calibration_quantiles[:, 0], calibration_quantiles[:, 2]
    )
    refit_point = LightGBMLoadForecaster(n_estimators=n_estimators).fit(
        usable_features, usable_target
    )
    refit_quantiles = [
        LightGBMQuantileForecaster(quantile, n_estimators=n_estimators).fit(
            usable_features, usable_target
        )
        for quantile in (LOWER_QUANTILE, MEDIAN_QUANTILE, UPPER_QUANTILE)
    ]
    lag_values = usable_features["lag_168h"].to_numpy(dtype=float)
    edges = np.quantile(lag_values, np.linspace(0.0, 1.0, 11)).tolist()
    # Quantile ties are common in small synthetic sets; make edges strictly increasing.
    for index in range(1, len(edges)):
        edges[index] = max(edges[index], np.nextafter(edges[index - 1], np.inf))
    counts, _ = np.histogram(lag_values, bins=edges)
    manifest = ModelManifest(
        model_version=model_version,
        created_at=datetime.now(UTC),
        feature_columns=list(BASE_FEATURE_COLUMNS),
        horizon_hours=MAX_HORIZON_HOURS,
        min_history_hours=MIN_HISTORY_HOURS,
        quantiles=[LOWER_QUANTILE, MEDIAN_QUANTILE, UPPER_QUANTILE],
        conformal_correction_mw=correction,
        training=TrainingManifest(
            start=data[Col.TIMESTAMP].iloc[0],
            end=data[Col.TIMESTAMP].iloc[-1],
            rows=len(data),
            data_sha256=dataframe_sha256(data),
        ),
        validation=ValidationManifest(
            calibration_hours=calibration_hours,
            point_mae_mw=mean_absolute_error(
                calibration_target, point_model.predict(calibration_features)
            ),
            raw_interval_coverage=interval_coverage(
                calibration_target,
                calibration_quantiles[:, 0],
                calibration_quantiles[:, 2],
            ),
            calibrated_interval_coverage=interval_coverage(
                calibration_target,
                calibration_quantiles[:, 0] - correction,
                calibration_quantiles[:, 2] + correction,
            ),
        ),
        fallback=FallbackManifest(seasonal_period=168),
        drift_reference=DriftReference(
            feature="lag_168h",
            bin_edges=edges,
            proportions=(counts / counts.sum()).astype(float).tolist(),
        ),
        runtime=RuntimeManifest(
            python=platform.python_version(),
            lightgbm=lightgbm.__version__,
            numpy=np.__version__,
            pandas=pd.__version__,
            gridcast=getattr(gridcast, "__version__", "0.1.0"),
            git_commit=git_commit,
        ),
        files=dict.fromkeys(MODEL_FILENAMES, "0" * 64),
    )
    return TrainedBundle(
        manifest=manifest,
        point=refit_point.booster,
        p10=refit_quantiles[0].booster,
        p50=refit_quantiles[1].booster,
        p90=refit_quantiles[2].booster,
    )


def write_bundle(trained: TrainedBundle, directory: Path) -> Path:
    """Atomically write a native-LightGBM bundle into an empty directory."""
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty bundle directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    boosters = (trained.point, trained.p10, trained.p50, trained.p90)
    hashes: dict[str, str] = {}
    for filename, booster in zip(MODEL_FILENAMES, boosters, strict=True):
        path = directory / filename
        booster.save_model(str(path))
        digest = file_sha256(path)
        if digest is None:
            raise RuntimeError(f"failed to write {filename}")
        hashes[filename] = digest
    manifest = trained.manifest.model_copy(update={"files": hashes})
    (directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return directory


def load_bundle(directory: Path) -> LoadedBundle:
    """Verify and load a native model bundle without executing serialized Python."""
    expected = set(MODEL_FILENAMES) | {"manifest.json"}
    if not directory.is_dir():
        raise BundleIntegrityError(f"bundle directory does not exist: {directory}")
    observed = {path.name for path in directory.iterdir()}
    if observed != expected:
        raise BundleIntegrityError("bundle files are missing or unexpected")
    try:
        payload: Any = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = ModelManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise BundleIntegrityError(f"invalid bundle manifest: {error}") from error
    for filename, expected_hash in manifest.files.items():
        actual_hash = file_sha256(directory / filename)
        if actual_hash != expected_hash:
            raise BundleIntegrityError(f"checksum mismatch for {filename}")
    try:
        boosters = [
            Booster(model_file=str(directory / filename))
            for filename in MODEL_FILENAMES
        ]
    except lightgbm.basic.LightGBMError as error:
        raise BundleIntegrityError(f"could not load native model: {error}") from error
    return LoadedBundle(
        manifest=manifest,
        point=boosters[0],
        p10=boosters[1],
        p50=boosters[2],
        p90=boosters[3],
        directory=directory,
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
