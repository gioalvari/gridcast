from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gridcast.foundation_models import TIMESFM_2P5


class APIModel(BaseModel):
    """Base API model with strict response serialization."""

    model_config = ConfigDict(extra="forbid")


class StrictArtifactModel(APIModel):
    """Base model for generated artifacts without implicit coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class HealthResponse(APIModel):
    """API process health response.

    Parameters
    ----------
    status : str
        Current service status.
    service : str
        Service identifier.
    version : str
        Public API version.
    """

    status: str
    service: str
    version: str


class DatasetMetadata(APIModel):
    """Public PJME dataset metadata.

    Parameters
    ----------
    observations : int
        Number of normalized hourly observations.
    start : datetime
        First observation timestamp.
    end : datetime
        Last observation timestamp.
    years : float
        Approximate coverage in years.
    mean_load_mw : float
        Mean hourly electricity load.
    maximum_load_mw : float
        Maximum observed electricity load.
    """

    observations: int = Field(ge=1)
    start: datetime
    end: datetime
    years: float = Field(gt=0.0)
    mean_load_mw: float = Field(gt=0.0)
    maximum_load_mw: float = Field(gt=0.0)


class ExperimentMetadata(APIModel):
    """Chronological experiment boundaries.

    Parameters
    ----------
    horizon_hours : int
        Forecast horizon in hours.
    validation_folds : int
        Number of validation folds.
    holdout_folds : int
        Number of historical holdout folds.
    validation_start : datetime
        First validation timestamp.
    holdout_start : datetime
        First historical holdout timestamp.
    holdout_end : datetime
        Last historical holdout timestamp.
    """

    horizon_hours: int = Field(ge=1)
    validation_folds: int = Field(ge=1)
    holdout_folds: int = Field(ge=1)
    validation_start: datetime
    holdout_start: datetime
    holdout_end: datetime


class ProbabilisticMetadata(APIModel):
    """Probabilistic calibration metadata.

    Parameters
    ----------
    quantiles : list of float
        Quantile levels produced by the model.
    target_coverage : float
        Nominal interval coverage.
    conformal_correction_mw : float
        Symmetric correction applied to each interval bound.
    raw_coverage : float
        Historical holdout coverage before calibration.
    calibrated_coverage : float
        Historical holdout coverage after calibration.
    hourly_calibrated_coverage : float
        Frozen-test coverage after hour-conditional calibration.
    global_hourly_coverage_mae : float
        Mean absolute hourly deviation from nominal coverage for global calibration.
    conditional_hourly_coverage_mae : float
        Mean absolute hourly deviation for hour-conditional calibration.
    rolling_calibrated_coverage : float
        Prequential holdout coverage with causal rolling calibration.
    global_weekly_coverage_mae : float
        Mean absolute weekly coverage deviation for static global calibration.
    rolling_weekly_coverage_mae : float
        Mean absolute weekly coverage deviation for rolling calibration.
    """

    quantiles: list[float]
    target_coverage: float = Field(ge=0.0, le=1.0)
    conformal_correction_mw: float = Field(ge=0.0)
    raw_coverage: float = Field(ge=0.0, le=1.0)
    calibrated_coverage: float = Field(ge=0.0, le=1.0)
    hourly_calibrated_coverage: float = Field(ge=0.0, le=1.0)
    global_hourly_coverage_mae: float = Field(ge=0.0, le=1.0)
    conditional_hourly_coverage_mae: float = Field(ge=0.0, le=1.0)
    rolling_calibrated_coverage: float = Field(ge=0.0, le=1.0)
    global_weekly_coverage_mae: float = Field(ge=0.0, le=1.0)
    rolling_weekly_coverage_mae: float = Field(ge=0.0, le=1.0)


class MetadataResponse(APIModel):
    """Combined dataset and experiment metadata response."""

    dataset: DatasetMetadata
    experiment: ExperimentMetadata
    probabilistic: ProbabilisticMetadata


class LeaderboardEntry(APIModel):
    """Aggregate point-forecast model result."""

    split: str
    model: str
    label: str
    folds: int = Field(ge=1)
    observations: int = Field(ge=1)
    mae_mw: float = Field(ge=0.0)
    rmse_mw: float = Field(ge=0.0)
    mase: float = Field(ge=0.0)
    improvement_vs_weekly_percent: float


class PointForecast(APIModel):
    """One timestamped point forecast."""

    timestamp: datetime
    actual_mw: float
    prediction_mw: float
    model: str
    model_label: str
    split: str
    fold: int = Field(ge=1)


class PointForecastResponse(APIModel):
    """Point forecasts selected for one chronological fold."""

    split: str
    fold: int
    models: list[str]
    forecasts: list[PointForecast]


class ProbabilisticForecast(APIModel):
    """One timestamped calibrated probabilistic forecast."""

    timestamp: datetime
    actual_mw: float
    p10_mw: float
    p50_mw: float
    p90_mw: float
    p10_calibrated_mw: float
    p90_calibrated_mw: float
    p10_hourly_calibrated_mw: float
    p90_hourly_calibrated_mw: float
    p10_rolling_calibrated_mw: float
    p90_rolling_calibrated_mw: float
    split: str
    fold: int = Field(ge=1)


class ProbabilisticForecastResponse(APIModel):
    """Probabilistic forecasts selected for one chronological fold."""

    split: str
    fold: int
    target_coverage: float
    forecasts: list[ProbabilisticForecast]


class PerformanceMeasurement(APIModel):
    """One environment-qualified local model performance measurement."""

    model: str
    fit_time_ms: float = Field(ge=0.0)
    prediction_median_ms: float = Field(gt=0.0)
    prediction_p95_ms: float = Field(gt=0.0)
    throughput_rows_per_second: float = Field(gt=0.0)
    serialized_model_kib: float = Field(gt=0.0)
    fit_rss_delta_mib: float = Field(ge=0.0)
    horizon_rows: int = Field(ge=1)
    repetitions: int = Field(ge=1)


class PerformanceResponse(APIModel):
    """Local benchmark environment and per-model measurements."""

    environment: dict[str, str | int]
    measurements: list[PerformanceMeasurement]


class PointDecisionResult(APIModel):
    """Point-model result under one synthetic decision scenario."""

    scenario: str
    model: str
    shortage_cost: float = Field(ge=0.0)
    surplus_cost: float = Field(ge=0.0)
    optimal_quantile: float = Field(ge=0.0, le=1.0)
    folds: int = Field(ge=1)
    observations: int = Field(ge=1)
    mean_cost: float = Field(ge=0.0)
    regret_vs_perfect: float = Field(ge=0.0)
    cost_increase_vs_best_pct: float = Field(ge=0.0)


class QuantileDecisionResult(APIModel):
    """Cost-aware quantile result under one synthetic decision scenario."""

    scenario: str
    shortage_cost: float = Field(ge=0.0)
    surplus_cost: float = Field(ge=0.0)
    selected_quantile: float = Field(ge=0.0, le=1.0)
    p50_cost: float = Field(ge=0.0)
    cost_aware_quantile_cost: float = Field(ge=0.0)
    cost_savings_pct: float


class DecisionResponse(APIModel):
    """Point-model and probabilistic scheduling sensitivity results."""

    point_models: list[PointDecisionResult]
    quantile_schedules: list[QuantileDecisionResult]


class FoundationMetrics(StrictArtifactModel):
    """Zero-shot foundation-model historical holdout metrics."""

    model: str
    observations: int = Field(ge=1)
    folds: int = Field(ge=1)
    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    mase: float = Field(ge=0.0)
    p10_pinball_loss: float = Field(ge=0.0)
    p50_pinball_loss: float = Field(ge=0.0)
    p90_pinball_loss: float = Field(ge=0.0)
    raw_80_coverage: float = Field(ge=0.0, le=1.0)
    raw_80_mean_width_mw: float = Field(ge=0.0)


class FoundationTiming(StrictArtifactModel):
    """First-call and warm timings for the foundation-model benchmark."""

    first_call_seconds: float = Field(ge=0.0)
    warm_call_seconds: float = Field(ge=0.0)


class FoundationConfigMetadata(StrictArtifactModel):
    """Fully resolved foundation-model benchmark configuration."""

    model_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    weights_license: str = Field(default="unspecified", min_length=1)
    context_length: int = Field(ge=1)
    horizon: int = Field(ge=1)
    holdout_folds: int = Field(ge=1)
    model_parameters: dict[str, bool | int | float | str]

    @model_validator(mode="after")
    def resolve_legacy_license(self) -> Self:
        """Resolve the known TimesFM 2.5 license in legacy summaries."""
        if self.weights_license != "unspecified":
            return self
        if (
            self.model_name,
            self.model_id,
            self.model_revision,
        ) == (
            TIMESFM_2P5.model_name,
            TIMESFM_2P5.model_id,
            TIMESFM_2P5.model_revision,
        ):
            self.weights_license = TIMESFM_2P5.weights_license
            return self
        raise ValueError("foundation weights license is required")


class FoundationEnvironment(StrictArtifactModel):
    """Reproducible runtime metadata for a foundation-model benchmark."""

    python: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    timesfm: str = Field(min_length=1)
    device: Literal["cpu"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    dependency_lock_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    timesfm_lock_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    installed_packages: dict[str, str]

    @model_validator(mode="after")
    def validate_dependency_lock(self) -> Self:
        """Require a canonical or legacy dependency-lock digest."""
        if self.dependency_lock_sha256 is None and self.timesfm_lock_sha256 is None:
            raise ValueError("foundation dependency lock digest is required")
        if (
            self.dependency_lock_sha256 is not None
            and self.timesfm_lock_sha256 is not None
            and self.dependency_lock_sha256 != self.timesfm_lock_sha256
        ):
            raise ValueError("foundation dependency lock digests do not match")
        return self


class FoundationRuntime(StrictArtifactModel):
    """Stable public subset of foundation-model runtime metadata."""

    python: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    timesfm: str = Field(min_length=1)
    device: Literal["cpu"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timesfm_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FoundationSummary(StrictArtifactModel):
    """Validated on-disk foundation-model benchmark summary."""

    config: FoundationConfigMetadata
    metrics: FoundationMetrics
    timing: FoundationTiming
    environment: FoundationEnvironment

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Ensure aggregate metrics match the configured benchmark dimensions."""
        if self.metrics.model != self.config.model_name:
            raise ValueError("foundation model names do not match")
        if self.metrics.folds != self.config.holdout_folds:
            raise ValueError("foundation fold counts do not match")
        expected_observations = self.config.holdout_folds * self.config.horizon
        if self.metrics.observations != expected_observations:
            raise ValueError("foundation observation count does not match dimensions")
        return self


class FoundationResponse(APIModel):
    """Optional foundation-model configuration, metrics, and runtime."""

    model_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    weights_license: str = Field(min_length=1)
    context_length: int = Field(ge=1)
    horizon: int = Field(ge=1)
    holdout_folds: int = Field(ge=1)
    model_parameters: dict[str, bool | int | float | str]
    timing: FoundationTiming
    metrics: FoundationMetrics
    environment: FoundationRuntime
