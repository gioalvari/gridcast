from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """Base API model with strict response serialization."""

    model_config = ConfigDict(extra="forbid")


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
