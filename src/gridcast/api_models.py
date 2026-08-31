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
    test_folds : int
        Number of frozen test folds.
    validation_start : datetime
        First validation timestamp.
    test_start : datetime
        First frozen-test timestamp.
    test_end : datetime
        Last frozen-test timestamp.
    """

    horizon_hours: int = Field(ge=1)
    validation_folds: int = Field(ge=1)
    test_folds: int = Field(ge=1)
    validation_start: datetime
    test_start: datetime
    test_end: datetime


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
        Frozen-test coverage before calibration.
    calibrated_coverage : float
        Frozen-test coverage after calibration.
    """

    quantiles: list[float]
    target_coverage: float = Field(ge=0.0, le=1.0)
    conformal_correction_mw: float = Field(ge=0.0)
    raw_coverage: float = Field(ge=0.0, le=1.0)
    calibrated_coverage: float = Field(ge=0.0, le=1.0)


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
    split: str
    fold: int = Field(ge=1)


class ProbabilisticForecastResponse(APIModel):
    """Probabilistic forecasts selected for one chronological fold."""

    split: str
    fold: int
    target_coverage: float
    forecasts: list[ProbabilisticForecast]
