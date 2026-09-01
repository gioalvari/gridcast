import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, Col
from gridcast.metrics import (
    interval_coverage,
    mean_absolute_error,
    mean_absolute_scaled_error,
    mean_interval_width,
    pinball_loss,
    root_mean_squared_error,
)
from gridcast.pjm import validate_hourly_load
from gridcast.provenance import build_experiment_manifest, write_manifest


class FoundationForecaster(Protocol):
    """Protocol implemented by zero-shot time-series foundation models."""

    def forecast(
        self,
        inputs: list[NDArray[np.float32]],
        horizon: int,
    ) -> "FoundationForecast":
        """Forecast point values and selected quantiles for a context batch."""
        ...


@dataclass(frozen=True)
class FoundationForecast:
    """Point and selected quantile forecasts returned by an adapter."""

    point: NDArray[np.float64]
    p10: NDArray[np.float64]
    p50: NDArray[np.float64]
    p90: NDArray[np.float64]


@dataclass(frozen=True)
class FoundationConfig:
    """Configuration for a zero-shot foundation-model benchmark.

    Parameters
    ----------
    model_name : str
        Stable model name written to forecast artifacts.
    model_id : str
        Public checkpoint identifier.
    model_revision : str
        Immutable checkpoint revision.
    context_length : int, default=1024
        Historical observations supplied to each forecast origin.
    horizon : int, default=168
        Forecast horizon in hours.
    holdout_folds : int, default=52
        Historical holdout folds evaluated.
    model_parameters : dict, optional
        Fully resolved adapter-specific forecast parameters.
    """

    model_name: str
    model_id: str
    model_revision: str
    context_length: int = 1024
    horizon: int = 24 * 7
    holdout_folds: int = 52
    model_parameters: dict[str, bool | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate zero-shot benchmark settings."""
        if not self.model_name or not self.model_id or not self.model_revision:
            msg = "model_name, model_id, and model_revision are required"
            raise ValueError(msg)
        if min(self.context_length, self.horizon, self.holdout_folds) < 1:
            msg = "context, horizon, and holdout folds must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class FoundationResult:
    """Zero-shot forecasts, aggregate metrics, and inference timings."""

    forecasts: pd.DataFrame
    metrics: dict[str, float | int | str]
    first_call_seconds: float
    warm_call_seconds: float


def _validate_forecast(
    forecast: FoundationForecast,
    expected_shape: tuple[int, int],
) -> None:
    arrays = {
        "point": forecast.point,
        "p10": forecast.p10,
        "p50": forecast.p50,
        "p90": forecast.p90,
    }
    for name, values in arrays.items():
        if values.shape != expected_shape:
            msg = f"{name} forecast shape must be {expected_shape}, got {values.shape}"
            raise ValueError(msg)
        if not np.isfinite(values).all():
            msg = "foundation forecasts must contain only finite values"
            raise ValueError(msg)


def build_holdout_contexts(
    data: pd.DataFrame,
    config: FoundationConfig,
) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float64]], list[int]]:
    """Build one strictly historical context per holdout forecast origin.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical hourly load data.
    config : FoundationConfig
        Context, horizon, and fold configuration.

    Returns
    -------
    tuple
        Float32 contexts, float64 actual horizons, and integer origins.
    """
    validate_hourly_load(data)
    required = config.context_length + config.holdout_folds * config.horizon
    if len(data) < required:
        msg = f"foundation benchmark requires at least {required} observations"
        raise ValueError(msg)
    target = data[Col.TARGET].to_numpy(dtype=np.float64)
    holdout_start = len(data) - config.holdout_folds * config.horizon
    origins = list(range(holdout_start, len(data), config.horizon))
    contexts = [
        np.asarray(target[origin - config.context_length : origin], dtype=np.float32)
        for origin in origins
    ]
    actuals = [
        np.asarray(target[origin : origin + config.horizon], dtype=np.float64)
        for origin in origins
    ]
    return contexts, actuals, origins


def run_foundation_benchmark(
    data: pd.DataFrame,
    forecaster: FoundationForecaster,
    config: FoundationConfig,
) -> FoundationResult:
    """Evaluate a zero-shot foundation model on the historical holdout.

    The first call captures lazy compilation and cold inference. The repeated call
    measures warm inference and supplies the forecasts used for evaluation.
    """
    contexts, actuals, origins = build_holdout_contexts(data, config)
    expected_shape = (config.holdout_folds, config.horizon)
    started = perf_counter()
    first_forecast = forecaster.forecast(contexts, config.horizon)
    first_call_seconds = perf_counter() - started
    _validate_forecast(first_forecast, expected_shape)
    started = perf_counter()
    forecast = forecaster.forecast(contexts, config.horizon)
    warm_call_seconds = perf_counter() - started
    _validate_forecast(forecast, expected_shape)

    frames: list[pd.DataFrame] = []
    mase_values: list[float] = []
    target = data[Col.TARGET].to_numpy(dtype=np.float64)
    for fold, (origin, actual) in enumerate(
        zip(origins, actuals, strict=True), start=1
    ):
        prediction = np.asarray(forecast.point[fold - 1], dtype=np.float64)
        p10 = np.asarray(forecast.p10[fold - 1], dtype=np.float64)
        p50 = np.asarray(forecast.p50[fold - 1], dtype=np.float64)
        p90 = np.asarray(forecast.p90[fold - 1], dtype=np.float64)
        ordered = np.sort(np.column_stack([p10, p50, p90]), axis=1)
        mase_values.append(
            mean_absolute_scaled_error(
                actual,
                prediction,
                target[:origin],
                seasonal_period=24 * 7,
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    Col.TIMESTAMP: data[Col.TIMESTAMP]
                    .iloc[origin : origin + config.horizon]
                    .to_numpy(),
                    Col.TARGET: actual,
                    Col.PREDICTION: prediction,
                    Col.P10: ordered[:, 0],
                    Col.P50: ordered[:, 1],
                    Col.P90: ordered[:, 2],
                    Col.MODEL: config.model_name,
                    Col.SPLIT: HISTORICAL_HOLDOUT_SPLIT,
                    Col.FOLD: fold,
                    Col.CUTOFF: data[Col.TIMESTAMP].iloc[origin - 1],
                }
            )
        )
    forecasts = pd.concat(frames, ignore_index=True)
    actual = cast(NDArray[np.float64], forecasts[Col.TARGET].to_numpy(dtype=np.float64))
    prediction = cast(
        NDArray[np.float64], forecasts[Col.PREDICTION].to_numpy(dtype=np.float64)
    )
    p10 = cast(NDArray[np.float64], forecasts[Col.P10].to_numpy(dtype=np.float64))
    p50 = cast(NDArray[np.float64], forecasts[Col.P50].to_numpy(dtype=np.float64))
    p90 = cast(NDArray[np.float64], forecasts[Col.P90].to_numpy(dtype=np.float64))
    metrics: dict[str, float | int | str] = {
        "model": config.model_name,
        "observations": len(forecasts),
        "folds": config.holdout_folds,
        "mae": mean_absolute_error(actual, prediction),
        "rmse": root_mean_squared_error(actual, prediction),
        "mase": float(np.mean(mase_values)),
        "p10_pinball_loss": pinball_loss(actual, p10, 0.1),
        "p50_pinball_loss": pinball_loss(actual, p50, 0.5),
        "p90_pinball_loss": pinball_loss(actual, p90, 0.9),
        "raw_80_coverage": interval_coverage(actual, p10, p90),
        "raw_80_mean_width_mw": mean_interval_width(p10, p90),
    }
    return FoundationResult(
        forecasts,
        metrics,
        first_call_seconds,
        warm_call_seconds,
    )


def write_foundation_artifacts(
    result: FoundationResult,
    config: FoundationConfig,
    data: pd.DataFrame,
    output_dir: Path,
    *,
    environment: dict[str, object],
) -> None:
    """Persist foundation-model forecasts, metrics, and provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.forecasts.to_parquet(
        output_dir / "forecasts.parquet", index=False, compression="snappy"
    )
    report = {
        "config": asdict(config),
        "metrics": result.metrics,
        "timing": {
            "first_call_seconds": result.first_call_seconds,
            "warm_call_seconds": result.warm_call_seconds,
        },
        "environment": environment,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    holdout = result.forecasts
    write_manifest(
        build_experiment_manifest(
            "pjme-timesfm-zero-shot",
            asdict(config),
            {"load": data},
            features=[f"target_context_{config.context_length}h"],
            boundaries={
                "holdout_start": holdout[Col.TIMESTAMP].min().isoformat(),
                "holdout_end": holdout[Col.TIMESTAMP].max().isoformat(),
            },
            environment=environment,
        ),
        output_dir / "experiment_manifest.json",
    )
