from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.baselines import SeasonalNaiveForecaster
from gridcast.columns import Col
from gridcast.metrics import (
    mean_absolute_error,
    mean_absolute_scaled_error,
    root_mean_squared_error,
)


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for an expanding-window temporal backtest.

    Parameters
    ----------
    initial_window : int
        Number of observations in the first training window.
    horizon : int
        Number of observations forecast in each fold.
    step : int
        Number of observations between consecutive fold origins.
    seasonal_period : int
        Number of observations in one seasonal cycle.
    """

    initial_window: int
    horizon: int
    step: int
    seasonal_period: int

    def __post_init__(self) -> None:
        """Validate window sizes after initialization."""
        if min(self.initial_window, self.horizon, self.step) < 1:
            msg = "initial_window, horizon, and step must be positive"
            raise ValueError(msg)
        if self.seasonal_period < 1:
            msg = "seasonal_period must be positive"
            raise ValueError(msg)
        if self.initial_window <= self.seasonal_period:
            msg = "initial_window must exceed seasonal_period"
            raise ValueError(msg)


@dataclass(frozen=True)
class BacktestResult:
    """Forecast and metric artifacts produced by a temporal backtest.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Out-of-sample actuals and predictions for every fold.
    metrics : pandas.DataFrame
        Evaluation metrics for every fold.
    """

    forecasts: pd.DataFrame
    metrics: pd.DataFrame

    @property
    def summary(self) -> dict[str, float | int]:
        """Return horizon-weighted aggregate metrics.

        Returns
        -------
        dict
            Fold count, observation count, and aggregate metric values.
        """
        weights = self.metrics["observations"].to_numpy(dtype=float)
        return {
            "folds": len(self.metrics),
            "observations": len(self.forecasts),
            "mae": float(np.average(self.metrics["mae"], weights=weights)),
            "rmse": root_mean_squared_error(
                self.forecasts[Col.TARGET].to_numpy(dtype=float),
                self.forecasts[Col.PREDICTION].to_numpy(dtype=float),
            ),
            "mase": float(np.average(self.metrics["mase"], weights=weights)),
        }


def rolling_backtest(data: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Evaluate a weekly seasonal baseline over expanding temporal windows.

    Parameters
    ----------
    data : pandas.DataFrame
        Chronologically ordered timestamps and electricity load.
    config : BacktestConfig
        Window and seasonality configuration.

    Returns
    -------
    BacktestResult
        Out-of-sample forecasts and fold-level metrics.

    Raises
    ------
    ValueError
        If data are invalid or cannot produce an evaluation fold.
    """
    _validate_data(data, config)
    target = data[Col.TARGET].to_numpy(dtype=float)
    forecast_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int]] = []

    for fold, test_start in enumerate(
        range(config.initial_window, len(data) - config.horizon + 1, config.step),
        start=1,
    ):
        test_end = test_start + config.horizon
        training = target[:test_start]
        actual = target[test_start:test_end]
        prediction = (
            SeasonalNaiveForecaster(config.seasonal_period)
            .fit(training)
            .predict(config.horizon)
        )
        cutoff = data[Col.TIMESTAMP].iloc[test_start - 1]

        forecast_frames.append(
            pd.DataFrame(
                {
                    Col.TIMESTAMP: data[Col.TIMESTAMP]
                    .iloc[test_start:test_end]
                    .to_numpy(),
                    Col.TARGET: actual,
                    Col.PREDICTION: prediction,
                    Col.FOLD: fold,
                    Col.CUTOFF: cutoff,
                }
            )
        )
        metric_rows.append(
            {
                Col.FOLD: fold,
                "observations": len(actual),
                "mae": mean_absolute_error(actual, prediction),
                "rmse": root_mean_squared_error(actual, prediction),
                "mase": mean_absolute_scaled_error(
                    actual,
                    prediction,
                    training,
                    config.seasonal_period,
                ),
            }
        )

    if not forecast_frames:
        msg = "data do not contain a complete backtest fold"
        raise ValueError(msg)
    return BacktestResult(
        forecasts=pd.concat(forecast_frames, ignore_index=True),
        metrics=pd.DataFrame(metric_rows),
    )


def _validate_data(data: pd.DataFrame, config: BacktestConfig) -> None:
    required = {Col.TIMESTAMP, Col.TARGET}
    missing = required.difference(data.columns)
    if missing:
        names = ", ".join(sorted(missing))
        msg = f"data are missing required columns: {names}"
        raise ValueError(msg)
    if len(data) < config.initial_window + config.horizon:
        msg = "data must cover the initial window and at least one horizon"
        raise ValueError(msg)
    timestamps = data[Col.TIMESTAMP]
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        msg = f"{Col.TIMESTAMP} must contain datetimes"
        raise ValueError(msg)
    if timestamps.isna().any() or timestamps.duplicated().any():
        msg = "timestamps must be unique and non-null"
        raise ValueError(msg)
    if not timestamps.is_monotonic_increasing:
        msg = "timestamps must be sorted in increasing order"
        raise ValueError(msg)
    target = data[Col.TARGET].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        msg = f"{Col.TARGET} must contain only finite numeric values"
        raise ValueError(msg)
