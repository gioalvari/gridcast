"""Leakage-safe online inference over verified native model bundles."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gridcast.baselines import SeasonalNaiveForecaster
from gridcast.columns import Col
from gridcast.features import BASE_FEATURE_COLUMNS, build_forecast_features
from gridcast.pjm import validate_hourly_load
from gridcast.serving.bundle import MAX_HORIZON_HOURS, LoadedBundle


@dataclass(frozen=True)
class ForecastResult:
    """Forecast arrays returned by the primary or degraded serving path."""

    timestamps: pd.Series
    point: NDArray[np.float64]
    p10: NDArray[np.float64]
    p50: NDArray[np.float64]
    p90: NDArray[np.float64]
    model_version: str
    degraded: bool
    fallback_reason: str | None


class InvalidHistoryError(ValueError):
    """Raised when client-supplied load history violates the request contract."""


class Predictor:
    """Produce calibrated forecasts from one verified model bundle."""

    def __init__(self, bundle: LoadedBundle) -> None:
        """Initialize prediction with a verified, loaded model bundle."""
        self.bundle = bundle

    def predict(self, history: pd.DataFrame, horizon_hours: int) -> ForecastResult:
        """Forecast up to 168 hours while excluding target values in the horizon."""
        validate_history(history, self.bundle.manifest.min_history_hours)
        if (
            not 1
            <= horizon_hours
            <= min(MAX_HORIZON_HOURS, self.bundle.manifest.horizon_hours)
        ):
            raise ValueError("horizon_hours must be between 1 and 168")
        origin = history[Col.TIMESTAMP].iloc[-1] + pd.Timedelta(hours=1)
        future = pd.DataFrame(
            {
                Col.TIMESTAMP: pd.date_range(
                    start=origin, periods=horizon_hours, freq="h", tz=origin.tz
                ),
                # Any positive placeholder is safe: target features are delayed >=168h.
                Col.TARGET: np.ones(horizon_hours, dtype=float),
            }
        )
        combined = pd.concat([history, future], ignore_index=True)
        features = build_forecast_features(combined).iloc[-horizon_hours:][
            BASE_FEATURE_COLUMNS
        ]
        raw = np.column_stack(
            [
                self.bundle.p10.predict(features),
                self.bundle.p50.predict(features),
                self.bundle.p90.predict(features),
            ]
        )
        ordered = np.sort(raw.astype(np.float64), axis=1)
        correction = self.bundle.manifest.conformal_correction_mw
        return ForecastResult(
            timestamps=future[Col.TIMESTAMP],
            point=np.asarray(self.bundle.point.predict(features), dtype=np.float64),
            p10=ordered[:, 0] - correction,
            p50=ordered[:, 1],
            p90=ordered[:, 2] + correction,
            model_version=self.bundle.manifest.model_version,
            degraded=False,
            fallback_reason=None,
        )


def fallback_forecast(
    history: pd.DataFrame,
    horizon_hours: int,
    model_version: str,
    reason: str,
) -> ForecastResult:
    """Return a weekly seasonal-naive forecast with no uncertainty estimate."""
    validate_history(history, 168)
    if not 1 <= horizon_hours <= MAX_HORIZON_HOURS:
        raise ValueError("horizon_hours must be between 1 and 168")
    point = (
        SeasonalNaiveForecaster(168)
        .fit(history[Col.TARGET].to_numpy(dtype=np.float64))
        .predict(horizon_hours)
    )
    origin = history[Col.TIMESTAMP].iloc[-1] + pd.Timedelta(hours=1)
    return ForecastResult(
        timestamps=pd.Series(
            pd.date_range(start=origin, periods=horizon_hours, freq="h", tz=origin.tz)
        ),
        point=point,
        p10=point.copy(),
        p50=point.copy(),
        p90=point.copy(),
        model_version=model_version,
        degraded=True,
        fallback_reason=reason,
    )


def population_stability_index(
    values: NDArray[np.float64],
    reference_edges: list[float],
    reference_proportions: list[float],
) -> float:
    """Compute epsilon-stabilized PSI against a reference histogram."""
    observed = np.asarray(values, dtype=float)
    edges = np.asarray(reference_edges, dtype=float)
    reference = np.asarray(reference_proportions, dtype=float)
    if observed.ndim != 1 or len(observed) == 0 or not np.isfinite(observed).all():
        raise ValueError("values must be a non-empty finite one-dimensional array")
    if len(edges) != len(reference) + 1 or not np.isclose(reference.sum(), 1.0):
        raise ValueError("reference histogram is invalid")
    actual = np.histogram(observed, bins=edges)[0] / len(observed)
    epsilon = 1e-6
    safe_actual = np.clip(actual, epsilon, None)
    safe_reference = np.clip(reference, epsilon, None)
    return float(
        np.sum((safe_actual - safe_reference) * np.log(safe_actual / safe_reference))
    )


def validate_history(history: pd.DataFrame, min_history_hours: int) -> None:
    """Validate client history before admission to the prediction execution path.

    This public boundary is shared by primary and fallback forecasts so malformed
    client history is consistently identified before resilience controls run.
    """
    if len(history) < min_history_hours:
        raise InvalidHistoryError(
            f"history must contain at least {min_history_hours} hours"
        )
    try:
        validate_hourly_load(history)
    except ValueError as error:
        raise InvalidHistoryError(str(error)) from error
