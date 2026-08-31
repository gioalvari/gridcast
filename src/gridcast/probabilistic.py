import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gridcast.columns import Col
from gridcast.features import EXOGENOUS_WARMUP_HOURS, build_exogenous_features
from gridcast.metrics import (
    interval_coverage,
    mean_absolute_error,
    mean_interval_width,
    pinball_loss,
)
from gridcast.models import LightGBMQuantileForecaster
from gridcast.pjm import validate_hourly_load

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOWER_QUANTILE = 0.1
MEDIAN_QUANTILE = 0.5
UPPER_QUANTILE = 0.9
TARGET_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE


@dataclass(frozen=True)
class ProbabilisticConfig:
    """Configuration for quantile forecasting and conformal calibration.

    Parameters
    ----------
    horizon : int, default=168
        Hours forecast by each weekly fold.
    validation_folds : int, default=12
        Folds used exclusively to estimate conformal correction.
    test_folds : int, default=52
        Frozen folds used for final probabilistic evaluation.
    max_train_hours : int, default=43800
        Most recent training history retained for each model.
    n_estimators : int, default=300
        LightGBM boosting iterations for each quantile and fold.
    """

    horizon: int = 24 * 7
    validation_folds: int = 12
    test_folds: int = 52
    max_train_hours: int = 24 * 365 * 5
    n_estimators: int = 300

    def __post_init__(self) -> None:
        """Validate probabilistic experiment sizes."""
        if min(self.horizon, self.validation_folds, self.test_folds) < 1:
            msg = "horizon and fold counts must be positive"
            raise ValueError(msg)
        if self.horizon > 24 * 7:
            msg = "horizon cannot exceed the 168-hour feature delay"
            raise ValueError(msg)
        if self.max_train_hours <= EXOGENOUS_WARMUP_HOURS:
            msg = "max_train_hours must exceed exogenous feature warmup"
            raise ValueError(msg)
        if self.n_estimators < 1:
            msg = "n_estimators must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class ProbabilisticResult:
    """Forecasts and summary metrics from the probabilistic experiment.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Raw quantiles and calibrated interval bounds for every fold.
    metrics : pandas.DataFrame
        Validation and test probabilistic metrics.
    conformal_correction_mw : float
        Symmetric interval expansion learned from validation residuals.
    """

    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    conformal_correction_mw: float


def conformal_correction(
    actual: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    miscoverage: float = 1.0 - TARGET_COVERAGE,
) -> float:
    """Estimate a finite-sample split-conformal interval correction.

    Parameters
    ----------
    actual : numpy.ndarray
        Calibration observations.
    lower : numpy.ndarray
        Raw lower quantile predictions.
    upper : numpy.ndarray
        Raw upper quantile predictions.
    miscoverage : float, default=0.2
        Desired probability outside the calibrated interval.

    Returns
    -------
    float
        Non-negative symmetric expansion in target units.
    """
    if not 0.0 < miscoverage < 1.0:
        msg = "miscoverage must be strictly between zero and one"
        raise ValueError(msg)
    actual_values = np.asarray(actual, dtype=float)
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    if (
        actual_values.ndim != 1
        or lower_values.ndim != 1
        or upper_values.ndim != 1
        or len(actual_values) == 0
        or len(actual_values) != len(lower_values)
        or len(actual_values) != len(upper_values)
    ):
        msg = "actual and interval bounds must have equal, non-zero lengths"
        raise ValueError(msg)
    if not (
        np.isfinite(actual_values).all()
        and np.isfinite(lower_values).all()
        and np.isfinite(upper_values).all()
    ):
        msg = "actual and interval bounds must contain only finite values"
        raise ValueError(msg)
    if np.any(lower_values > upper_values):
        msg = "lower interval bounds cannot exceed upper bounds"
        raise ValueError(msg)

    scores = np.maximum(lower_values - actual_values, actual_values - upper_values)
    probability = min(
        1.0,
        np.ceil((len(scores) + 1) * (1.0 - miscoverage)) / len(scores),
    )
    return max(0.0, float(np.quantile(scores, probability, method="higher")))


def run_probabilistic_benchmark(
    data: pd.DataFrame,
    weather: pd.DataFrame,
    config: ProbabilisticConfig | None = None,
) -> ProbabilisticResult:
    """Evaluate raw and conformalized quantile forecasts chronologically.

    Validation predictions are generated first and are the only observations
    used to estimate the conformal correction. That frozen correction is then
    applied to all final test folds.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical regular hourly load data.
    weather : pandas.DataFrame
        Canonical regular hourly temperature data.
    config : ProbabilisticConfig, optional
        Evaluation and model configuration.

    Returns
    -------
    ProbabilisticResult
        Timestamped forecasts, metrics, and conformal correction.
    """
    validate_hourly_load(data)
    if config is None:
        config = ProbabilisticConfig()
    required_hours = (
        EXOGENOUS_WARMUP_HOURS
        + (config.validation_folds + config.test_folds) * config.horizon
    )
    if len(data) < required_hours:
        msg = f"probabilistic benchmark requires at least {required_hours} hours"
        raise ValueError(msg)

    features = build_exogenous_features(data, weather)
    target = data[Col.TARGET].astype(float)
    test_start = len(data) - config.test_folds * config.horizon
    validation_start = test_start - config.validation_folds * config.horizon
    validation = _forecast_origins(
        data,
        features,
        target,
        range(validation_start, test_start, config.horizon),
        "validation",
        config,
    )
    correction = conformal_correction(
        validation[Col.TARGET].to_numpy(dtype=np.float64),
        validation[Col.P10].to_numpy(dtype=np.float64),
        validation[Col.P90].to_numpy(dtype=np.float64),
    )
    test = _forecast_origins(
        data,
        features,
        target,
        range(test_start, len(data), config.horizon),
        "test",
        config,
    )
    forecasts = pd.concat([validation, test], ignore_index=True)
    forecasts[Col.P10_CALIBRATED] = forecasts[Col.P10] - correction
    forecasts[Col.P90_CALIBRATED] = forecasts[Col.P90] + correction
    metrics = _probabilistic_metrics(forecasts)
    return ProbabilisticResult(forecasts, metrics, correction)


def write_probabilistic_artifacts(
    result: ProbabilisticResult,
    config: ProbabilisticConfig,
    output_dir: Path,
) -> None:
    """Persist probabilistic forecasts, metrics, metadata, and charts.

    Parameters
    ----------
    result : ProbabilisticResult
        Completed probabilistic experiment.
    config : ProbabilisticConfig
        Configuration used by the run.
    output_dir : pathlib.Path
        Directory receiving experiment artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result.forecasts.to_parquet(
        output_dir / "forecasts.parquet", index=False, compression="snappy"
    )
    result.metrics.to_csv(output_dir / "metrics.csv", index=False)
    test_metrics = result.metrics.loc[result.metrics[Col.SPLIT].eq("test")].iloc[0]
    metadata = {
        "config": asdict(config),
        "quantiles": [LOWER_QUANTILE, MEDIAN_QUANTILE, UPPER_QUANTILE],
        "target_coverage": TARGET_COVERAGE,
        "conformal_correction_mw": result.conformal_correction_mw,
        "test": {
            key: float(test_metrics[key])
            for key in [
                "p10_pinball_loss",
                "p50_pinball_loss",
                "p90_pinball_loss",
                "median_mae",
                "raw_coverage",
                "calibrated_coverage",
                "raw_mean_width_mw",
                "calibrated_mean_width_mw",
            ]
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    _plot_latest_interval(result.forecasts, output_dir / "latest_test_interval.png")
    _plot_calibration(result.metrics, output_dir / "coverage.png")


def _forecast_origins(
    data: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    origins: range,
    split: str,
    config: ProbabilisticConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold, origin in enumerate(origins, start=1):
        end = origin + config.horizon
        train_start = max(0, origin - config.max_train_hours)
        training_features = features.iloc[train_start:origin]
        training_target = target.iloc[train_start:origin]
        forecast_features = features.iloc[origin:end]
        raw_predictions = [
            LightGBMQuantileForecaster(
                quantile=quantile,
                n_estimators=config.n_estimators,
            )
            .fit(training_features, training_target)
            .predict(forecast_features)
            for quantile in [LOWER_QUANTILE, MEDIAN_QUANTILE, UPPER_QUANTILE]
        ]
        ordered = np.sort(np.column_stack(raw_predictions), axis=1)
        frames.append(
            pd.DataFrame(
                {
                    Col.TIMESTAMP: data[Col.TIMESTAMP].iloc[origin:end].to_numpy(),
                    Col.TARGET: target.iloc[origin:end].to_numpy(dtype=float),
                    Col.P10: ordered[:, 0],
                    Col.P50: ordered[:, 1],
                    Col.P90: ordered[:, 2],
                    Col.SPLIT: split,
                    Col.FOLD: fold,
                    Col.CUTOFF: data[Col.TIMESTAMP].iloc[origin - 1],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _probabilistic_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for split in ["validation", "test"]:
        group = forecasts.loc[forecasts[Col.SPLIT].eq(split)]
        actual = cast(NDArray[np.float64], group[Col.TARGET].to_numpy(dtype=np.float64))
        p10 = cast(NDArray[np.float64], group[Col.P10].to_numpy(dtype=np.float64))
        p50 = cast(NDArray[np.float64], group[Col.P50].to_numpy(dtype=np.float64))
        p90 = cast(NDArray[np.float64], group[Col.P90].to_numpy(dtype=np.float64))
        lower = cast(
            NDArray[np.float64],
            group[Col.P10_CALIBRATED].to_numpy(dtype=np.float64),
        )
        upper = cast(
            NDArray[np.float64],
            group[Col.P90_CALIBRATED].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                Col.SPLIT: split,
                "folds": group[Col.FOLD].nunique(),
                "observations": len(group),
                "p10_pinball_loss": pinball_loss(actual, p10, LOWER_QUANTILE),
                "p50_pinball_loss": pinball_loss(actual, p50, MEDIAN_QUANTILE),
                "p90_pinball_loss": pinball_loss(actual, p90, UPPER_QUANTILE),
                "median_mae": mean_absolute_error(actual, p50),
                "raw_coverage": interval_coverage(actual, p10, p90),
                "calibrated_coverage": interval_coverage(actual, lower, upper),
                "raw_mean_width_mw": mean_interval_width(p10, p90),
                "calibrated_mean_width_mw": mean_interval_width(lower, upper),
            }
        )
    return pd.DataFrame(rows)


def _plot_latest_interval(forecasts: pd.DataFrame, output_path: Path) -> None:
    test = forecasts.loc[forecasts[Col.SPLIT].eq("test")]
    latest = test.loc[test[Col.FOLD].eq(test[Col.FOLD].max())]
    figure, axis = plt.subplots(figsize=(14, 5))
    timestamps = latest[Col.TIMESTAMP]
    axis.fill_between(
        timestamps,
        latest[Col.P10_CALIBRATED],
        latest[Col.P90_CALIBRATED],
        color="#7bdff2",
        alpha=0.35,
        label="conformal P10-P90",
    )
    axis.plot(timestamps, latest[Col.TARGET], color="#001524", label="actual")
    axis.plot(timestamps, latest[Col.P50], color="#15616d", label="P50")
    axis.set(
        title="Latest frozen test week with calibrated interval", ylabel="Load (MW)"
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_calibration(metrics: pd.DataFrame, output_path: Path) -> None:
    test = metrics.loc[metrics[Col.SPLIT].eq("test")].iloc[0]
    figure, axis = plt.subplots(figsize=(7, 5))
    labels = ["Target", "Raw", "Conformal"]
    values = [TARGET_COVERAGE, test["raw_coverage"], test["calibrated_coverage"]]
    axis.bar(labels, values, color=["#ff7d00", "#7bdff2", "#15616d"])
    axis.set(title="Frozen test interval coverage", ylabel="Coverage", ylim=(0.0, 1.0))
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
