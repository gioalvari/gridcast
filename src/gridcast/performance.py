import gc
import json
import pickle
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import pandas as pd
import psutil

from gridcast.baselines import SeasonalNaiveForecaster
from gridcast.columns import Col
from gridcast.features import build_exogenous_features, build_forecast_features
from gridcast.models import LightGBMLoadForecaster
from gridcast.pjm import validate_hourly_load
from gridcast.weather import validate_temperature_data

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

WEEKLY_BASELINE = "seasonal_naive_168h"
LIGHTGBM_BASE = "lightgbm"
LIGHTGBM_EXOGENOUS = "lightgbm_exogenous"


@dataclass(frozen=True)
class PerformanceConfig:
    """Configuration for local training and inference measurements.

    Parameters
    ----------
    horizon : int, default=168
        Number of hourly predictions in each measured request.
    max_train_hours : int, default=43800
        Most recent training history used by LightGBM.
    n_estimators : int, default=300
        Number of LightGBM boosting iterations.
    warmup_runs : int, default=5
        Unrecorded prediction calls before measurement.
    repetitions : int, default=100
        Recorded prediction calls used for latency statistics.
    """

    horizon: int = 24 * 7
    max_train_hours: int = 24 * 365 * 5
    n_estimators: int = 300
    warmup_runs: int = 5
    repetitions: int = 100

    def __post_init__(self) -> None:
        """Validate performance benchmark settings."""
        if (
            min(
                self.horizon,
                self.max_train_hours,
                self.n_estimators,
                self.repetitions,
            )
            < 1
        ):
            msg = "horizon, training size, estimators, and repetitions must be positive"
            raise ValueError(msg)
        if self.warmup_runs < 0:
            msg = "warmup_runs cannot be negative"
            raise ValueError(msg)
        if self.horizon > 24 * 7:
            msg = "horizon cannot exceed the 168-hour feature delay"
            raise ValueError(msg)


@dataclass(frozen=True)
class PerformanceResult:
    """Performance measurements and execution environment metadata.

    Parameters
    ----------
    measurements : pandas.DataFrame
        One row per model with fit, latency, size, and memory metrics.
    environment : dict
        Runtime and hardware information for result interpretation.
    """

    measurements: pd.DataFrame
    environment: dict[str, str | int]


def run_performance_benchmark(
    data: pd.DataFrame,
    weather: pd.DataFrame,
    config: PerformanceConfig | None = None,
) -> PerformanceResult:
    """Measure local inference performance on the latest weekly horizon.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical hourly load history.
    weather : pandas.DataFrame
        Canonical hourly temperature history.
    config : PerformanceConfig, optional
        Measurement settings.

    Returns
    -------
    PerformanceResult
        Per-model measurements and host metadata.

    Raises
    ------
    ValueError
        If the history cannot provide the requested train and forecast windows.
    """
    validate_hourly_load(data)
    validate_temperature_data(weather)
    selected = config or PerformanceConfig()
    if len(data) < selected.horizon + selected.max_train_hours:
        msg = "performance benchmark requires a complete train and forecast window"
        raise ValueError(msg)

    origin = len(data) - selected.horizon
    train_start = origin - selected.max_train_hours
    target = data[Col.TARGET].astype(float)
    base_features = build_forecast_features(data)
    exogenous_features = build_exogenous_features(data, weather)
    process = psutil.Process()
    rows = [
        _measure_baseline(target.iloc[:origin], selected, process),
        _measure_lightgbm(
            LIGHTGBM_BASE,
            base_features.iloc[train_start:origin],
            base_features.iloc[origin:],
            target.iloc[train_start:origin],
            selected,
            process,
        ),
        _measure_lightgbm(
            LIGHTGBM_EXOGENOUS,
            exogenous_features.iloc[train_start:origin],
            exogenous_features.iloc[origin:],
            target.iloc[train_start:origin],
            selected,
            process,
        ),
    ]
    environment: dict[str, str | int] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": psutil.cpu_count(logical=True) or 0,
        "physical_cpu_count": psutil.cpu_count(logical=False) or 0,
    }
    return PerformanceResult(pd.DataFrame(rows), environment)


def write_performance_artifacts(
    result: PerformanceResult,
    config: PerformanceConfig,
    output_dir: Path,
) -> None:
    """Write performance measurements, metadata, and comparison chart.

    Parameters
    ----------
    result : PerformanceResult
        Completed local performance benchmark.
    config : PerformanceConfig
        Settings used for the measurement.
    output_dir : pathlib.Path
        Directory receiving CSV, JSON, and PNG artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result.measurements.to_csv(output_dir / "measurements.csv", index=False)
    report = {
        "config": asdict(config),
        "environment": result.environment,
        "measurements": result.measurements.to_dict(orient="records"),
        "notes": {
            "latency": "warm in-process prediction on one 168-hour request",
            "rss_delta": "indicative process RSS increase during fit; order-sensitive",
            "model_size": "Python pickle size; environment-dependent",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _plot_latency(result.measurements, output_dir / "latency.png")


def load_performance_summary(path: Path) -> dict[str, object]:
    """Load a generated performance summary.

    Parameters
    ----------
    path : pathlib.Path
        JSON summary path.

    Returns
    -------
    dict
        Parsed performance summary.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "performance summary must contain a JSON object"
        raise ValueError(msg)
    return {str(key): value for key, value in payload.items()}


def _measure_baseline(
    target: pd.Series, config: PerformanceConfig, process: psutil.Process
) -> dict[str, float | int | str]:
    rss_before = process.memory_info().rss
    fit_start = time.perf_counter_ns()
    model = SeasonalNaiveForecaster(config.horizon).fit(target.to_numpy(dtype=float))
    fit_ms = _elapsed_ms(fit_start)
    rss_delta = max(0, process.memory_info().rss - rss_before)
    return _measurement_row(
        WEEKLY_BASELINE,
        model,
        lambda: model.predict(config.horizon),
        fit_ms,
        rss_delta,
        config,
    )


def _measure_lightgbm(
    model_name: str,
    training_features: pd.DataFrame,
    forecast_features: pd.DataFrame,
    target: pd.Series,
    config: PerformanceConfig,
    process: psutil.Process,
) -> dict[str, float | int | str]:
    gc.collect()
    rss_before = process.memory_info().rss
    fit_start = time.perf_counter_ns()
    model = LightGBMLoadForecaster(n_estimators=config.n_estimators).fit(
        training_features, target
    )
    fit_ms = _elapsed_ms(fit_start)
    rss_delta = max(0, process.memory_info().rss - rss_before)
    return _measurement_row(
        model_name,
        model,
        lambda: model.predict(forecast_features),
        fit_ms,
        rss_delta,
        config,
    )


def _measurement_row(
    model_name: str,
    model: object,
    predict: Callable[[], object],
    fit_ms: float,
    rss_delta_bytes: int,
    config: PerformanceConfig,
) -> dict[str, float | int | str]:
    for _ in range(config.warmup_runs):
        predict()
    latencies_ms: list[float] = []
    for _ in range(config.repetitions):
        start = time.perf_counter_ns()
        predict()
        latencies_ms.append(_elapsed_ms(start))
    median_ms = statistics.median(latencies_ms)
    p95_ms = _percentile(latencies_ms, 0.95)
    return {
        Col.MODEL: model_name,
        "fit_time_ms": fit_ms,
        "prediction_median_ms": median_ms,
        "prediction_p95_ms": p95_ms,
        "throughput_rows_per_second": config.horizon / (median_ms / 1_000.0),
        "serialized_model_kib": len(pickle.dumps(model)) / 1_024.0,
        "fit_rss_delta_mib": rss_delta_bytes / (1_024.0**2),
        "horizon_rows": config.horizon,
        "repetitions": config.repetitions,
    }


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * probability))
    return ordered[index]


def _plot_latency(measurements: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(
        measurements[Col.MODEL],
        measurements["prediction_median_ms"],
        color=["#9a6fb0", "#42b7c8", "#f05d23"],
    )
    axis.set(title="Warm weekly inference latency", xlabel="Median latency (ms)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
