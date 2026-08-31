import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gridcast.baselines import SeasonalNaiveForecaster
from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, VALIDATION_SPLIT, Col
from gridcast.decision import evaluate_decision_costs
from gridcast.features import (
    BASE_FEATURE_COLUMNS,
    EXOGENOUS_WARMUP_HOURS,
    FEATURE_WARMUP_HOURS,
    HOLIDAY_FEATURE_COLUMNS,
    WEATHER_FEATURE_COLUMNS,
    build_exogenous_features,
    build_forecast_features,
)
from gridcast.metrics import (
    mean_absolute_error,
    mean_absolute_scaled_error,
    root_mean_squared_error,
)
from gridcast.models import LightGBMLoadForecaster
from gridcast.pjm import validate_hourly_load
from gridcast.provenance import build_experiment_manifest, write_manifest

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

WEEKLY_NAIVE = "seasonal_naive_168h"
BASELINE_PERIODS = {
    "persistence_1h": 1,
    "seasonal_naive_24h": 24,
    WEEKLY_NAIVE: 24 * 7,
}
LIGHTGBM_MODEL = "lightgbm"
LIGHTGBM_HOLIDAY_MODEL = "lightgbm_holidays"
LIGHTGBM_WEATHER_MODEL = "lightgbm_weather"
LIGHTGBM_EXOGENOUS_MODEL = "lightgbm_exogenous"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the PJME development and historical benchmark.

    Parameters
    ----------
    horizon : int, default=168
        Hours forecast by each operational fold.
    validation_folds : int, default=12
        Weekly folds immediately preceding the historical holdout.
    holdout_folds : int, default=52
        Historical holdout weekly folds at the end of the dataset.
    max_train_hours : int, default=43800
        Most recent training history retained for LightGBM.
    n_estimators : int, default=300
        LightGBM boosting iterations per fold.
    """

    horizon: int = 24 * 7
    validation_folds: int = 12
    holdout_folds: int = 52
    max_train_hours: int = 24 * 365 * 5
    n_estimators: int = 300

    def __post_init__(self) -> None:
        """Validate benchmark sizes."""
        if min(self.horizon, self.validation_folds, self.holdout_folds) < 1:
            msg = "horizon and fold counts must be positive"
            raise ValueError(msg)
        if self.horizon > 24 * 7:
            msg = "horizon cannot exceed the 168-hour feature delay"
            raise ValueError(msg)
        if self.max_train_hours <= FEATURE_WARMUP_HOURS:
            msg = "max_train_hours must exceed feature warmup"
            raise ValueError(msg)
        if self.n_estimators < 1:
            msg = "n_estimators must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class BenchmarkResult:
    """Forecasts and metrics from the multi-model benchmark.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Timestamped predictions for every model and fold.
    fold_metrics : pandas.DataFrame
        Metrics for every model and fold.
    leaderboard : pandas.DataFrame
        Aggregate metrics by split and model.
    """

    forecasts: pd.DataFrame
    fold_metrics: pd.DataFrame
    leaderboard: pd.DataFrame


def run_pjme_benchmark(
    data: pd.DataFrame,
    config: BenchmarkConfig | None = None,
    weather: pd.DataFrame | None = None,
) -> BenchmarkResult:
    """Run leakage-safe baselines and LightGBM on chronological folds.

    The final ``holdout_folds`` follow the preceding validation folds.
    Each fold trains only on earlier rows and predicts one complete horizon.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical regular hourly load data.
    config : BenchmarkConfig, optional
        Evaluation and model configuration.
    weather : pandas.DataFrame, optional
        Hourly temperature data. When provided, adds a separate exogenous model.

    Returns
    -------
    BenchmarkResult
        Detailed forecasts, fold metrics, and aggregate leaderboard.
    """
    validate_hourly_load(data)
    if config is None:
        config = BenchmarkConfig()
    warmup_hours = (
        EXOGENOUS_WARMUP_HOURS if weather is not None else FEATURE_WARMUP_HOURS
    )
    required_hours = (
        warmup_hours + (config.validation_folds + config.holdout_folds) * config.horizon
    )
    if len(data) < required_hours:
        msg = f"benchmark requires at least {required_hours} hourly observations"
        raise ValueError(msg)

    features = build_forecast_features(data)
    exogenous_features = (
        build_exogenous_features(data, weather) if weather is not None else None
    )
    model_names = [*BASELINE_PERIODS, LIGHTGBM_MODEL]
    if exogenous_features is not None:
        model_names.extend(
            [
                LIGHTGBM_HOLIDAY_MODEL,
                LIGHTGBM_WEATHER_MODEL,
                LIGHTGBM_EXOGENOUS_MODEL,
            ]
        )
    target = data[Col.TARGET].astype(float)
    holdout_start = len(data) - config.holdout_folds * config.horizon
    validation_start = holdout_start - config.validation_folds * config.horizon
    split_origins = {
        VALIDATION_SPLIT: range(validation_start, holdout_start, config.horizon),
        HISTORICAL_HOLDOUT_SPLIT: range(holdout_start, len(data), config.horizon),
    }
    forecast_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []

    for split, origins in split_origins.items():
        for fold, origin in enumerate(origins, start=1):
            end = origin + config.horizon
            training = cast(
                NDArray[np.float64], target.iloc[:origin].to_numpy(dtype=np.float64)
            )
            actual = cast(
                NDArray[np.float64],
                target.iloc[origin:end].to_numpy(dtype=np.float64),
            )
            predictions = {
                model_name: SeasonalNaiveForecaster(period)
                .fit(training)
                .predict(config.horizon)
                for model_name, period in BASELINE_PERIODS.items()
            }
            train_start = max(0, origin - config.max_train_hours)
            predictions[LIGHTGBM_MODEL] = (
                LightGBMLoadForecaster(n_estimators=config.n_estimators)
                .fit(
                    features.iloc[train_start:origin],
                    target.iloc[train_start:origin],
                )
                .predict(features.iloc[origin:end])
            )
            if exogenous_features is not None:
                feature_sets = {
                    LIGHTGBM_HOLIDAY_MODEL: [
                        *BASE_FEATURE_COLUMNS,
                        *HOLIDAY_FEATURE_COLUMNS,
                    ],
                    LIGHTGBM_WEATHER_MODEL: [
                        *BASE_FEATURE_COLUMNS,
                        *WEATHER_FEATURE_COLUMNS,
                    ],
                    LIGHTGBM_EXOGENOUS_MODEL: list(exogenous_features.columns),
                }
                for model_name, columns in feature_sets.items():
                    predictions[model_name] = (
                        LightGBMLoadForecaster(n_estimators=config.n_estimators)
                        .fit(
                            exogenous_features.loc[
                                exogenous_features.index[train_start:origin], columns
                            ],
                            target.iloc[train_start:origin],
                        )
                        .predict(
                            exogenous_features.loc[
                                exogenous_features.index[origin:end], columns
                            ]
                        )
                    )
            cutoff = data[Col.TIMESTAMP].iloc[origin - 1]

            for model_name, prediction in predictions.items():
                forecast_frames.append(
                    pd.DataFrame(
                        {
                            Col.TIMESTAMP: data[Col.TIMESTAMP]
                            .iloc[origin:end]
                            .to_numpy(),
                            Col.TARGET: actual,
                            Col.PREDICTION: prediction,
                            Col.MODEL: model_name,
                            Col.SPLIT: split,
                            Col.FOLD: fold,
                            Col.CUTOFF: cutoff,
                        }
                    )
                )
                metric_rows.append(
                    {
                        Col.MODEL: model_name,
                        Col.SPLIT: split,
                        Col.FOLD: fold,
                        "observations": len(actual),
                        "mae": mean_absolute_error(actual, prediction),
                        "rmse": root_mean_squared_error(actual, prediction),
                        "mase": mean_absolute_scaled_error(
                            actual, prediction, training, 24 * 7
                        ),
                    }
                )

    forecasts = pd.concat(forecast_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    leaderboard = _build_leaderboard(forecasts, fold_metrics, model_names)
    return BenchmarkResult(forecasts, fold_metrics, leaderboard)


def write_benchmark_artifacts(
    result: BenchmarkResult,
    config: BenchmarkConfig,
    output_dir: Path,
    data: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> None:
    """Persist benchmark tables, metadata, and diagnostic charts.

    Parameters
    ----------
    result : BenchmarkResult
        Completed benchmark output.
    config : BenchmarkConfig
        Configuration used by the run.
    output_dir : pathlib.Path
        Directory receiving benchmark artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result.forecasts.to_parquet(
        output_dir / "forecasts.parquet", index=False, compression="snappy"
    )
    result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    result.leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    decision_costs = evaluate_decision_costs(result.forecasts)
    decision_costs.to_csv(output_dir / "decision_costs.csv", index=False)
    metadata = {
        "config": asdict(config),
        "models": result.leaderboard[Col.MODEL].drop_duplicates().tolist(),
        "exogenous_features": bool(
            result.leaderboard[Col.MODEL].eq(LIGHTGBM_EXOGENOUS_MODEL).any()
        ),
        "validation_start": result.forecasts.loc[
            result.forecasts[Col.SPLIT].eq(VALIDATION_SPLIT), Col.TIMESTAMP
        ]
        .min()
        .isoformat(),
        "holdout_start": result.forecasts.loc[
            result.forecasts[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT), Col.TIMESTAMP
        ]
        .min()
        .isoformat(),
        "holdout_end": result.forecasts.loc[
            result.forecasts[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT), Col.TIMESTAMP
        ]
        .max()
        .isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if data is not None:
        datasets = {"load": data}
        if weather is not None:
            datasets["weather"] = weather
        features = (
            list(build_exogenous_features(data, weather).columns)
            if weather is not None
            else list(build_forecast_features(data).columns)
        )
        write_manifest(
            build_experiment_manifest(
                "pjme-point-benchmark",
                asdict(config),
                datasets,
                features=features,
                boundaries={
                    "validation_start": metadata["validation_start"],
                    "holdout_start": metadata["holdout_start"],
                    "holdout_end": metadata["holdout_end"],
                },
            ),
            output_dir / "experiment_manifest.json",
        )
    _plot_leaderboard(result.leaderboard, output_dir / "leaderboard.png")
    _plot_decision_costs(decision_costs, output_dir / "decision_costs.png")
    _plot_latest_holdout_week(result.forecasts, output_dir / "latest_holdout_week.png")


def _build_leaderboard(
    forecasts: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for split in [VALIDATION_SPLIT, HISTORICAL_HOLDOUT_SPLIT]:
        for model in model_names:
            group = forecasts.loc[
                forecasts[Col.SPLIT].eq(split) & forecasts[Col.MODEL].eq(model)
            ]
            actual = cast(
                NDArray[np.float64],
                group[Col.TARGET].to_numpy(dtype=np.float64),
            )
            prediction = cast(
                NDArray[np.float64],
                group[Col.PREDICTION].to_numpy(dtype=np.float64),
            )
            model_folds = fold_metrics.loc[
                fold_metrics[Col.SPLIT].eq(split) & fold_metrics[Col.MODEL].eq(model)
            ]
            rows.append(
                {
                    Col.SPLIT: split,
                    Col.MODEL: model,
                    "folds": model_folds[Col.FOLD].nunique(),
                    "observations": len(group),
                    "mae": mean_absolute_error(actual, prediction),
                    "rmse": root_mean_squared_error(actual, prediction),
                    "mase": float(model_folds["mase"].mean()),
                }
            )
    leaderboard = pd.DataFrame(rows)
    weekly_mae = (
        leaderboard.loc[leaderboard[Col.MODEL].eq(WEEKLY_NAIVE)]
        .set_index(Col.SPLIT)["mae"]
        .to_dict()
    )
    leaderboard["mae_improvement_vs_weekly_pct"] = [
        100.0 * (weekly_mae[split] - mae) / weekly_mae[split]
        for split, mae in zip(leaderboard[Col.SPLIT], leaderboard["mae"], strict=True)
    ]
    return leaderboard.sort_values([Col.SPLIT, "mae"], ignore_index=True)


def _plot_leaderboard(leaderboard: pd.DataFrame, output_path: Path) -> None:
    holdout = leaderboard.loc[
        leaderboard[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)
    ].sort_values("mae")
    figure, axis = plt.subplots(figsize=(9, 5))
    colors = [
        "#15616d" if str(model).startswith("lightgbm") else "#ff7d00"
        for model in holdout[Col.MODEL]
    ]
    axis.barh(holdout[Col.MODEL], holdout["mae"], color=colors)
    axis.invert_yaxis()
    axis.set(title="PJME historical holdout benchmark", xlabel="MAE (MW)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_latest_holdout_week(forecasts: pd.DataFrame, output_path: Path) -> None:
    holdout = forecasts.loc[forecasts[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)]
    latest_fold = int(holdout[Col.FOLD].max())
    latest = holdout.loc[holdout[Col.FOLD].eq(latest_fold)]
    actual = latest.loc[latest[Col.MODEL].eq(WEEKLY_NAIVE)]
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(
        actual[Col.TIMESTAMP], actual[Col.TARGET], label="actual", color="#001524"
    )
    model_names = [WEEKLY_NAIVE, LIGHTGBM_MODEL]
    if latest[Col.MODEL].eq(LIGHTGBM_EXOGENOUS_MODEL).any():
        model_names.append(LIGHTGBM_EXOGENOUS_MODEL)
    for model_name in model_names:
        model = latest.loc[latest[Col.MODEL].eq(model_name)]
        axis.plot(
            model[Col.TIMESTAMP],
            model[Col.PREDICTION],
            label=model_name,
            linewidth=1.5,
        )
    axis.set(title="Latest historical holdout week", ylabel="Load (MW)")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_decision_costs(costs: pd.DataFrame, output_path: Path) -> None:
    scenarios = list(costs["scenario"].drop_duplicates())
    models = list(costs[Col.MODEL].drop_duplicates())
    positions = np.arange(len(models), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(12, 6))
    for index, scenario in enumerate(scenarios):
        scenario_costs = costs.loc[costs["scenario"].eq(scenario)].set_index(Col.MODEL)[
            "mean_cost"
        ]
        axis.bar(
            positions + (index - 1) * width,
            [scenario_costs[model] for model in models],
            width=width,
            label=scenario.replace("_", " "),
        )
    axis.set(
        title="Historical holdout decision cost by scenario",
        ylabel="Mean synthetic cost units",
        xticks=positions,
        xticklabels=models,
    )
    axis.tick_params(axis="x", rotation=30)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
