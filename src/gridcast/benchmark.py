import json
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
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
from gridcast.model_comparison import circular_block_indices
from gridcast.models import (
    create_point_forecaster,
    ensure_point_estimator_available,
    resolved_point_model_parameters,
)
from gridcast.pjm import validate_hourly_load
from gridcast.provenance import build_experiment_manifest, file_sha256, write_manifest

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
HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL = "hist_gradient_boosting_exogenous"
CATBOOST_EXOGENOUS_MODEL = "catboost_exogenous"
XGBOOST_EXOGENOUS_MODEL = "xgboost_exogenous"
AUTOML_EXOGENOUS_MODEL = "automl_exogenous"
EXTENDED_POINT_MODELS = {
    HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL: "hist_gradient_boosting",
    CATBOOST_EXOGENOUS_MODEL: "catboost",
    XGBOOST_EXOGENOUS_MODEL: "xgboost",
}
AUTOML_CANDIDATES = (
    LIGHTGBM_EXOGENOUS_MODEL,
    HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL,
    CATBOOST_EXOGENOUS_MODEL,
    XGBOOST_EXOGENOUS_MODEL,
)
EXTENDED_COMPARISON_SEED = 20_260_906
EXTENDED_COMPARISON_REPLICATES = 100_000
EXTENDED_COMPARISON_BLOCK_LENGTH = 4
EXTENDED_SENSITIVITY_BLOCK_LENGTHS = (2, 4, 6, 8, 13, 26)


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
    extended_models : bool, default=False
        Run optional CatBoost, XGBoost, histogram boosting, and AutoML selection.
    """

    horizon: int = 24 * 7
    validation_folds: int = 12
    holdout_folds: int = 52
    max_train_hours: int = 24 * 365 * 5
    n_estimators: int = 300
    extended_models: bool = False

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
        if self.extended_models and (
            self.horizon != 24 * 7
            or self.validation_folds != 12
            or self.holdout_folds != 52
        ):
            raise ValueError(
                "extended models require the published 168-hour, 12-validation, "
                "52-holdout protocol"
            )


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
    model_selection : dict
        Validation-only AutoML policy audit.
    extended_comparisons : pandas.DataFrame
        Exploratory paired comparisons for optional booster candidates.
    """

    forecasts: pd.DataFrame
    fold_metrics: pd.DataFrame
    leaderboard: pd.DataFrame
    model_selection: dict[str, object] = field(default_factory=dict)
    extended_comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)
    extended_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)


def select_automl_candidate(fold_metrics: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Select the lowest-MAE candidate using validation folds only.

    Parameters
    ----------
    fold_metrics : pandas.DataFrame
        Fold metrics containing validation and optional holdout rows.

    Returns
    -------
    tuple
        Selected model ID and mean validation MAE for every candidate.
    """
    required = {Col.MODEL, Col.SPLIT, "mae"}
    missing = required.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"AutoML fold metrics missing columns: {sorted(missing)}")
    candidates = fold_metrics.loc[
        fold_metrics[Col.SPLIT].eq(VALIDATION_SPLIT)
        & fold_metrics[Col.MODEL].isin(AUTOML_CANDIDATES)
    ]
    if candidates[[Col.MODEL, Col.FOLD, "mae"]].isna().any().any():
        raise ValueError("AutoML validation metrics must be complete")
    if not np.isfinite(candidates["mae"].to_numpy(dtype=np.float64)).all():
        raise ValueError("AutoML validation MAE must be finite")
    if candidates.duplicated([Col.MODEL, Col.FOLD]).any():
        raise ValueError("AutoML validation contains duplicate model folds")
    expected_folds = sorted(candidates[Col.FOLD].unique())
    for _, model_rows in candidates.groupby(Col.MODEL):
        if sorted(model_rows[Col.FOLD].unique()) != expected_folds:
            raise ValueError("AutoML candidates must use identical validation folds")
    scores = candidates.groupby(Col.MODEL, as_index=False).agg(mae=("mae", "mean"))
    if set(scores[Col.MODEL]) != set(AUTOML_CANDIDATES):
        raise ValueError("AutoML selection requires every candidate on validation")
    scores = scores.sort_values(["mae", Col.MODEL], ignore_index=True)
    return str(scores.iloc[0][Col.MODEL]), {
        str(row[Col.MODEL]): float(row["mae"]) for _, row in scores.iterrows()
    }


def extended_package_versions() -> dict[str, str]:
    """Return resolved versions for all extended benchmark backends."""
    return {
        "lightgbm": version("lightgbm"),
        "scikit-learn": version("scikit-learn"),
        "catboost": version("catboost"),
        "xgboost": version("xgboost"),
    }


def build_extended_comparisons(
    fold_metrics: pd.DataFrame,
    *,
    expected_folds: int = 52,
) -> pd.DataFrame:
    """Compare optional booster candidates with LightGBM on paired holdout folds."""
    selected = fold_metrics.loc[
        fold_metrics[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)
        & fold_metrics[Col.MODEL].isin(
            [LIGHTGBM_EXOGENOUS_MODEL, *EXTENDED_POINT_MODELS]
        )
    ]
    if selected[[Col.MODEL, Col.FOLD, "mae"]].isna().any().any():
        raise ValueError("extended comparison metrics must be complete")
    if not np.isfinite(selected["mae"].to_numpy(dtype=np.float64)).all():
        raise ValueError("extended comparison MAE must be finite")
    if selected.duplicated([Col.MODEL, Col.FOLD]).any():
        raise ValueError("extended comparison contains duplicate model folds")
    pivot = selected.pivot(index=Col.FOLD, columns=Col.MODEL, values="mae")
    expected_models = {LIGHTGBM_EXOGENOUS_MODEL, *EXTENDED_POINT_MODELS}
    if set(pivot.columns) != expected_models:
        raise ValueError(
            "extended comparison requires every candidate holdout forecast"
        )
    if (
        list(pivot.index) != list(range(1, expected_folds + 1))
        or pivot.isna().any().any()
    ):
        raise ValueError(
            f"extended comparison requires {expected_folds} paired holdout folds"
        )
    indices = circular_block_indices(
        len(pivot),
        EXTENDED_COMPARISON_BLOCK_LENGTH,
        EXTENDED_COMPARISON_REPLICATES,
        np.random.Generator(np.random.PCG64(EXTENDED_COMPARISON_SEED)),
    )
    alpha = 0.05
    family_alpha = alpha / len(EXTENDED_POINT_MODELS)
    reference = pivot[LIGHTGBM_EXOGENOUS_MODEL].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []
    for candidate in EXTENDED_POINT_MODELS:
        candidate_loss = pivot[candidate].to_numpy(dtype=np.float64)
        difference = reference - candidate_loss
        bootstrap = difference[indices].mean(axis=1)
        marginal = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
        adjusted = np.quantile(
            bootstrap,
            [family_alpha / 2.0, 1.0 - family_alpha / 2.0],
        )
        rows.append(
            {
                "candidate_model": candidate,
                "reference_model": LIGHTGBM_EXOGENOUS_MODEL,
                "mean_mae_improvement_mw": float(difference.mean()),
                "wins": int((difference > 0.0).sum()),
                "folds": len(difference),
                "ci_low_mw": float(marginal[0]),
                "ci_high_mw": float(marginal[1]),
                "adjusted_ci_low_mw": float(adjusted[0]),
                "adjusted_ci_high_mw": float(adjusted[1]),
                "simultaneous_superiority_supported": bool(adjusted[0] > 0.0),
                "exploratory": True,
                "block_length_folds": EXTENDED_COMPARISON_BLOCK_LENGTH,
                "bootstrap_replicates": EXTENDED_COMPARISON_REPLICATES,
                "bootstrap_seed": EXTENDED_COMPARISON_SEED,
                "family_size": len(EXTENDED_POINT_MODELS),
                "familywise_confidence_level": 0.95,
                "adjusted_per_comparison_confidence_level": 1.0 - family_alpha,
            }
        )
    return pd.DataFrame(rows)


def build_extended_comparison_sensitivity(
    fold_metrics: pd.DataFrame,
    *,
    expected_folds: int = 52,
    block_lengths: tuple[int, ...] = EXTENDED_SENSITIVITY_BLOCK_LENGTHS,
) -> pd.DataFrame:
    """Evaluate exploratory conclusions across circular block lengths."""
    selected = fold_metrics.loc[
        fold_metrics[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)
        & fold_metrics[Col.MODEL].isin(
            [LIGHTGBM_EXOGENOUS_MODEL, *EXTENDED_POINT_MODELS]
        )
    ]
    pivot = selected.pivot(index=Col.FOLD, columns=Col.MODEL, values="mae")
    if (
        list(pivot.index) != list(range(1, expected_folds + 1))
        or pivot.isna().any().any()
    ):
        raise ValueError(
            f"extended sensitivity requires {expected_folds} paired holdout folds"
        )
    reference = pivot[LIGHTGBM_EXOGENOUS_MODEL].to_numpy(dtype=np.float64)
    alpha = 0.05
    family_alpha = alpha / len(EXTENDED_POINT_MODELS)
    rows: list[dict[str, object]] = []
    for spec_index, block_length in enumerate(block_lengths, 1):
        seed = (
            EXTENDED_COMPARISON_SEED
            if block_length == EXTENDED_COMPARISON_BLOCK_LENGTH
            else EXTENDED_COMPARISON_SEED + spec_index
        )
        indices = circular_block_indices(
            len(pivot),
            block_length,
            EXTENDED_COMPARISON_REPLICATES,
            np.random.Generator(np.random.PCG64(seed)),
        )
        for candidate in EXTENDED_POINT_MODELS:
            difference = reference - pivot[candidate].to_numpy(dtype=np.float64)
            bootstrap = difference[indices].mean(axis=1)
            adjusted = np.quantile(
                bootstrap,
                [family_alpha / 2.0, 1.0 - family_alpha / 2.0],
            )
            rows.append(
                {
                    "candidate_model": candidate,
                    "block_length_folds": block_length,
                    "bootstrap_seed": seed,
                    "adjusted_ci_low_mw": float(adjusted[0]),
                    "adjusted_ci_high_mw": float(adjusted[1]),
                    "simultaneous_superiority_supported": bool(adjusted[0] > 0.0),
                }
            )
    return pd.DataFrame(rows)


def run_pjme_benchmark(
    data: pd.DataFrame,
    config: BenchmarkConfig | None = None,
    weather: pd.DataFrame | None = None,
) -> BenchmarkResult:
    """Run leakage-safe baselines and selected point models on chronological folds.

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
    if config.extended_models and weather is None:
        raise ValueError("extended models require exogenous weather features")
    if config.extended_models:
        for estimator in EXTENDED_POINT_MODELS.values():
            ensure_point_estimator_available(estimator)
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
        if config.extended_models:
            model_names.extend(EXTENDED_POINT_MODELS)
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
                create_point_forecaster("lightgbm", n_estimators=config.n_estimators)
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
                        create_point_forecaster(
                            "lightgbm", n_estimators=config.n_estimators
                        )
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
                if config.extended_models:
                    full_columns = list(exogenous_features.columns)
                    train_features = exogenous_features.loc[
                        exogenous_features.index[train_start:origin], full_columns
                    ]
                    forecast_features = exogenous_features.loc[
                        exogenous_features.index[origin:end], full_columns
                    ]
                    for model_name, estimator in EXTENDED_POINT_MODELS.items():
                        predictions[model_name] = (
                            create_point_forecaster(
                                estimator,
                                n_estimators=config.n_estimators,
                            )
                            .fit(train_features, target.iloc[train_start:origin])
                            .predict(forecast_features)
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
    model_selection: dict[str, object] = {}
    extended_comparisons = pd.DataFrame()
    extended_sensitivity = pd.DataFrame()
    if config.extended_models:
        selected_model, validation_scores = select_automl_candidate(fold_metrics)
        alias_forecasts = forecasts.loc[
            forecasts[Col.MODEL].eq(selected_model)
            & forecasts[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)
        ].assign(**{Col.MODEL: AUTOML_EXOGENOUS_MODEL})
        alias_metrics = fold_metrics.loc[
            fold_metrics[Col.MODEL].eq(selected_model)
            & fold_metrics[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)
        ].assign(**{Col.MODEL: AUTOML_EXOGENOUS_MODEL})
        forecasts = pd.concat([forecasts, alias_forecasts], ignore_index=True)
        fold_metrics = pd.concat([fold_metrics, alias_metrics], ignore_index=True)
        model_selection = {
            "policy": "lowest mean validation-fold MAE",
            "selection_split": VALIDATION_SPLIT,
            "selected_model": selected_model,
            "candidate_validation_mae": validation_scores,
            "holdout_target_used_for_selection": False,
            "package_versions": extended_package_versions(),
        }
        model_names.append(AUTOML_EXOGENOUS_MODEL)
        extended_comparisons = build_extended_comparisons(
            fold_metrics, expected_folds=config.holdout_folds
        )
        sensitivity_blocks = tuple(
            length
            for length in EXTENDED_SENSITIVITY_BLOCK_LENGTHS
            if length <= config.holdout_folds
        )
        if sensitivity_blocks:
            extended_sensitivity = build_extended_comparison_sensitivity(
                fold_metrics,
                expected_folds=config.holdout_folds,
                block_lengths=sensitivity_blocks,
            )
    leaderboard = _build_leaderboard(forecasts, fold_metrics, model_names)
    return BenchmarkResult(
        forecasts,
        fold_metrics,
        leaderboard,
        model_selection,
        extended_comparisons,
        extended_sensitivity,
    )


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
    if config.extended_models and (
        not result.model_selection
        or result.extended_comparisons.empty
        or result.extended_sensitivity.empty
    ):
        raise ValueError("extended benchmark result bundle is incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.unlink(missing_ok=True)
    for filename in (
        "model_selection.json",
        "extended_comparisons.csv",
        "extended_comparison_sensitivity.csv",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    result.forecasts.to_parquet(
        output_dir / "forecasts.parquet", index=False, compression="snappy"
    )
    result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    result.leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    decision_costs = evaluate_decision_costs(result.forecasts)
    decision_costs.to_csv(output_dir / "decision_costs.csv", index=False)
    if result.model_selection:
        (output_dir / "model_selection.json").write_text(
            json.dumps(result.model_selection, indent=2) + "\n", encoding="utf-8"
        )
    if not result.extended_comparisons.empty:
        result.extended_comparisons.to_csv(
            output_dir / "extended_comparisons.csv", index=False
        )
    if not result.extended_sensitivity.empty:
        result.extended_sensitivity.to_csv(
            output_dir / "extended_comparison_sensitivity.csv", index=False
        )
    metadata = {
        "config": asdict(config),
        "models": result.leaderboard[Col.MODEL].drop_duplicates().tolist(),
        "exogenous_features": bool(
            result.leaderboard[Col.MODEL].eq(LIGHTGBM_EXOGENOUS_MODEL).any()
        ),
        "model_selection": result.model_selection,
        "point_model_parameters": (
            resolved_point_model_parameters(config.n_estimators)
            if config.extended_models
            else {
                "lightgbm": resolved_point_model_parameters(config.n_estimators)[
                    "lightgbm"
                ]
            }
        ),
        "extended_comparison_protocol": (
            {
                "exploratory": True,
                "reference_model": LIGHTGBM_EXOGENOUS_MODEL,
                "family_size": len(EXTENDED_POINT_MODELS),
                "block_length_folds": EXTENDED_COMPARISON_BLOCK_LENGTH,
                "bootstrap_replicates": EXTENDED_COMPARISON_REPLICATES,
                "bootstrap_seed": EXTENDED_COMPARISON_SEED,
                "familywise_confidence_level": 0.95,
                "sensitivity_block_lengths": sorted(
                    result.extended_sensitivity["block_length_folds"]
                    .astype(int)
                    .unique()
                    .tolist()
                ),
            }
            if config.extended_models
            else None
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
    if config.extended_models:
        metadata["artifact_sha256"] = {
            filename: file_sha256(output_dir / filename)
            for filename in (
                "leaderboard.csv",
                "model_selection.json",
                "extended_comparisons.csv",
                "extended_comparison_sensitivity.csv",
            )
        }
    summary_temp = output_dir / ".summary.json.tmp"
    summary_temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary_temp.replace(summary_path)
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
            if group.empty:
                continue
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
