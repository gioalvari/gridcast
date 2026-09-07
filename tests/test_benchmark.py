from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.benchmark import (
    AUTOML_EXOGENOUS_MODEL,
    BASELINE_PERIODS,
    CATBOOST_EXOGENOUS_MODEL,
    HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL,
    LIGHTGBM_EXOGENOUS_MODEL,
    LIGHTGBM_MODEL,
    XGBOOST_EXOGENOUS_MODEL,
    BenchmarkConfig,
    BenchmarkResult,
    build_extended_comparison_sensitivity,
    build_extended_comparisons,
    run_pjme_benchmark,
    select_automl_candidate,
    write_benchmark_artifacts,
)
from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, VALIDATION_SPLIT, Col
from gridcast.data import generate_synthetic_load
from gridcast.extended_models import load_extended_benchmark_bundle


def test_benchmark_separates_validation_and_holdout_folds(tmp_path: Path) -> None:
    data = generate_synthetic_load(periods=24 * 49)
    config = BenchmarkConfig(
        horizon=24 * 7,
        validation_folds=1,
        holdout_folds=1,
        max_train_hours=24 * 21,
        n_estimators=5,
    )

    result = run_pjme_benchmark(data, config)

    models = {*BASELINE_PERIODS, LIGHTGBM_MODEL}
    assert set(result.leaderboard[Col.MODEL]) == models
    assert set(result.leaderboard[Col.SPLIT]) == {
        VALIDATION_SPLIT,
        HISTORICAL_HOLDOUT_SPLIT,
    }
    assert len(result.forecasts) == 2 * len(models) * config.horizon
    assert (result.forecasts[Col.TIMESTAMP] > result.forecasts[Col.CUTOFF]).all()
    validation_end = result.forecasts.loc[
        result.forecasts[Col.SPLIT].eq("validation"), Col.TIMESTAMP
    ].max()
    holdout_start = result.forecasts.loc[
        result.forecasts[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT), Col.TIMESTAMP
    ].min()
    assert validation_end < holdout_start

    write_benchmark_artifacts(result, config, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "decision_costs.csv",
        "decision_costs.png",
        "fold_metrics.csv",
        "forecasts.parquet",
        "latest_holdout_week.png",
        "leaderboard.csv",
        "leaderboard.png",
        "summary.json",
    }


def test_benchmark_config_and_minimum_history_are_validated() -> None:
    with pytest.raises(ValueError, match="positive"):
        BenchmarkConfig(validation_folds=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        BenchmarkConfig(horizon=169)
    with pytest.raises(ValueError, match="warmup"):
        BenchmarkConfig(max_train_hours=100)
    with pytest.raises(ValueError, match="n_estimators"):
        BenchmarkConfig(n_estimators=0)
    with pytest.raises(ValueError, match="published 168-hour"):
        BenchmarkConfig(extended_models=True, holdout_folds=2)

    short = generate_synthetic_load(periods=24 * 21)
    with pytest.raises(ValueError, match="requires at least"):
        run_pjme_benchmark(
            short,
            BenchmarkConfig(
                validation_folds=1,
                holdout_folds=1,
                max_train_hours=24 * 15,
                n_estimators=2,
            ),
        )


def test_benchmark_adds_exogenous_model_when_weather_is_available() -> None:
    data = generate_synthetic_load(periods=24 * 400, start="2016-01-01")
    weather = data[[Col.TIMESTAMP]].assign(**{Col.TEMPERATURE: 10.0})
    config = BenchmarkConfig(
        validation_folds=1,
        holdout_folds=1,
        max_train_hours=24 * 380,
        n_estimators=2,
    )

    result = run_pjme_benchmark(data, config, weather)

    assert {
        "lightgbm_holidays",
        "lightgbm_weather",
        "lightgbm_exogenous",
    }.issubset(result.leaderboard[Col.MODEL])


def test_extended_benchmark_selects_automl_on_validation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = generate_synthetic_load(periods=24 * 820, start="2016-01-01")
    weather = data[[Col.TIMESTAMP]].assign(**{Col.TEMPERATURE: 10.0})

    class FakeForecaster:
        def __init__(self, estimator: str) -> None:
            self.offset = {
                "lightgbm": 30.0,
                "hist_gradient_boosting": 20.0,
                "catboost": 10.0,
                "xgboost": 40.0,
            }[estimator]

        def fit(self, features: object, target: object) -> "FakeForecaster":
            return self

        def predict(self, features: object) -> np.ndarray:
            return np.full(len(features), 1000.0 + self.offset)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "gridcast.benchmark.ensure_point_estimator_available", lambda _: None
    )
    monkeypatch.setattr(
        "gridcast.benchmark.create_point_forecaster",
        lambda estimator, **_: FakeForecaster(estimator),
    )
    monkeypatch.setattr(
        "gridcast.benchmark.extended_package_versions",
        lambda: {
            "lightgbm": "test",
            "scikit-learn": "test",
            "catboost": "test",
            "xgboost": "test",
        },
    )
    result = run_pjme_benchmark(
        data,
        BenchmarkConfig(
            n_estimators=2,
            extended_models=True,
        ),
        weather,
    )

    selected = str(result.model_selection["selected_model"])
    assert selected in {
        LIGHTGBM_EXOGENOUS_MODEL,
        HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL,
        CATBOOST_EXOGENOUS_MODEL,
        XGBOOST_EXOGENOUS_MODEL,
    }
    assert result.model_selection["selection_split"] == VALIDATION_SPLIT
    assert result.model_selection["holdout_target_used_for_selection"] is False
    automl = result.forecasts.loc[
        result.forecasts[Col.MODEL].eq(AUTOML_EXOGENOUS_MODEL)
    ]
    selected_rows = result.forecasts.loc[result.forecasts[Col.MODEL].eq(selected)]
    selected_holdout = selected_rows.loc[
        selected_rows[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)
    ]
    assert np.array_equal(automl[Col.PREDICTION], selected_holdout[Col.PREDICTION])
    assert result.leaderboard.loc[
        result.leaderboard[Col.MODEL].eq(AUTOML_EXOGENOUS_MODEL)
        & result.leaderboard[Col.SPLIT].eq(VALIDATION_SPLIT)
    ].empty


def test_extended_benchmark_requires_weather() -> None:
    data = generate_synthetic_load(periods=24 * 49)
    with pytest.raises(ValueError, match="require exogenous weather"):
        run_pjme_benchmark(
            data,
            BenchmarkConfig(
                n_estimators=2,
                extended_models=True,
            ),
        )


def test_automl_selection_ignores_holdout_metrics() -> None:
    candidates = [
        LIGHTGBM_EXOGENOUS_MODEL,
        HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL,
        CATBOOST_EXOGENOUS_MODEL,
        XGBOOST_EXOGENOUS_MODEL,
    ]
    rows: list[dict[str, object]] = []
    for index, model in enumerate(candidates):
        for fold in range(1, 3):
            rows.extend(
                [
                    {
                        Col.MODEL: model,
                        Col.SPLIT: VALIDATION_SPLIT,
                        Col.FOLD: fold,
                        "mae": index + 1.0,
                    },
                    {
                        Col.MODEL: model,
                        Col.SPLIT: HISTORICAL_HOLDOUT_SPLIT,
                        Col.FOLD: fold,
                        "mae": 1000.0 - index * 100.0,
                    },
                ]
            )
    metrics = pd.DataFrame(rows)

    selected, scores = select_automl_candidate(metrics)
    changed = metrics.copy()
    changed.loc[changed[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT), "mae"] *= -100.0

    assert selected == LIGHTGBM_EXOGENOUS_MODEL
    assert select_automl_candidate(changed) == (selected, scores)


def test_extended_comparisons_are_paired_and_exploratory() -> None:
    candidates = [
        LIGHTGBM_EXOGENOUS_MODEL,
        HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL,
        CATBOOST_EXOGENOUS_MODEL,
        XGBOOST_EXOGENOUS_MODEL,
    ]
    rows = [
        {
            Col.MODEL: model,
            Col.SPLIT: HISTORICAL_HOLDOUT_SPLIT,
            Col.FOLD: fold,
            "mae": 10.0 + index,
        }
        for fold in range(1, 5)
        for index, model in enumerate(candidates)
    ]

    comparisons = build_extended_comparisons(pd.DataFrame(rows), expected_folds=4)

    assert len(comparisons) == 3
    assert comparisons["exploratory"].all()
    assert comparisons["family_size"].eq(3).all()
    assert comparisons["mean_mae_improvement_mw"].tolist() == [-1.0, -2.0, -3.0]

    sensitivity = build_extended_comparison_sensitivity(
        pd.DataFrame(rows),
        expected_folds=4,
        block_lengths=(2, 4),
    )
    assert len(sensitivity) == 6
    assert set(sensitivity["block_length_folds"]) == {2, 4}


def test_extended_writer_creates_hash_validated_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = [
        LIGHTGBM_EXOGENOUS_MODEL,
        HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL,
        CATBOOST_EXOGENOUS_MODEL,
        XGBOOST_EXOGENOUS_MODEL,
        AUTOML_EXOGENOUS_MODEL,
        "seasonal_naive_168h",
    ]
    leaderboard_rows: list[dict[str, object]] = []
    for split, folds, observations in [
        (VALIDATION_SPLIT, 12, 2016),
        (HISTORICAL_HOLDOUT_SPLIT, 52, 8736),
    ]:
        for model in models:
            if split == VALIDATION_SPLIT and model == AUTOML_EXOGENOUS_MODEL:
                continue
            mae = {
                CATBOOST_EXOGENOUS_MODEL: 2800.0,
                AUTOML_EXOGENOUS_MODEL: 2800.0,
                LIGHTGBM_EXOGENOUS_MODEL: 2900.0,
            }.get(model, 2950.0)
            leaderboard_rows.append(
                {
                    Col.SPLIT: split,
                    Col.MODEL: model,
                    "folds": folds,
                    "observations": observations,
                    "mae": mae,
                    "rmse": mae + 100.0,
                    "mase": 0.9,
                    "mae_improvement_vs_weekly_pct": 10.0,
                }
            )
    holdout_mae = {
        LIGHTGBM_EXOGENOUS_MODEL: 2900.0,
        HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL: 2950.0,
        CATBOOST_EXOGENOUS_MODEL: 2800.0,
        XGBOOST_EXOGENOUS_MODEL: 2950.0,
    }
    comparisons = pd.DataFrame(
        [
            {
                "candidate_model": model,
                "reference_model": LIGHTGBM_EXOGENOUS_MODEL,
                "mean_mae_improvement_mw": 2900.0 - holdout_mae[model],
                "wins": 30,
                "folds": 52,
                "ci_low_mw": -50.0,
                "ci_high_mw": 100.0,
                "adjusted_ci_low_mw": -70.0,
                "adjusted_ci_high_mw": 120.0,
                "simultaneous_superiority_supported": False,
                "exploratory": True,
                "block_length_folds": 4,
                "bootstrap_replicates": 100_000,
                "bootstrap_seed": 20_260_906,
                "family_size": 3,
                "familywise_confidence_level": 0.95,
                "adjusted_per_comparison_confidence_level": 1.0 - 0.05 / 3.0,
            }
            for model in [
                HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL,
                CATBOOST_EXOGENOUS_MODEL,
                XGBOOST_EXOGENOUS_MODEL,
            ]
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "candidate_model": model,
                "block_length_folds": block,
                "bootstrap_seed": (
                    20_260_906
                    if block == 4
                    else 20_260_906 + [2, 4, 6, 8, 13, 26].index(block) + 1
                ),
                "adjusted_ci_low_mw": -70.0 if block == 4 else -80.0,
                "adjusted_ci_high_mw": 120.0 if block == 4 else 130.0,
                "simultaneous_superiority_supported": False,
            }
            for model in [
                HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL,
                CATBOOST_EXOGENOUS_MODEL,
                XGBOOST_EXOGENOUS_MODEL,
            ]
            for block in [2, 4, 6, 8, 13, 26]
        ]
    )
    result = BenchmarkResult(
        forecasts=pd.DataFrame(
            {
                Col.TIMESTAMP: pd.date_range("2018-01-01", periods=2, freq="h"),
                Col.TARGET: [1.0, 1.0],
                Col.PREDICTION: [1.0, 1.0],
                Col.MODEL: [LIGHTGBM_EXOGENOUS_MODEL] * 2,
                Col.SPLIT: [VALIDATION_SPLIT, HISTORICAL_HOLDOUT_SPLIT],
                Col.FOLD: [1, 1],
                Col.CUTOFF: pd.date_range("2017-12-31 23:00", periods=2, freq="h"),
            }
        ),
        fold_metrics=pd.DataFrame(),
        leaderboard=pd.DataFrame(leaderboard_rows),
        model_selection={
            "selected_model": CATBOOST_EXOGENOUS_MODEL,
            "selection_split": VALIDATION_SPLIT,
            "holdout_target_used_for_selection": False,
            "candidate_validation_mae": {
                LIGHTGBM_EXOGENOUS_MODEL: 2900.0,
                HIST_GRADIENT_BOOSTING_EXOGENOUS_MODEL: 2950.0,
                CATBOOST_EXOGENOUS_MODEL: 2800.0,
                XGBOOST_EXOGENOUS_MODEL: 2950.0,
            },
            "package_versions": {
                "lightgbm": "1",
                "scikit-learn": "1",
                "catboost": "1",
                "xgboost": "1",
            },
        },
        extended_comparisons=comparisons,
        extended_sensitivity=sensitivity,
    )

    monkeypatch.setattr(
        "gridcast.benchmark.evaluate_decision_costs", lambda _: pd.DataFrame()
    )
    monkeypatch.setattr("gridcast.benchmark._plot_leaderboard", lambda *_: None)
    monkeypatch.setattr("gridcast.benchmark._plot_decision_costs", lambda *_: None)
    monkeypatch.setattr("gridcast.benchmark._plot_latest_holdout_week", lambda *_: None)
    write_benchmark_artifacts(
        result,
        BenchmarkConfig(extended_models=True),
        tmp_path,
    )

    bundle = load_extended_benchmark_bundle(tmp_path)
    assert bundle.selection["selected_model"] == CATBOOST_EXOGENOUS_MODEL
    (tmp_path / "leaderboard.csv").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_extended_benchmark_bundle(tmp_path)
