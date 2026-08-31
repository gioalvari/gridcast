from pathlib import Path

import pytest

from gridcast.benchmark import (
    BASELINE_PERIODS,
    LIGHTGBM_MODEL,
    BenchmarkConfig,
    run_pjme_benchmark,
    write_benchmark_artifacts,
)
from gridcast.columns import Col
from gridcast.data import generate_synthetic_load


def test_benchmark_separates_validation_and_test_folds(tmp_path: Path) -> None:
    data = generate_synthetic_load(periods=24 * 49)
    config = BenchmarkConfig(
        horizon=24 * 7,
        validation_folds=1,
        test_folds=1,
        max_train_hours=24 * 21,
        n_estimators=5,
    )

    result = run_pjme_benchmark(data, config)

    models = {*BASELINE_PERIODS, LIGHTGBM_MODEL}
    assert set(result.leaderboard[Col.MODEL]) == models
    assert set(result.leaderboard[Col.SPLIT]) == {"validation", "test"}
    assert len(result.forecasts) == 2 * len(models) * config.horizon
    assert (result.forecasts[Col.TIMESTAMP] > result.forecasts[Col.CUTOFF]).all()
    validation_end = result.forecasts.loc[
        result.forecasts[Col.SPLIT].eq("validation"), Col.TIMESTAMP
    ].max()
    test_start = result.forecasts.loc[
        result.forecasts[Col.SPLIT].eq("test"), Col.TIMESTAMP
    ].min()
    assert validation_end < test_start

    write_benchmark_artifacts(result, config, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "fold_metrics.csv",
        "forecasts.parquet",
        "latest_test_week.png",
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

    short = generate_synthetic_load(periods=24 * 21)
    with pytest.raises(ValueError, match="requires at least"):
        run_pjme_benchmark(
            short,
            BenchmarkConfig(
                validation_folds=1,
                test_folds=1,
                max_train_hours=24 * 15,
                n_estimators=2,
            ),
        )


def test_benchmark_adds_exogenous_model_when_weather_is_available() -> None:
    data = generate_synthetic_load(periods=24 * 400, start="2016-01-01")
    weather = data[[Col.TIMESTAMP]].assign(**{Col.TEMPERATURE: 10.0})
    config = BenchmarkConfig(
        validation_folds=1,
        test_folds=1,
        max_train_hours=24 * 380,
        n_estimators=2,
    )

    result = run_pjme_benchmark(data, config, weather)

    assert {
        "lightgbm_holidays",
        "lightgbm_weather",
        "lightgbm_exogenous",
    }.issubset(result.leaderboard[Col.MODEL])
