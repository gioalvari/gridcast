from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.columns import Col
from gridcast.data import generate_synthetic_load
from gridcast.probabilistic import (
    ProbabilisticConfig,
    conformal_correction,
    hourly_conformal_corrections,
    rolling_conformal_corrections,
    run_probabilistic_benchmark,
    write_probabilistic_artifacts,
)


def test_conformal_correction_expands_undercovered_intervals() -> None:
    actual = np.array([0.0, 1.0, 2.0, 10.0])
    lower = np.array([0.0, 0.0, 1.0, 2.0])
    upper = np.array([1.0, 2.0, 3.0, 4.0])

    correction = conformal_correction(actual, lower, upper, miscoverage=0.25)

    assert correction == pytest.approx(6.0)


def test_conformal_correction_validates_inputs() -> None:
    values = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="miscoverage"):
        conformal_correction(values, values, values, miscoverage=0.0)
    with pytest.raises(ValueError, match="equal"):
        conformal_correction(values, values[:1], values)
    with pytest.raises(ValueError, match="finite"):
        conformal_correction(np.array([np.nan]), np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="lower"):
        conformal_correction(values, values + 1.0, values)


def test_hourly_corrections_require_and_calibrate_every_hour() -> None:
    timestamps = pd.date_range("2024-01-01", periods=48, freq="h")
    validation = pd.DataFrame(
        {
            Col.TIMESTAMP: timestamps,
            Col.TARGET: np.arange(48, dtype=float),
            Col.P10: np.arange(48, dtype=float) - 1.0,
            Col.P90: np.arange(48, dtype=float) + 1.0,
        }
    )

    corrections = hourly_conformal_corrections(validation)

    assert set(corrections) == set(range(24))
    assert all(correction == 0.0 for correction in corrections.values())
    with pytest.raises(ValueError, match="hour 23"):
        hourly_conformal_corrections(
            validation.loc[validation[Col.TIMESTAMP].dt.hour.ne(23)]
        )


def test_rolling_corrections_are_causal() -> None:
    timestamps = pd.date_range("2024-01-01", periods=24, freq="h")
    base = pd.DataFrame(
        {
            Col.TIMESTAMP: timestamps,
            Col.TARGET: np.arange(24, dtype=float),
            Col.P10: np.arange(24, dtype=float) - 1.0,
            Col.P90: np.arange(24, dtype=float) + 1.0,
            Col.FOLD: np.repeat([1, 2, 3, 4], 6),
        }
    )
    validation = base.iloc[:12].copy()
    validation[Col.FOLD] = np.repeat([1, 2], 6)
    test = base.copy()

    original = rolling_conformal_corrections(validation, test, window_folds=2)
    changed = test.copy()
    changed.loc[changed[Col.FOLD].eq(3), Col.TARGET] += 1_000.0
    modified = rolling_conformal_corrections(validation, changed, window_folds=2)

    assert original[1] == modified[1]
    assert original[2] == modified[2]
    assert original[3] == modified[3]
    assert original[4] != modified[4]
    with pytest.raises(ValueError, match="positive"):
        rolling_conformal_corrections(validation, test, window_folds=0)
    with pytest.raises(ValueError, match="validation"):
        rolling_conformal_corrections(validation.iloc[:0], test, window_folds=2)


def test_probabilistic_benchmark_separates_calibration_and_test(
    tmp_path: Path,
) -> None:
    data = generate_synthetic_load(periods=24 * 400, start="2016-01-01")
    weather = data[[Col.TIMESTAMP]].assign(
        **{Col.TEMPERATURE: np.sin(np.arange(len(data)) / 100.0) * 10.0}
    )
    config = ProbabilisticConfig(
        validation_folds=1,
        test_folds=1,
        max_train_hours=24 * 380,
        n_estimators=3,
    )

    result = run_probabilistic_benchmark(data, weather, config)

    assert set(result.forecasts[Col.SPLIT]) == {"validation", "test"}
    assert (result.forecasts[Col.P10] <= result.forecasts[Col.P50]).all()
    assert (result.forecasts[Col.P50] <= result.forecasts[Col.P90]).all()
    assert (result.forecasts[Col.P10_CALIBRATED] <= result.forecasts[Col.P10]).all()
    assert (result.forecasts[Col.P90_CALIBRATED] >= result.forecasts[Col.P90]).all()
    assert result.conformal_correction_mw >= 0.0
    assert set(result.hourly_corrections_mw) == set(range(24))
    assert set(result.rolling_corrections_mw) == {1}
    assert (
        result.forecasts[Col.P10_HOURLY_CALIBRATED] <= result.forecasts[Col.P10]
    ).all()

    write_probabilistic_artifacts(result, config, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "coverage.png",
        "forecasts.parquet",
        "hourly_coverage.csv",
        "hourly_coverage.png",
        "latest_test_interval.png",
        "metrics.csv",
        "rolling_corrections.csv",
        "summary.json",
        "weekly_coverage.csv",
    }


def test_probabilistic_config_and_history_are_validated() -> None:
    with pytest.raises(ValueError, match="positive"):
        ProbabilisticConfig(test_folds=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        ProbabilisticConfig(horizon=169)
    with pytest.raises(ValueError, match="warmup"):
        ProbabilisticConfig(max_train_hours=100)
    with pytest.raises(ValueError, match="n_estimators"):
        ProbabilisticConfig(n_estimators=0)
    with pytest.raises(ValueError, match="rolling_window_folds"):
        ProbabilisticConfig(rolling_window_folds=0)

    data = generate_synthetic_load(periods=24 * 30)
    weather = data[[Col.TIMESTAMP]].assign(**{Col.TEMPERATURE: 10.0})
    with pytest.raises(ValueError, match="requires at least"):
        run_probabilistic_benchmark(data, weather, ProbabilisticConfig())
