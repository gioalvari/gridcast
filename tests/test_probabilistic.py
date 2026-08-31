from pathlib import Path

import numpy as np
import pytest

from gridcast.columns import Col
from gridcast.data import generate_synthetic_load
from gridcast.probabilistic import (
    ProbabilisticConfig,
    conformal_correction,
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

    write_probabilistic_artifacts(result, config, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "coverage.png",
        "forecasts.parquet",
        "latest_test_interval.png",
        "metrics.csv",
        "summary.json",
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

    data = generate_synthetic_load(periods=24 * 30)
    weather = data[[Col.TIMESTAMP]].assign(**{Col.TEMPERATURE: 10.0})
    with pytest.raises(ValueError, match="requires at least"):
        run_probabilistic_benchmark(data, weather, ProbabilisticConfig())
