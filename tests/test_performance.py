import json
from pathlib import Path

import numpy as np
import pytest

from gridcast.columns import Col
from gridcast.data import generate_synthetic_load
from gridcast.performance import (
    PerformanceConfig,
    load_performance_summary,
    run_performance_benchmark,
    write_performance_artifacts,
)


def test_performance_benchmark_measures_three_models(tmp_path: Path) -> None:
    data = generate_synthetic_load(periods=24 * 400, start="2017-01-01")
    weather = data[[Col.TIMESTAMP]].assign(
        **{Col.TEMPERATURE: np.sin(np.arange(len(data)) / 100.0) * 10.0}
    )
    config = PerformanceConfig(
        horizon=24 * 7,
        max_train_hours=24 * 380,
        n_estimators=3,
        warmup_runs=1,
        repetitions=3,
    )

    result = run_performance_benchmark(data, weather, config)

    assert set(result.measurements[Col.MODEL]) == {
        "seasonal_naive_168h",
        "lightgbm",
        "lightgbm_exogenous",
    }
    assert result.measurements["prediction_median_ms"].gt(0.0).all()
    assert (
        result.measurements["prediction_p95_ms"]
        .ge(result.measurements["prediction_median_ms"])
        .all()
    )
    assert result.measurements["serialized_model_kib"].gt(0.0).all()
    assert result.environment["logical_cpu_count"]

    write_performance_artifacts(result, config, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "latency.png",
        "measurements.csv",
        "summary.json",
    }
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["config"]["repetitions"] == 3
    assert load_performance_summary(tmp_path / "summary.json") == report


def test_performance_config_and_history_are_validated() -> None:
    with pytest.raises(ValueError, match="positive"):
        PerformanceConfig(repetitions=0)
    with pytest.raises(ValueError, match="negative"):
        PerformanceConfig(warmup_runs=-1)
    with pytest.raises(ValueError, match="cannot exceed"):
        PerformanceConfig(horizon=169)

    data = generate_synthetic_load(periods=24 * 14)
    weather = data[[Col.TIMESTAMP]].assign(**{Col.TEMPERATURE: 10.0})
    with pytest.raises(ValueError, match="complete train"):
        run_performance_benchmark(
            data,
            weather,
            PerformanceConfig(max_train_hours=24 * 10, repetitions=1),
        )
