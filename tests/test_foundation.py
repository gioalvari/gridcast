import json
from pathlib import Path

import numpy as np
import pytest

from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, Col
from gridcast.data import generate_synthetic_load
from gridcast.foundation import (
    FoundationConfig,
    FoundationForecast,
    build_holdout_contexts,
    run_foundation_benchmark,
    write_foundation_artifacts,
)


class FakeFoundationForecaster:
    """Deterministic fake returning the final context value."""

    def forecast(
        self,
        inputs: list[np.ndarray],
        horizon: int,
    ) -> FoundationForecast:
        point = np.asarray(
            [np.repeat(context[-1], horizon) for context in inputs], dtype=float
        )
        quantiles = np.empty((*point.shape, 10), dtype=float)
        quantiles[:, :, 0] = point
        for index in range(1, 10):
            quantiles[:, :, index] = point + (index - 5) * 10.0
        return FoundationForecast(
            point=point,
            p10=quantiles[:, :, 1],
            p50=quantiles[:, :, 5],
            p90=quantiles[:, :, 9],
        )


def _config() -> FoundationConfig:
    return FoundationConfig(
        model_name="example_model",
        model_id="example/model",
        model_revision="abc123",
        context_length=24 * 7,
        horizon=24,
        holdout_folds=2,
    )


def test_holdout_contexts_stop_before_each_forecast_origin() -> None:
    data = generate_synthetic_load(periods=24 * 30)

    contexts, actuals, origins = build_holdout_contexts(data, _config())

    assert len(contexts) == len(actuals) == len(origins) == 2
    assert contexts[0][-1] == pytest.approx(data.loc[origins[0] - 1, Col.TARGET])
    assert actuals[0][0] == pytest.approx(data.loc[origins[0], Col.TARGET])


def test_foundation_benchmark_generates_point_and_quantile_metrics(
    tmp_path: Path,
) -> None:
    data = generate_synthetic_load(periods=24 * 30)
    config = _config()

    result = run_foundation_benchmark(data, FakeFoundationForecaster(), config)

    assert len(result.forecasts) == config.horizon * config.holdout_folds
    assert result.forecasts[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT).all()
    assert result.forecasts[Col.P10].le(result.forecasts[Col.P50]).all()
    assert result.forecasts[Col.P50].le(result.forecasts[Col.P90]).all()
    assert result.metrics["folds"] == 2
    assert float(result.metrics["mae"]) > 0.0
    assert result.first_call_seconds >= 0.0
    assert result.warm_call_seconds >= 0.0

    write_foundation_artifacts(
        result,
        config,
        data,
        tmp_path,
        environment={"device": "test"},
    )
    assert {path.name for path in tmp_path.iterdir()} == {
        "experiment_manifest.json",
        "forecasts.parquet",
        "summary.json",
    }
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["config"]["model_revision"] == "abc123"
    assert report["timing"]["warm_call_seconds"] >= 0.0


def test_foundation_config_and_history_are_validated() -> None:
    with pytest.raises(ValueError, match="required"):
        FoundationConfig(model_name="model", model_id="", model_revision="revision")
    with pytest.raises(ValueError, match="positive"):
        FoundationConfig(
            model_name="model",
            model_id="model",
            model_revision="revision",
            holdout_folds=0,
        )
    data = generate_synthetic_load(periods=24 * 14)
    with pytest.raises(ValueError, match="requires at least"):
        build_holdout_contexts(
            data,
            FoundationConfig(
                model_name="model",
                model_id="model",
                model_revision="revision",
                context_length=24 * 14,
                horizon=24,
                holdout_folds=2,
            ),
        )


@pytest.mark.parametrize("invalid", ["point", "quantiles", "finite"])
def test_foundation_benchmark_rejects_invalid_model_output(invalid: str) -> None:
    data = generate_synthetic_load(periods=24 * 30)
    config = _config()

    class InvalidForecaster:
        def forecast(
            self,
            inputs: list[np.ndarray],
            horizon: int,
        ) -> FoundationForecast:
            point = np.ones((2, horizon), dtype=float)
            p10 = np.ones((2, horizon), dtype=float)
            if invalid == "point":
                point = point[:1]
            elif invalid == "quantiles":
                p10 = p10[:1]
            else:
                point[0, 0] = np.nan
            return FoundationForecast(
                point=point,
                p10=p10,
                p50=np.ones((2, horizon), dtype=float),
                p90=np.ones((2, horizon), dtype=float),
            )

    with pytest.raises(ValueError):
        run_foundation_benchmark(data, InvalidForecaster(), config)
