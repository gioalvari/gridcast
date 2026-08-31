import pandas as pd
import pytest

from gridcast.backtesting import BacktestConfig, rolling_backtest
from gridcast.columns import Col
from gridcast.data import generate_synthetic_load


def test_rolling_backtest_produces_strictly_out_of_sample_folds() -> None:
    data = generate_synthetic_load(periods=24 * 35)
    config = BacktestConfig(
        initial_window=24 * 21,
        horizon=24 * 7,
        step=24 * 7,
        seasonal_period=24 * 7,
    )

    result = rolling_backtest(data, config)

    assert len(result.metrics) == 2
    assert len(result.forecasts) == 24 * 14
    assert result.forecasts.groupby(Col.FOLD).size().eq(config.horizon).all()
    assert (result.forecasts[Col.TIMESTAMP] > result.forecasts[Col.CUTOFF]).all()
    assert result.summary["folds"] == 2
    assert result.summary["observations"] == 24 * 14
    assert result.summary["mae"] > 0
    assert result.summary["rmse"] >= result.summary["mae"]
    assert result.summary["mase"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_window": 0, "horizon": 1, "step": 1, "seasonal_period": 1},
        {"initial_window": 2, "horizon": 1, "step": 1, "seasonal_period": 0},
        {"initial_window": 2, "horizon": 1, "step": 1, "seasonal_period": 2},
    ],
)
def test_backtest_config_rejects_invalid_windows(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


def test_rolling_backtest_rejects_invalid_data() -> None:
    config = BacktestConfig(10, 2, 2, 3)
    valid = pd.DataFrame(
        {
            Col.TIMESTAMP: pd.date_range("2024-01-01", periods=12, freq="h"),
            Col.TARGET: range(12),
        }
    )

    with pytest.raises(ValueError, match="missing"):
        rolling_backtest(valid.drop(columns=Col.TARGET), config)
    with pytest.raises(ValueError, match="datetimes"):
        rolling_backtest(valid.assign(**{Col.TIMESTAMP: range(12)}), config)
    with pytest.raises(ValueError, match="unique"):
        rolling_backtest(
            valid.assign(**{Col.TIMESTAMP: [valid[Col.TIMESTAMP].iloc[0]] * 12}),
            config,
        )
    with pytest.raises(ValueError, match="sorted"):
        rolling_backtest(valid.iloc[::-1].reset_index(drop=True), config)
    with pytest.raises(ValueError, match="finite"):
        rolling_backtest(
            valid.assign(**{Col.TARGET: [*range(11), float("nan")]}), config
        )
    with pytest.raises(ValueError, match="initial window"):
        rolling_backtest(valid.iloc[:-1], config)
