import numpy as np
import pytest

from gridcast.baselines import SeasonalNaiveForecaster


def test_forecaster_repeats_last_season_for_long_horizon() -> None:
    history = np.arange(10, dtype=float)

    prediction = SeasonalNaiveForecaster(4).fit(history).predict(7)

    np.testing.assert_array_equal(prediction, [6, 7, 8, 9, 6, 7, 8])


def test_forecaster_retains_only_the_required_season() -> None:
    model = SeasonalNaiveForecaster(4).fit(np.arange(1_000, dtype=float))

    assert model._history is not None
    assert len(model._history) == 4


def test_forecaster_validates_lifecycle_and_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        SeasonalNaiveForecaster(0)

    model = SeasonalNaiveForecaster(3)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(1)
    with pytest.raises(ValueError, match="at least"):
        model.fit(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        model.fit(np.ones((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([1.0, 2.0, np.nan]))

    model.fit(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="positive"):
        model.predict(0)
