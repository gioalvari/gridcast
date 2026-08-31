import pandas as pd
import pytest

from gridcast.models import LightGBMLoadForecaster, LightGBMQuantileForecaster


def test_lightgbm_forecaster_fits_and_predicts() -> None:
    features = pd.DataFrame({"x": range(30), "lag": range(30, 60)})
    target = pd.Series(range(100, 130), dtype=float)

    prediction = (
        LightGBMLoadForecaster(n_estimators=5)
        .fit(features, target)
        .predict(features.iloc[:3])
    )

    assert prediction.shape == (3,)


def test_lightgbm_forecaster_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        LightGBMLoadForecaster(n_estimators=0)
    with pytest.raises(ValueError, match="learning_rate"):
        LightGBMLoadForecaster(learning_rate=0.0)

    model = LightGBMLoadForecaster(n_estimators=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError, match="complete feature"):
        model.fit(pd.DataFrame({"x": [float("nan")]}), pd.Series([1.0]))

    model.fit(pd.DataFrame({"x": [1.0, 2.0]}), pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="complete"):
        model.predict(pd.DataFrame({"x": [float("nan")]}))


def test_quantile_forecaster_fits_and_validates_quantile() -> None:
    features = pd.DataFrame({"x": range(30), "lag": range(30, 60)})
    target = pd.Series(range(100, 130), dtype=float)

    prediction = (
        LightGBMQuantileForecaster(quantile=0.9, n_estimators=5)
        .fit(features, target)
        .predict(features.iloc[:3])
    )

    assert prediction.shape == (3,)
    with pytest.raises(ValueError, match="quantile"):
        LightGBMQuantileForecaster(quantile=0.0)
    with pytest.raises(ValueError, match="n_estimators"):
        LightGBMQuantileForecaster(quantile=0.5, n_estimators=0)
    with pytest.raises(ValueError, match="learning_rate"):
        LightGBMQuantileForecaster(quantile=0.5, learning_rate=0.0)


def test_quantile_forecaster_validates_lifecycle_and_features() -> None:
    model = LightGBMQuantileForecaster(quantile=0.5, n_estimators=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError, match="complete feature"):
        model.fit(pd.DataFrame({"x": [float("nan")]}), pd.Series([1.0]))

    model.fit(pd.DataFrame({"x": [1.0, 2.0]}), pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="complete"):
        model.predict(pd.DataFrame({"x": [float("nan")]}))
