import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from gridcast.models import (
    POINT_ESTIMATORS,
    CatBoostLoadForecaster,
    HistGradientBoostingLoadForecaster,
    LightGBMLoadForecaster,
    LightGBMQuantileForecaster,
    MissingOptionalDependencyError,
    XGBoostLoadForecaster,
    create_point_forecaster,
    resolved_point_model_parameters,
)


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


class FakeRegressor:
    """Minimal optional-backend test double."""

    parameters: dict[str, object] = {}

    def __init__(self, **parameters: object) -> None:
        FakeRegressor.parameters = parameters

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "FakeRegressor":
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 123.0)


@pytest.mark.parametrize(
    ("estimator", "module_name", "class_name", "expected_parameter"),
    [
        (CatBoostLoadForecaster, "catboost", "CatBoostRegressor", "loss_function"),
        (XGBoostLoadForecaster, "xgboost", "XGBRegressor", "objective"),
    ],
)
def test_optional_boosters_lazy_load_and_forward_parameters(
    monkeypatch: pytest.MonkeyPatch,
    estimator: type[CatBoostLoadForecaster] | type[XGBoostLoadForecaster],
    module_name: str,
    class_name: str,
    expected_parameter: str,
) -> None:
    module = ModuleType(module_name)
    setattr(module, class_name, FakeRegressor)
    monkeypatch.setitem(sys.modules, module_name, module)
    features = pd.DataFrame({"x": range(30), "lag": range(30, 60)})
    target = pd.Series(range(100, 130), dtype=float)

    prediction = (
        estimator(n_estimators=5).fit(features, target).predict(features.iloc[:3])
    )

    assert np.array_equal(prediction, np.full(3, 123.0))
    assert expected_parameter in FakeRegressor.parameters


def test_hist_gradient_boosting_and_registry() -> None:
    features = pd.DataFrame({"x": range(50), "lag": range(50, 100)})
    target = pd.Series(range(100, 150), dtype=float)

    prediction = (
        HistGradientBoostingLoadForecaster(n_estimators=5)
        .fit(features, target)
        .predict(features.iloc[:3])
    )

    assert prediction.shape == (3,)
    assert set(POINT_ESTIMATORS) == {
        "lightgbm",
        "hist_gradient_boosting",
        "catboost",
        "xgboost",
    }
    assert isinstance(
        create_point_forecaster("lightgbm", n_estimators=2), LightGBMLoadForecaster
    )
    with pytest.raises(ValueError, match="unknown point estimator"):
        create_point_forecaster("unknown")
    parameters = resolved_point_model_parameters(300)
    assert parameters["hist_gradient_boosting"]["early_stopping"] is False
    assert parameters["catboost"]["depth"] == 8
    assert parameters["xgboost"]["tree_method"] == "hist"


def test_point_forecasters_validate_shared_contract() -> None:
    model = HistGradientBoostingLoadForecaster(n_estimators=2)
    with pytest.raises(ValueError, match="indexes must match"):
        model.fit(
            pd.DataFrame({"x": [1.0]}, index=pd.Index([0])),
            pd.Series([1.0], index=pd.Index([1])),
        )
    with pytest.raises(ValueError, match="finite"):
        model.fit(pd.DataFrame({"x": [np.inf]}), pd.Series([1.0]))
    model.fit(pd.DataFrame({"x": [1.0, 2.0]}), pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        model.predict(pd.DataFrame({"x": [np.inf]}))


def test_missing_optional_dependency_has_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gridcast.models.import_module",
        lambda _: (_ for _ in ()).throw(ImportError("missing")),
    )

    with pytest.raises(MissingOptionalDependencyError, match="--extra extended"):
        CatBoostLoadForecaster(n_estimators=2).fit(
            pd.DataFrame({"x": [1.0, 2.0]}),
            pd.Series([1.0, 2.0]),
        )
