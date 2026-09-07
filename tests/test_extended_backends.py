import importlib.util

import numpy as np
import pandas as pd
import pytest

from gridcast.models import create_point_forecaster


@pytest.mark.parametrize("estimator", ["catboost", "xgboost"])
def test_real_optional_backend_fits_and_predicts(estimator: str) -> None:
    """Smoke-test native optional packages when the extended extra is installed."""
    if importlib.util.find_spec(estimator) is None:
        pytest.skip("extended model extra is not installed")
    features = pd.DataFrame(
        {
            "hour": np.tile(np.arange(24), 5),
            "lag": np.linspace(10_000.0, 20_000.0, 120),
        }
    )
    target = pd.Series(20_000.0 + features["hour"] * 100.0, dtype=float)

    prediction = (
        create_point_forecaster(estimator, n_estimators=5)
        .fit(features, target)
        .predict(features.iloc[:24])
    )

    assert prediction.shape == (24,)
    assert np.isfinite(prediction).all()
