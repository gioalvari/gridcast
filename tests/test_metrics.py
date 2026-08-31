import numpy as np
import pytest

from gridcast.metrics import (
    interval_coverage,
    mean_absolute_error,
    mean_absolute_scaled_error,
    mean_interval_width,
    pinball_loss,
    root_mean_squared_error,
)


def test_point_metrics_return_expected_values() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.0, 4.0, 2.0])

    assert mean_absolute_error(actual, prediction) == pytest.approx(1.0)
    assert root_mean_squared_error(actual, prediction) == pytest.approx(
        np.sqrt(5.0 / 3.0)
    )


def test_mase_uses_seasonal_training_scale() -> None:
    training = np.array([1.0, 2.0, 3.0, 3.0, 4.0, 5.0])

    result = mean_absolute_scaled_error(
        actual=np.array([6.0, 7.0]),
        prediction=np.array([5.0, 5.0]),
        training=training,
        seasonal_period=3,
    )

    assert result == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("actual", "prediction"),
    [
        (np.array([]), np.array([])),
        (np.array([1.0]), np.array([1.0, 2.0])),
        (np.ones((1, 1)), np.ones((1, 1))),
        (np.array([np.nan]), np.array([1.0])),
    ],
)
def test_point_metrics_reject_invalid_arrays(
    actual: np.ndarray, prediction: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        mean_absolute_error(actual, prediction)


def test_mase_rejects_invalid_scale_inputs() -> None:
    actual = np.array([1.0])
    prediction = np.array([1.0])

    with pytest.raises(ValueError, match="positive"):
        mean_absolute_scaled_error(actual, prediction, np.arange(4.0), 0)
    with pytest.raises(ValueError, match="exceed"):
        mean_absolute_scaled_error(actual, prediction, np.arange(3.0), 3)
    with pytest.raises(ValueError, match="finite"):
        mean_absolute_scaled_error(actual, prediction, np.array([1.0, np.nan]), 1)
    with pytest.raises(ValueError, match="scale"):
        mean_absolute_scaled_error(actual, prediction, np.ones(3), 1)


def test_probabilistic_metrics_return_expected_values() -> None:
    actual = np.array([1.0, 2.0, 5.0])
    median = np.array([2.0, 2.0, 3.0])
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([2.0, 3.0, 4.0])

    assert pinball_loss(actual, median, 0.5) == pytest.approx(0.5)
    assert interval_coverage(actual, lower, upper) == pytest.approx(2.0 / 3.0)
    assert mean_interval_width(lower, upper) == pytest.approx(2.0)


def test_probabilistic_metrics_validate_parameters_and_bounds() -> None:
    values = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss(values, values, 1.0)
    with pytest.raises(ValueError, match="lower"):
        interval_coverage(values, values + 1.0, values)
    with pytest.raises(ValueError, match="lower"):
        mean_interval_width(values + 1.0, values)
