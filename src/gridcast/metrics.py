import numpy as np
from numpy.typing import NDArray


def mean_absolute_error(
    actual: NDArray[np.float64], prediction: NDArray[np.float64]
) -> float:
    """Calculate mean absolute error.

    Parameters
    ----------
    actual : numpy.ndarray
        Observed values.
    prediction : numpy.ndarray
        Forecast values.

    Returns
    -------
    float
        Mean absolute error.
    """
    actual_values, prediction_values = _validated_pair(actual, prediction)
    return float(np.mean(np.abs(actual_values - prediction_values)))


def root_mean_squared_error(
    actual: NDArray[np.float64], prediction: NDArray[np.float64]
) -> float:
    """Calculate root mean squared error.

    Parameters
    ----------
    actual : numpy.ndarray
        Observed values.
    prediction : numpy.ndarray
        Forecast values.

    Returns
    -------
    float
        Root mean squared error.
    """
    actual_values, prediction_values = _validated_pair(actual, prediction)
    return float(np.sqrt(np.mean((actual_values - prediction_values) ** 2)))


def mean_absolute_scaled_error(
    actual: NDArray[np.float64],
    prediction: NDArray[np.float64],
    training: NDArray[np.float64],
    seasonal_period: int,
) -> float:
    """Calculate MASE using an in-sample seasonal-naive scale.

    Parameters
    ----------
    actual : numpy.ndarray
        Observed out-of-sample values.
    prediction : numpy.ndarray
        Forecast values.
    training : numpy.ndarray
        In-sample target used to estimate the scale.
    seasonal_period : int
        Number of observations in one seasonal cycle.

    Returns
    -------
    float
        Mean absolute scaled error.

    Raises
    ------
    ValueError
        If the training history cannot produce a positive seasonal scale.
    """
    if seasonal_period < 1:
        msg = "seasonal_period must be positive"
        raise ValueError(msg)
    training_values = np.asarray(training, dtype=float)
    if training_values.ndim != 1 or len(training_values) <= seasonal_period:
        msg = "training must exceed one seasonal period"
        raise ValueError(msg)
    if not np.isfinite(training_values).all():
        msg = "training must contain only finite values"
        raise ValueError(msg)

    scale = np.mean(
        np.abs(training_values[seasonal_period:] - training_values[:-seasonal_period])
    )
    if scale <= 0.0:
        msg = "seasonal scale must be positive"
        raise ValueError(msg)
    return mean_absolute_error(actual, prediction) / float(scale)


def pinball_loss(
    actual: NDArray[np.float64],
    prediction: NDArray[np.float64],
    quantile: float,
) -> float:
    """Calculate mean pinball loss for a quantile forecast.

    Parameters
    ----------
    actual : numpy.ndarray
        Observed values.
    prediction : numpy.ndarray
        Quantile predictions.
    quantile : float
        Quantile level strictly between zero and one.

    Returns
    -------
    float
        Mean pinball loss.
    """
    if not 0.0 < quantile < 1.0:
        msg = "quantile must be strictly between zero and one"
        raise ValueError(msg)
    actual_values, prediction_values = _validated_pair(actual, prediction)
    residual = actual_values - prediction_values
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def interval_coverage(
    actual: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Calculate the share of observations inside a prediction interval.

    Parameters
    ----------
    actual : numpy.ndarray
        Observed values.
    lower : numpy.ndarray
        Lower interval bounds.
    upper : numpy.ndarray
        Upper interval bounds.

    Returns
    -------
    float
        Empirical interval coverage in the range zero to one.
    """
    actual_values, lower_values = _validated_pair(actual, lower)
    _, upper_values = _validated_pair(actual, upper)
    if np.any(lower_values > upper_values):
        msg = "lower interval bounds cannot exceed upper bounds"
        raise ValueError(msg)
    return float(
        np.mean((actual_values >= lower_values) & (actual_values <= upper_values))
    )


def mean_interval_width(
    lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> float:
    """Calculate mean prediction interval width.

    Parameters
    ----------
    lower : numpy.ndarray
        Lower interval bounds.
    upper : numpy.ndarray
        Upper interval bounds.

    Returns
    -------
    float
        Mean upper-minus-lower width.
    """
    lower_values, upper_values = _validated_pair(lower, upper)
    if np.any(lower_values > upper_values):
        msg = "lower interval bounds cannot exceed upper bounds"
        raise ValueError(msg)
    return float(np.mean(upper_values - lower_values))


def _validated_pair(
    actual: NDArray[np.float64], prediction: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    actual_values = np.asarray(actual, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    if actual_values.ndim != 1 or prediction_values.ndim != 1:
        msg = "actual and prediction must be one-dimensional"
        raise ValueError(msg)
    if len(actual_values) == 0 or len(actual_values) != len(prediction_values):
        msg = "actual and prediction must have equal, non-zero lengths"
        raise ValueError(msg)
    if not np.isfinite(actual_values).all() or not np.isfinite(prediction_values).all():
        msg = "actual and prediction must contain only finite values"
        raise ValueError(msg)
    return actual_values, prediction_values
