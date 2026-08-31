import numpy as np
from numpy.typing import NDArray


def asymmetric_cost(
    actual: NDArray[np.float64],
    scheduled: NDArray[np.float64],
    shortage_cost: float,
    surplus_cost: float,
) -> float:
    """Calculate mean asymmetric scheduling cost in synthetic cost units.

    Parameters
    ----------
    actual : numpy.ndarray
        Realized load.
    scheduled : numpy.ndarray
        Scheduled or procured load.
    shortage_cost : float
        Unit penalty when actual load exceeds the schedule.
    surplus_cost : float
        Unit penalty when the schedule exceeds actual load.

    Returns
    -------
    float
        Mean asymmetric cost.
    """
    if shortage_cost < 0.0 or surplus_cost < 0.0:
        msg = "costs must be non-negative"
        raise ValueError(msg)
    actual_values = np.asarray(actual, dtype=float)
    scheduled_values = np.asarray(scheduled, dtype=float)
    if actual_values.shape != scheduled_values.shape or actual_values.size == 0:
        msg = "actual and scheduled must have equal, non-empty shapes"
        raise ValueError(msg)
    if not np.isfinite(actual_values).all() or not np.isfinite(scheduled_values).all():
        msg = "actual and scheduled must contain finite values"
        raise ValueError(msg)
    shortage = np.maximum(actual_values - scheduled_values, 0.0)
    surplus = np.maximum(scheduled_values - actual_values, 0.0)
    return float(np.mean(shortage_cost * shortage + surplus_cost * surplus))


def optimal_scheduling_quantile(shortage_cost: float, surplus_cost: float) -> float:
    """Return the forecast quantile optimal for asymmetric linear cost.

    Parameters
    ----------
    shortage_cost : float
        Unit shortage penalty.
    surplus_cost : float
        Unit surplus penalty.

    Returns
    -------
    float
        Cost-optimal quantile in the closed interval zero to one.
    """
    total = shortage_cost + surplus_cost
    if shortage_cost < 0.0 or surplus_cost < 0.0 or total <= 0.0:
        msg = "at least one non-negative cost must be positive"
        raise ValueError(msg)
    return shortage_cost / total
