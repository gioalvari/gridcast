import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, Col

DECISION_SCENARIOS: dict[str, tuple[float, float]] = {
    "symmetric": (1.0, 1.0),
    "shortage_heavy": (3.0, 1.0),
    "surplus_heavy": (1.0, 3.0),
}


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


def evaluate_decision_costs(
    forecasts: pd.DataFrame,
    scenarios: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Evaluate point schedules under asymmetric synthetic cost scenarios.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Timestamped point predictions with split, fold, model, and actual load.
    scenarios : dict, optional
        Named ``(shortage_cost, surplus_cost)`` pairs.

    Returns
    -------
    pandas.DataFrame
        Aggregate cost, regret, and cost-optimal quantile by scenario and model.
    """
    required = {
        Col.SPLIT,
        Col.MODEL,
        Col.TARGET,
        Col.PREDICTION,
        Col.FOLD,
    }
    missing = required.difference(forecasts.columns)
    if missing:
        names = ", ".join(sorted(missing))
        msg = f"decision forecasts are missing required columns: {names}"
        raise ValueError(msg)
    selected = forecasts.loc[forecasts[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)]
    if selected.empty:
        msg = "decision evaluation requires historical holdout forecasts"
        raise ValueError(msg)
    rows: list[dict[str, float | int | str]] = []
    for scenario, (shortage_cost, surplus_cost) in (
        scenarios or DECISION_SCENARIOS
    ).items():
        for model, group in selected.groupby(Col.MODEL, sort=True):
            actual = group[Col.TARGET].to_numpy(dtype=np.float64)
            schedule = group[Col.PREDICTION].to_numpy(dtype=np.float64)
            mean_cost = asymmetric_cost(actual, schedule, shortage_cost, surplus_cost)
            rows.append(
                {
                    "scenario": scenario,
                    Col.MODEL: str(model),
                    "shortage_cost": shortage_cost,
                    "surplus_cost": surplus_cost,
                    "optimal_quantile": optimal_scheduling_quantile(
                        shortage_cost, surplus_cost
                    ),
                    "folds": group[Col.FOLD].nunique(),
                    "observations": len(group),
                    "mean_cost": mean_cost,
                    "regret_vs_perfect": mean_cost,
                }
            )
    result = pd.DataFrame(rows)
    best_cost = result.groupby("scenario")["mean_cost"].transform("min")
    result["cost_increase_vs_best_pct"] = (
        100.0 * (result["mean_cost"] - best_cost) / best_cost
    )
    return result.sort_values(["scenario", "mean_cost"], ignore_index=True)


def interpolated_quantile_schedule(
    p10: NDArray[np.float64],
    p50: NDArray[np.float64],
    p90: NDArray[np.float64],
    quantile: float,
) -> NDArray[np.float64]:
    """Interpolate a schedule from P10, P50, and P90 predictions.

    Parameters
    ----------
    p10, p50, p90 : numpy.ndarray
        Ordered forecast quantiles.
    quantile : float
        Requested quantile between 0.1 and 0.9.

    Returns
    -------
    numpy.ndarray
        Linearly interpolated schedule.
    """
    if not 0.1 <= quantile <= 0.9:
        msg = "interpolated quantile must be between 0.1 and 0.9"
        raise ValueError(msg)
    lower = np.asarray(p10, dtype=float)
    median = np.asarray(p50, dtype=float)
    upper = np.asarray(p90, dtype=float)
    if not (lower.shape == median.shape == upper.shape) or lower.size == 0:
        msg = "quantile forecasts must have equal, non-empty shapes"
        raise ValueError(msg)
    if np.any(lower > median) or np.any(median > upper):
        msg = "quantile forecasts must be ordered"
        raise ValueError(msg)
    if quantile <= 0.5:
        weight = (quantile - 0.1) / 0.4
        return np.asarray(lower + weight * (median - lower), dtype=np.float64)
    weight = (quantile - 0.5) / 0.4
    return np.asarray(median + weight * (upper - median), dtype=np.float64)


def evaluate_quantile_decisions(
    forecasts: pd.DataFrame,
    scenarios: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Compare median and cost-optimal quantile schedules on the holdout.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Historical holdout P10, P50, P90, and actual observations.
    scenarios : dict, optional
        Named ``(shortage_cost, surplus_cost)`` pairs.

    Returns
    -------
    pandas.DataFrame
        Scenario costs and savings from cost-aware quantile selection.
    """
    required = {Col.SPLIT, Col.TARGET, Col.P10, Col.P50, Col.P90}
    missing = required.difference(forecasts.columns)
    if missing:
        names = ", ".join(sorted(missing))
        msg = f"quantile decisions are missing required columns: {names}"
        raise ValueError(msg)
    selected = forecasts.loc[forecasts[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)]
    if selected.empty:
        msg = "quantile decision evaluation requires historical holdout forecasts"
        raise ValueError(msg)
    actual = selected[Col.TARGET].to_numpy(dtype=np.float64)
    p10 = selected[Col.P10].to_numpy(dtype=np.float64)
    p50 = selected[Col.P50].to_numpy(dtype=np.float64)
    p90 = selected[Col.P90].to_numpy(dtype=np.float64)
    rows: list[dict[str, float | str]] = []
    for scenario, (shortage_cost, surplus_cost) in (
        scenarios or DECISION_SCENARIOS
    ).items():
        quantile = optimal_scheduling_quantile(shortage_cost, surplus_cost)
        cost_aware = interpolated_quantile_schedule(p10, p50, p90, quantile)
        median_cost = asymmetric_cost(actual, p50, shortage_cost, surplus_cost)
        optimal_cost = asymmetric_cost(actual, cost_aware, shortage_cost, surplus_cost)
        rows.append(
            {
                "scenario": scenario,
                "shortage_cost": shortage_cost,
                "surplus_cost": surplus_cost,
                "selected_quantile": quantile,
                "p50_cost": median_cost,
                "cost_aware_quantile_cost": optimal_cost,
                "cost_savings_pct": 100.0 * (median_cost - optimal_cost) / median_cost,
            }
        )
    return pd.DataFrame(rows)
