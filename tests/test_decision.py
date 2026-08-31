import numpy as np
import pytest

from gridcast.decision import asymmetric_cost, optimal_scheduling_quantile


def test_asymmetric_cost_penalizes_shortage_and_surplus() -> None:
    actual = np.array([100.0, 100.0])
    scheduled = np.array([90.0, 120.0])

    cost = asymmetric_cost(actual, scheduled, shortage_cost=3.0, surplus_cost=1.0)

    assert cost == pytest.approx(25.0)


def test_cost_optimal_quantile_matches_penalty_ratio() -> None:
    assert optimal_scheduling_quantile(3.0, 1.0) == pytest.approx(0.75)
    assert optimal_scheduling_quantile(1.0, 3.0) == pytest.approx(0.25)


def test_decision_metrics_validate_inputs() -> None:
    values = np.array([1.0])
    with pytest.raises(ValueError, match="non-negative"):
        asymmetric_cost(values, values, -1.0, 1.0)
    with pytest.raises(ValueError, match="equal"):
        asymmetric_cost(values, np.array([1.0, 2.0]), 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        asymmetric_cost(np.array([np.nan]), values, 1.0, 1.0)
    with pytest.raises(ValueError, match="at least one"):
        optimal_scheduling_quantile(0.0, 0.0)
