import numpy as np
import pandas as pd
import pytest

from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, Col
from gridcast.decision import (
    asymmetric_cost,
    evaluate_decision_costs,
    evaluate_quantile_decisions,
    interpolated_quantile_schedule,
    optimal_scheduling_quantile,
)


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


def test_decision_evaluation_ranks_models_by_scenario_cost() -> None:
    forecasts = pd.DataFrame(
        {
            Col.SPLIT: [HISTORICAL_HOLDOUT_SPLIT] * 4,
            Col.MODEL: ["under", "under", "over", "over"],
            Col.FOLD: [1, 1, 1, 1],
            Col.TARGET: [100.0, 100.0, 100.0, 100.0],
            Col.PREDICTION: [90.0, 90.0, 110.0, 110.0],
        }
    )

    result = evaluate_decision_costs(forecasts)

    shortage = result.loc[result["scenario"].eq("shortage_heavy")]
    surplus = result.loc[result["scenario"].eq("surplus_heavy")]
    assert shortage.iloc[0][Col.MODEL] == "over"
    assert surplus.iloc[0][Col.MODEL] == "under"
    assert set(result["optimal_quantile"]) == {0.25, 0.5, 0.75}
    assert result["regret_vs_perfect"].eq(result["mean_cost"]).all()


def test_decision_evaluation_requires_schema_and_holdout() -> None:
    with pytest.raises(ValueError, match="missing required"):
        evaluate_decision_costs(pd.DataFrame({Col.TARGET: [1.0]}))
    empty = pd.DataFrame(
        {
            Col.SPLIT: ["validation"],
            Col.MODEL: ["model"],
            Col.FOLD: [1],
            Col.TARGET: [1.0],
            Col.PREDICTION: [1.0],
        }
    )
    with pytest.raises(ValueError, match="historical holdout"):
        evaluate_decision_costs(empty)


def test_cost_optimal_quantile_schedule_reduces_asymmetric_cost() -> None:
    forecasts = pd.DataFrame(
        {
            Col.SPLIT: [HISTORICAL_HOLDOUT_SPLIT] * 2,
            Col.TARGET: [120.0, 120.0],
            Col.P10: [80.0, 80.0],
            Col.P50: [100.0, 100.0],
            Col.P90: [140.0, 140.0],
        }
    )

    result = evaluate_quantile_decisions(forecasts)
    shortage = result.loc[result["scenario"].eq("shortage_heavy")].iloc[0]

    assert shortage["selected_quantile"] == pytest.approx(0.75)
    assert shortage["cost_savings_pct"] > 0.0
    np.testing.assert_array_equal(
        interpolated_quantile_schedule(
            np.array([80.0]),
            np.array([100.0]),
            np.array([140.0]),
            0.75,
        ),
        [125.0],
    )


def test_quantile_decisions_validate_inputs() -> None:
    values = np.array([1.0])
    with pytest.raises(ValueError, match="between"):
        interpolated_quantile_schedule(values, values, values, 0.95)
    with pytest.raises(ValueError, match="ordered"):
        interpolated_quantile_schedule(np.array([2.0]), values, np.array([3.0]), 0.5)
    with pytest.raises(ValueError, match="missing required"):
        evaluate_quantile_decisions(pd.DataFrame({Col.TARGET: [1.0]}))
