import json
from pathlib import Path

import pandas as pd
import pytest

from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, Col
from gridcast.dashboard_data import (
    DashboardPaths,
    MissingArtifactsError,
    benchmark_week,
    daily_history,
    display_model,
    load_dashboard_data,
    probabilistic_week,
)


def _write_dashboard_artifacts(root: Path) -> DashboardPaths:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h")
    history = pd.DataFrame({Col.TIMESTAMP: timestamps, Col.TARGET: [10, 20, 30, 40]})
    leaderboard = pd.DataFrame(
        {
            Col.SPLIT: [HISTORICAL_HOLDOUT_SPLIT],
            Col.MODEL: ["lightgbm"],
            "mae": [1.0],
            "rmse": [2.0],
            "mase": [0.5],
        }
    )
    benchmark = pd.DataFrame(
        {
            Col.TIMESTAMP: [*timestamps, *timestamps],
            Col.TARGET: [10, 20, 30, 40] * 2,
            Col.PREDICTION: [11, 19, 31, 39] * 2,
            Col.MODEL: ["lightgbm"] * 4 + ["seasonal_naive_24h"] * 4,
            Col.SPLIT: [HISTORICAL_HOLDOUT_SPLIT] * 8,
            Col.FOLD: [1] * 8,
        }
    )
    fold_metrics = pd.DataFrame(
        {
            Col.MODEL: ["lightgbm"],
            Col.SPLIT: [HISTORICAL_HOLDOUT_SPLIT],
            Col.FOLD: [1],
            "mae": [1.0],
        }
    )
    probabilistic = pd.DataFrame(
        {
            Col.TIMESTAMP: timestamps,
            Col.TARGET: [10, 20, 30, 40],
            Col.P10: [7, 17, 27, 37],
            Col.P50: [10, 20, 30, 40],
            Col.P90: [13, 23, 33, 43],
            Col.P10_CALIBRATED: [5, 15, 25, 35],
            Col.P90_CALIBRATED: [15, 25, 35, 45],
            Col.P10_HOURLY_CALIBRATED: [6, 16, 26, 36],
            Col.P90_HOURLY_CALIBRATED: [14, 24, 34, 44],
            Col.P10_ROLLING_CALIBRATED: [5, 15, 25, 35],
            Col.P90_ROLLING_CALIBRATED: [15, 25, 35, 45],
            Col.SPLIT: [HISTORICAL_HOLDOUT_SPLIT] * 4,
            Col.FOLD: [1] * 4,
        }
    )
    probabilistic_metrics = pd.DataFrame(
        {
            Col.SPLIT: [HISTORICAL_HOLDOUT_SPLIT],
            "raw_coverage": [0.5],
            "calibrated_coverage": [0.8],
            "hourly_calibrated_coverage": [0.8],
            "rolling_calibrated_coverage": [0.8],
            "raw_mean_width_mw": [5.0],
            "calibrated_mean_width_mw": [10.0],
            "hourly_calibrated_mean_width_mw": [8.0],
            "rolling_calibrated_mean_width_mw": [10.0],
        }
    )
    paths = DashboardPaths(
        history=root / "history.parquet",
        eda_summary=root / "eda.json",
        leaderboard=root / "leaderboard.csv",
        benchmark_forecasts=root / "benchmark.parquet",
        benchmark_fold_metrics=root / "fold_metrics.csv",
        probabilistic_forecasts=root / "probabilistic.parquet",
        probabilistic_metrics=root / "probabilistic_metrics.csv",
        probabilistic_summary=root / "probabilistic.json",
    )
    history.to_parquet(paths.history, index=False)
    leaderboard.to_csv(paths.leaderboard, index=False)
    benchmark.to_parquet(paths.benchmark_forecasts, index=False)
    fold_metrics.to_csv(paths.benchmark_fold_metrics, index=False)
    probabilistic.to_parquet(paths.probabilistic_forecasts, index=False)
    probabilistic_metrics.to_csv(paths.probabilistic_metrics, index=False)
    paths.eda_summary.write_text(
        json.dumps({"observations": 4, "load_mw": {"mean": 25}}),
        encoding="utf-8",
    )
    paths.probabilistic_summary.write_text(
        json.dumps({"conformal_correction_mw": 5, "historical_holdout": {}}),
        encoding="utf-8",
    )
    return paths


def test_load_dashboard_data_validates_and_loads_artifacts(tmp_path: Path) -> None:
    data = load_dashboard_data(_write_dashboard_artifacts(tmp_path))

    assert len(data.history) == 4
    assert data.eda_summary["observations"] == 4
    assert data.probabilistic_summary["conformal_correction_mw"] == 5


def test_load_dashboard_data_reports_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(MissingArtifactsError, match="make data weather"):
        load_dashboard_data(DashboardPaths(history=tmp_path / "missing.parquet"))


def test_load_dashboard_data_rejects_invalid_schema(tmp_path: Path) -> None:
    paths = _write_dashboard_artifacts(tmp_path)
    pd.DataFrame({"wrong": [1]}).to_csv(paths.leaderboard, index=False)

    with pytest.raises(ValueError, match="leaderboard is missing"):
        load_dashboard_data(paths)


def test_dashboard_transformations_select_expected_rows(tmp_path: Path) -> None:
    data = load_dashboard_data(_write_dashboard_artifacts(tmp_path))

    daily = daily_history(data.history)
    point = benchmark_week(
        data.benchmark_forecasts, HISTORICAL_HOLDOUT_SPLIT, 1, ["lightgbm"]
    )
    probabilistic = probabilistic_week(
        data.probabilistic_forecasts, HISTORICAL_HOLDOUT_SPLIT, 1
    )

    assert len(daily) == 1
    assert daily["mean_load_mw"].iloc[0] == pytest.approx(25.0)
    assert len(point) == 4
    assert point[Col.MODEL].eq("lightgbm").all()
    assert len(probabilistic) == 4
    assert display_model("lightgbm_exogenous") == "LightGBM + weather + holidays"
    assert display_model("timesfm_2_5_200m_zero_shot") == ("TimesFM 2.5 200M zero-shot")
    assert display_model("custom_model") == "Custom Model"
