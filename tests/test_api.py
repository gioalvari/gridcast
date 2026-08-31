from pathlib import Path

import httpx
import pandas as pd
import pytest

from gridcast.api import app, get_gridcast_service
from gridcast.api_service import GridCastService
from gridcast.columns import Col
from gridcast.dashboard_data import DashboardData, MissingArtifactsError


def _service() -> GridCastService:
    timestamps = pd.date_range("2018-01-01", periods=2, freq="h")
    validation_timestamps = pd.date_range("2017-12-25", periods=2, freq="h")
    leaderboard = pd.DataFrame(
        {
            Col.SPLIT: ["test", "test"],
            Col.MODEL: ["lightgbm_exogenous", "seasonal_naive_168h"],
            "folds": [52, 52],
            "observations": [8736, 8736],
            "mae": [2901.5, 3585.1],
            "rmse": [3934.2, 4856.2],
            "mase": [0.962, 1.189],
            "mae_improvement_vs_weekly_pct": [19.06, 0.0],
        }
    )
    point = pd.DataFrame(
        {
            Col.TIMESTAMP: [*validation_timestamps, *timestamps, *timestamps],
            Col.TARGET: [29_000.0, 30_000.0, 30_000.0, 31_000.0, 30_000.0, 31_000.0],
            Col.PREDICTION: [
                29_100.0,
                29_900.0,
                30_100.0,
                30_900.0,
                29_000.0,
                30_000.0,
            ],
            Col.MODEL: ["lightgbm_exogenous"] * 4 + ["seasonal_naive_168h"] * 2,
            Col.SPLIT: ["validation"] * 2 + ["test"] * 4,
            Col.FOLD: [1] * 6,
        }
    )
    probabilistic = pd.DataFrame(
        {
            Col.TIMESTAMP: timestamps,
            Col.TARGET: [30_000.0, 31_000.0],
            Col.P10: [28_000.0, 29_000.0],
            Col.P50: [30_000.0, 31_000.0],
            Col.P90: [32_000.0, 33_000.0],
            Col.P10_CALIBRATED: [27_000.0, 28_000.0],
            Col.P90_CALIBRATED: [33_000.0, 34_000.0],
            Col.P10_HOURLY_CALIBRATED: [27_500.0, 28_500.0],
            Col.P90_HOURLY_CALIBRATED: [32_500.0, 33_500.0],
            Col.SPLIT: ["test"] * 2,
            Col.FOLD: [1] * 2,
        }
    )
    data = DashboardData(
        history=pd.DataFrame(
            {Col.TIMESTAMP: timestamps, Col.TARGET: [30_000.0, 31_000.0]}
        ),
        eda_summary={
            "observations": 145392,
            "start": "2002-01-01T01:00:00",
            "end": "2018-08-03T00:00:00",
            "years": 16.58,
            "load_mw": {"mean": 32078.9, "maximum": 62009.0},
        },
        leaderboard=leaderboard,
        benchmark_forecasts=point,
        benchmark_fold_metrics=pd.DataFrame(),
        probabilistic_forecasts=probabilistic,
        probabilistic_metrics=pd.DataFrame(),
        probabilistic_summary={
            "config": {
                "horizon": 168,
                "validation_folds": 12,
                "test_folds": 52,
            },
            "quantiles": [0.1, 0.5, 0.9],
            "target_coverage": 0.8,
            "conformal_correction_mw": 1571.37,
            "test": {
                "raw_coverage": 0.576,
                "calibrated_coverage": 0.8,
                "hourly_calibrated_coverage": 0.814,
                "global_hourly_coverage_mae": 0.024,
                "conditional_hourly_coverage_mae": 0.017,
            },
        },
    )
    return GridCastService(data)


@pytest.fixture
async def client() -> httpx.AsyncClient:
    app.dependency_overrides[get_gridcast_service] = _service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as async_client:
        yield async_client
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_health_does_not_require_artifacts(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "gridcast",
        "version": "1.0.0",
    }


@pytest.mark.anyio
async def test_metadata_returns_evaluation_contract(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"]["observations"] == 145392
    assert payload["experiment"]["test_folds"] == 52
    assert payload["probabilistic"]["calibrated_coverage"] == pytest.approx(0.8)
    assert payload["probabilistic"]["hourly_calibrated_coverage"] == pytest.approx(
        0.814
    )


@pytest.mark.anyio
async def test_leaderboard_is_sorted_by_mae(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/leaderboard")

    assert response.status_code == 200
    payload = response.json()
    assert [row["model"] for row in payload] == [
        "lightgbm_exogenous",
        "seasonal_naive_168h",
    ]
    assert payload[0]["label"] == "LightGBM + weather + holidays"


@pytest.mark.anyio
async def test_point_endpoint_filters_models(client: httpx.AsyncClient) -> None:
    response = await client.get(
        "/api/v1/forecasts/point",
        params=[("fold", "1"), ("models", "lightgbm_exogenous")],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["lightgbm_exogenous"]
    assert len(payload["forecasts"]) == 2
    assert {row["model"] for row in payload["forecasts"]} == {"lightgbm_exogenous"}


@pytest.mark.anyio
async def test_probabilistic_endpoint_returns_calibrated_bounds(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/api/v1/forecasts/probabilistic", params={"fold": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_coverage"] == pytest.approx(0.8)
    assert payload["forecasts"][0]["p10_calibrated_mw"] == 27_000.0
    assert payload["forecasts"][0]["p90_calibrated_mw"] == 33_000.0
    assert payload["forecasts"][0]["p10_hourly_calibrated_mw"] == 27_500.0
    assert payload["forecasts"][0]["p90_hourly_calibrated_mw"] == 32_500.0


@pytest.mark.anyio
async def test_missing_fold_and_invalid_query_return_errors(
    client: httpx.AsyncClient,
) -> None:
    missing = await client.get("/api/v1/forecasts/point", params={"fold": 99})
    invalid = await client.get("/api/v1/leaderboard", params={"split": "random"})

    assert missing.status_code == 404
    assert invalid.status_code == 422


@pytest.mark.anyio
async def test_missing_artifacts_return_service_unavailable(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    def unavailable() -> GridCastService:
        raise MissingArtifactsError(f"missing artifacts under {tmp_path}")

    app.dependency_overrides[get_gridcast_service] = unavailable
    response = await client.get("/api/v1/metadata")

    assert response.status_code == 503
    assert "missing artifacts" in response.json()["detail"]
