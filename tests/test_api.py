import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from gridcast.api import app, get_gridcast_service
from gridcast.api_service import GridCastService
from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, Col
from gridcast.dashboard_data import DashboardData, MissingArtifactsError
from gridcast.foundation_models import TIMESFM_2P5, TIMESFM_3
from gridcast.model_comparison import COMPARISONS


def _service() -> GridCastService:
    timestamps = pd.date_range("2018-01-01", periods=2, freq="h")
    validation_timestamps = pd.date_range("2017-12-25", periods=2, freq="h")
    leaderboard = pd.DataFrame(
        {
            Col.SPLIT: [HISTORICAL_HOLDOUT_SPLIT] * 2,
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
            Col.SPLIT: ["validation"] * 2 + [HISTORICAL_HOLDOUT_SPLIT] * 4,
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
            Col.P10_ROLLING_CALIBRATED: [27_200.0, 28_200.0],
            Col.P90_ROLLING_CALIBRATED: [32_800.0, 33_800.0],
            Col.SPLIT: [HISTORICAL_HOLDOUT_SPLIT] * 2,
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
                "holdout_folds": 52,
            },
            "quantiles": [0.1, 0.5, 0.9],
            "target_coverage": 0.8,
            "conformal_correction_mw": 1571.37,
            "historical_holdout": {
                "raw_coverage": 0.576,
                "calibrated_coverage": 0.8,
                "hourly_calibrated_coverage": 0.814,
                "global_hourly_coverage_mae": 0.024,
                "conditional_hourly_coverage_mae": 0.017,
                "rolling_calibrated_coverage": 0.805,
                "global_weekly_coverage_mae": 0.12,
                "rolling_weekly_coverage_mae": 0.10,
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
async def test_readiness_requires_valid_core_artifacts(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "gridcast",
        "version": "1.0.0",
    }


@pytest.mark.anyio
async def test_metadata_returns_evaluation_contract(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"]["observations"] == 145392
    assert payload["experiment"]["holdout_folds"] == 52
    assert payload["probabilistic"]["calibrated_coverage"] == pytest.approx(0.8)
    assert payload["probabilistic"]["hourly_calibrated_coverage"] == pytest.approx(
        0.814
    )
    assert payload["probabilistic"]["rolling_calibrated_coverage"] == pytest.approx(
        0.805
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
    assert payload["forecasts"][0]["p10_rolling_calibrated_mw"] == 27_200.0


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
    response = await client.get("/ready")

    assert response.status_code == 503
    assert "missing artifacts" in response.json()["detail"]


@pytest.mark.anyio
async def test_invalid_core_artifacts_return_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid() -> DashboardData:
        raise ValueError("invalid leaderboard schema")

    app.dependency_overrides.clear()
    get_gridcast_service.cache_clear()
    monkeypatch.setattr("gridcast.api.load_dashboard_data", invalid)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as invalid_client:
        response = await invalid_client.get("/ready")
    get_gridcast_service.cache_clear()

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "invalid core artifacts: invalid leaderboard schema"
    )


@pytest.mark.anyio
async def test_performance_endpoint_returns_optional_measurements(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gridcast.api_service.load_performance_summary",
        lambda path: {
            "environment": {"platform": "test", "logical_cpu_count": 4},
            "measurements": [
                {
                    "model": "lightgbm",
                    "fit_time_ms": 100.0,
                    "prediction_median_ms": 0.5,
                    "prediction_p95_ms": 0.8,
                    "throughput_rows_per_second": 336_000.0,
                    "serialized_model_kib": 836.0,
                    "fit_rss_delta_mib": 8.0,
                    "horizon_rows": 168,
                    "repetitions": 100,
                }
            ],
        },
    )

    response = await client.get("/api/v1/performance")

    assert response.status_code == 200
    assert response.json()["measurements"][0]["prediction_median_ms"] == 0.5


@pytest.mark.anyio
async def test_performance_endpoint_reports_missing_artifact(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(path: Path) -> dict[str, object]:
        raise FileNotFoundError(path)

    monkeypatch.setattr("gridcast.api_service.load_performance_summary", missing)

    response = await client.get("/api/v1/performance")

    assert response.status_code == 404
    assert "make performance" in response.json()["detail"]


@pytest.mark.anyio
async def test_decisions_endpoint_returns_optional_results(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = pd.DataFrame(
        {
            "scenario": ["symmetric"],
            "model": ["lightgbm"],
            "shortage_cost": [1.0],
            "surplus_cost": [1.0],
            "optimal_quantile": [0.5],
            "folds": [52],
            "observations": [8736],
            "mean_cost": [100.0],
            "regret_vs_perfect": [100.0],
            "cost_increase_vs_best_pct": [0.0],
        }
    )
    quantile = pd.DataFrame(
        {
            "scenario": ["symmetric"],
            "shortage_cost": [1.0],
            "surplus_cost": [1.0],
            "selected_quantile": [0.5],
            "p50_cost": [100.0],
            "cost_aware_quantile_cost": [100.0],
            "cost_savings_pct": [0.0],
        }
    )
    point_path = tmp_path / "benchmark" / "decision_costs.csv"
    quantile_path = tmp_path / "probabilistic" / "decision_costs.csv"
    point_path.parent.mkdir()
    quantile_path.parent.mkdir()
    point.to_csv(point_path, index=False)
    quantile.to_csv(quantile_path, index=False)

    monkeypatch.setattr("gridcast.api_service.POINT_DECISIONS_PATH", point_path)
    monkeypatch.setattr("gridcast.api_service.QUANTILE_DECISIONS_PATH", quantile_path)

    response = await client.get("/api/v1/decisions")

    assert response.status_code == 200
    assert response.json()["point_models"][0]["model"] == "lightgbm"


def _comparison_summary() -> dict[str, object]:
    model_mae = {
        "seasonal_naive_24h": 3000.0,
        "lightgbm_exogenous": 2900.0,
        TIMESFM_2P5.model_name: 1900.0,
        TIMESFM_3.model_name: 1750.0,
    }
    family_size = len(COMPARISONS)
    adjusted_level = 1.0 - 0.05 / family_size
    comparisons: list[dict[str, object]] = []
    for order, (comparison_id, candidate, reference) in enumerate(COMPARISONS, 1):
        improvement = model_mae[reference] - model_mae[candidate]
        comparisons.append(
            {
                "comparison_order": order,
                "comparison_id": comparison_id,
                "candidate_model": candidate,
                "reference_model": reference,
                "folds": 52,
                "observations_per_fold": 168,
                "candidate_mae_mw": model_mae[candidate],
                "reference_mae_mw": model_mae[reference],
                "mean_mae_improvement_mw": improvement,
                "relative_improvement_pct": (
                    100.0 * improvement / model_mae[reference]
                ),
                "wins": 31,
                "ties": 0,
                "losses": 21,
                "weekly_win_rate": 31 / 52,
                "bootstrap_method": "circular",
                "block_length_folds": 4,
                "bootstrap_replicates": 100_000,
                "bootstrap_seed": 20_260_903,
                "confidence_level": 0.95,
                "ci_low_mw": improvement - 100.0,
                "ci_high_mw": improvement + 100.0,
                "relative_ci_low_pct": -1.0,
                "relative_ci_high_pct": 1.0,
                "weekly_win_rate_ci_low": 0.48,
                "weekly_win_rate_ci_high": 0.71,
                "bootstrap_directional_support": 0.652,
                "first_half_improvement_mw": improvement - 10.0,
                "second_half_improvement_mw": improvement + 10.0,
                "adjusted_ci_low_mw": improvement - 200.0,
                "adjusted_ci_high_mw": improvement + 200.0,
                "adjusted_relative_ci_low_pct": -2.0,
                "adjusted_relative_ci_high_pct": 2.0,
                "simultaneous_superiority_supported": improvement > 200.0,
                "family_size": family_size,
                "multiplicity_method": "Bonferroni",
                "familywise_confidence_level": 0.95,
                "adjusted_per_comparison_confidence_level": adjusted_level,
            }
        )
    return {
        "schema_version": 1,
        "config": {
            "bootstrap_replicates": 100_000,
            "block_length_folds": 4,
            "seed": 20_260_903,
            "confidence_level": 0.95,
            "sensitivity_block_lengths": [2, 4, 6, 8, 13, 26],
            "expected_folds": 52,
            "observations_per_fold": 168,
        },
        "orientation": "positive favors candidate",
        "bootstrap_method": "paired circular block bootstrap",
        "rng": "numpy.random.PCG64",
        "quantile_method": "linear",
        "family_size": family_size,
        "multiplicity_method": "Bonferroni",
        "familywise_confidence_level": 0.95,
        "adjusted_per_comparison_confidence_level": adjusted_level,
        "resample_indices_sha256": "a" * 64,
        "source_artifacts": {},
        "provenance_warnings": ["legacy upstream manifest"],
        "comparisons": comparisons,
    }


@pytest.mark.anyio
async def test_comparisons_endpoint_returns_paired_uncertainty(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_comparison_summary()), encoding="utf-8")
    monkeypatch.setattr("gridcast.api_service.COMPARISON_SUMMARY_PATH", summary_path)

    response = await client.get("/api/v1/comparisons")

    assert response.status_code == 200
    payload = response.json()
    assert payload["bootstrap_replicates"] == 100_000
    assert payload["comparisons"][0]["mean_mae_improvement_mw"] == 100.0
    assert payload["familywise_confidence_level"] == 0.95
    assert payload["provenance_warnings"] == ["legacy upstream manifest"]


@pytest.mark.anyio
@pytest.mark.parametrize("contents", [None, "not JSON", "{}"])
async def test_comparisons_endpoint_handles_missing_or_invalid_artifact(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str | None,
) -> None:
    summary_path = tmp_path / "summary.json"
    if contents is not None:
        summary_path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr("gridcast.api_service.COMPARISON_SUMMARY_PATH", summary_path)

    response = await client.get("/api/v1/comparisons")

    assert response.status_code == (404 if contents is None else 503)


@pytest.mark.anyio
@pytest.mark.parametrize("invalid", ["cross_row_mae", "adjusted_interval"])
async def test_comparisons_endpoint_rejects_inconsistent_family(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    summary = _comparison_summary()
    comparisons = summary["comparisons"]
    assert isinstance(comparisons, list)
    first = comparisons[0]
    assert isinstance(first, dict)
    if invalid == "cross_row_mae":
        second = comparisons[1]
        assert isinstance(second, dict)
        second["reference_mae_mw"] = 3001.0
        second["mean_mae_improvement_mw"] = 1101.0
        second["relative_improvement_pct"] = 100.0 * 1101.0 / 3001.0
    else:
        first["adjusted_ci_low_mw"] = first["ci_low_mw"] + 1.0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr("gridcast.api_service.COMPARISON_SUMMARY_PATH", summary_path)

    response = await client.get("/api/v1/comparisons")

    assert response.status_code == 503


@pytest.mark.anyio
async def test_foundation_endpoint_returns_optional_result(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": {
                    "model_name": TIMESFM_2P5.model_name,
                    "model_id": TIMESFM_2P5.model_id,
                    "model_revision": TIMESFM_2P5.model_revision,
                    "context_length": 1024,
                    "horizon": 168,
                    "holdout_folds": 52,
                    "model_parameters": {"per_core_batch_size": 1},
                },
                "metrics": {
                    "model": TIMESFM_2P5.model_name,
                    "observations": 8736,
                    "folds": 52,
                    "mae": 1926.88,
                    "rmse": 2740.02,
                    "mase": 0.639,
                    "p10_pinball_loss": 433.01,
                    "p50_pinball_loss": 963.44,
                    "p90_pinball_loss": 459.82,
                    "raw_80_coverage": 0.749,
                    "raw_80_mean_width_mw": 5455.59,
                },
                "timing": {
                    "first_call_seconds": 51.88,
                    "warm_call_seconds": 9.34,
                },
                "environment": {
                    "python": "3.12.12",
                    "platform": "test-arm64",
                    "torch": "2.13.0",
                    "timesfm": "2.0.2",
                    "device": "cpu",
                    "checkpoint_sha256": "a" * 64,
                    "dependency_lock_sha256": "b" * 64,
                    "installed_packages": {"timesfm": "2.0.2"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("gridcast.api_service.FOUNDATION_SUMMARY_PATH", summary_path)

    response = await client.get("/api/v1/foundation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["mae"] == pytest.approx(1926.88)
    assert payload["horizon"] == 168
    assert payload["weights_license"] == "Apache-2.0"
    assert payload["model_parameters"]["per_core_batch_size"] == 1
    assert payload["timing"]["warm_call_seconds"] == pytest.approx(9.34)
    assert "installed_packages" not in payload["environment"]
    assert payload["environment"]["timesfm_lock_sha256"] == "b" * 64


@pytest.mark.anyio
@pytest.mark.parametrize(
    "contents",
    [
        None,
        "not JSON",
        "{}",
        json.dumps(
            {
                "config": {
                    "model_name": "timesfm_2_5_200m_zero_shot",
                    "model_id": "model",
                    "model_revision": "revision",
                    "context_length": 1024,
                    "horizon": 168,
                    "holdout_folds": 2,
                    "model_parameters": {},
                },
                "metrics": {
                    "model": "timesfm_2_5_200m_zero_shot",
                    "observations": 168,
                    "folds": 2,
                    "mae": 1.0,
                    "rmse": 1.0,
                    "mase": 1.0,
                    "p10_pinball_loss": 1.0,
                    "p50_pinball_loss": 1.0,
                    "p90_pinball_loss": 1.0,
                    "raw_80_coverage": 0.8,
                    "raw_80_mean_width_mw": 1.0,
                },
                "timing": {
                    "first_call_seconds": 1.0,
                    "warm_call_seconds": 1.0,
                },
                "environment": {},
            }
        ),
        """
        {
          "config": {
            "model_name": "timesfm_2_5_200m_zero_shot",
            "model_id": "model",
            "model_revision": "revision",
            "context_length": 1024,
            "horizon": 168,
            "holdout_folds": 1,
            "model_parameters": {}
          },
          "metrics": {
            "model": "timesfm_2_5_200m_zero_shot",
            "observations": 168,
            "folds": 1,
            "mae": 1.0,
            "rmse": 1.0,
            "mase": 1.0,
            "p10_pinball_loss": 1.0,
            "p50_pinball_loss": 1.0,
            "p90_pinball_loss": 1.0,
            "raw_80_coverage": 0.8,
            "raw_80_mean_width_mw": 1.0
          },
          "timing": {"first_call_seconds": 1.0, "warm_call_seconds": 1.0},
          "environment": {
            "python": "3.12.12",
            "platform": "test-arm64",
            "torch": "2.13.0",
            "timesfm": "2.0.2",
            "device": "cpu",
            "checkpoint_sha256": "CHECKPOINT_DIGEST",
            "dependency_lock_sha256": "LOCK_DIGEST",
            "installed_packages": {"bad": NaN}
          }
        }
        """.replace("CHECKPOINT_DIGEST", "a" * 64).replace("LOCK_DIGEST", "b" * 64),
    ],
)
async def test_foundation_endpoint_handles_missing_or_invalid_artifact(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str | None,
) -> None:
    summary_path = tmp_path / "summary.json"
    if contents is not None:
        summary_path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr("gridcast.api_service.FOUNDATION_SUMMARY_PATH", summary_path)

    response = await client.get("/api/v1/foundation")

    assert response.status_code == (404 if contents is None else 503)


@pytest.mark.anyio
async def test_timesfm3_endpoint_returns_not_found_without_artifact(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gridcast.api_service.FOUNDATION3_SUMMARY_PATH",
        tmp_path / "summary.json",
    )

    response = await client.get("/api/v1/foundation/timesfm-3.0")

    assert response.status_code == 404
    assert "make timesfm3" in response.json()["detail"]


@pytest.mark.anyio
@pytest.mark.parametrize("invalid_field", [None, "model_name", "checkpoint", "config"])
async def test_timesfm3_endpoint_validates_checkpoint_identity(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_field: str | None,
) -> None:
    summary_path = tmp_path / "summary.json"
    model_name = (
        TIMESFM_2P5.model_name
        if invalid_field == "model_name"
        else TIMESFM_3.model_name
    )
    summary_path.write_text(
        json.dumps(
            {
                "config": {
                    "model_name": model_name,
                    "model_id": TIMESFM_3.model_id,
                    "model_revision": TIMESFM_3.model_revision,
                    "weights_license": TIMESFM_3.weights_license,
                    "context_length": 1024,
                    "horizon": 168,
                    "holdout_folds": 1,
                    "model_parameters": {"per_core_batch_size": 1},
                },
                "metrics": {
                    "model": model_name,
                    "observations": 168,
                    "folds": 1,
                    "mae": 1.0,
                    "rmse": 1.0,
                    "mase": 1.0,
                    "p10_pinball_loss": 1.0,
                    "p50_pinball_loss": 1.0,
                    "p90_pinball_loss": 1.0,
                    "raw_80_coverage": 0.8,
                    "raw_80_mean_width_mw": 1.0,
                },
                "timing": {
                    "first_call_seconds": 1.0,
                    "warm_call_seconds": 1.0,
                },
                "environment": {
                    "python": "3.12.12",
                    "platform": "test-arm64",
                    "torch": "2.13.0",
                    "timesfm": "3.0.0",
                    "device": "cpu",
                    "checkpoint_sha256": (
                        "a" * 64
                        if invalid_field == "checkpoint"
                        else TIMESFM_3.checkpoint_sha256
                    ),
                    "checkpoint_config_sha256": (
                        "b" * 64
                        if invalid_field == "config"
                        else TIMESFM_3.config_sha256
                    ),
                    "dependency_lock_sha256": "b" * 64,
                    "installed_packages": {"timesfm": "3.0.0"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("gridcast.api_service.FOUNDATION3_SUMMARY_PATH", summary_path)

    response = await client.get("/api/v1/foundation/timesfm-3.0")

    assert response.status_code == (200 if invalid_field is None else 503)
