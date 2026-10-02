import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gridcast.data import generate_synthetic_load
from gridcast.serving.app import (
    AdmissionController,
    ForecastAdmissionMiddleware,
    create_app,
)
from gridcast.serving.bundle import load_bundle
from gridcast.serving.metrics import create_metrics
from gridcast.serving.predictor import Predictor
from gridcast.serving.resilience import ConcurrencyLimiter
from gridcast.serving.settings import ServingSettings


def settings(**updates: object) -> ServingSettings:
    values: dict[str, object] = {
        "model_uri": "file:///unused",
        "model_cache_dir": Path("/tmp/gridcast-tests"),
        "environment": "local",
        "request_timeout_s": 1.0,
        "max_in_flight": 2,
        "max_admitted": 4,
        "breaker_failures": 2,
        "breaker_reset_s": 10.0,
        "max_body_bytes": 1_000_000,
        "fault_error_rate": 0.0,
        "fault_latency_ms": 0,
        "v1_sunset": "2027-06-30",
    }
    values.update(updates)
    return ServingSettings(**values)  # type: ignore[arg-type]


def payload(hours: int = 336, horizon: int = 168) -> dict[str, object]:
    history = generate_synthetic_load(periods=hours)
    return {
        "history": [
            {"timestamp": row.dte_reference_date.isoformat(), "load_mw": row.load_mw}
            for row in history.itertuples(index=False)
        ],
        "horizon_hours": horizon,
    }


def app_for(bundle_directory: Path, **setting_updates: object):
    return create_app(
        settings(**setting_updates), predictor=Predictor(load_bundle(bundle_directory))
    )


def test_liveness_readiness_health_metrics_and_v2_contract(
    bundle_directory: Path,
) -> None:
    app = app_for(bundle_directory)
    with TestClient(app) as client:
        assert client.get("/live").json() == {"status": "live"}
        assert client.get("/ready").status_code == 200
        health = client.get("/health").json()
        assert health["model_version"] == "0.1.0"
        response = client.post(
            "/v2/forecasts", json=payload(), headers={"X-Request-ID": "client-id"}
        )
        assert response.status_code == 200
        assert response.headers["X-Model-Version"] == "0.1.0"
        assert response.headers["X-Request-ID"] == "client-id"
        forecasts = response.json()["forecasts"]
        assert len(forecasts) == 168
        assert all(
            point["p10_mw"] <= point["p50_mw"] <= point["p90_mw"] for point in forecasts
        )
        generated = client.get("/live")
        assert generated.headers["X-Request-ID"]
        metrics = client.get("/metrics").text
        for name in (
            "gridcast_http_requests_total",
            "gridcast_predictions_total",
            "gridcast_model_info",
            "gridcast_feature_psi",
        ):
            assert name in metrics


def test_readiness_failure_and_draining(bundle_directory: Path) -> None:
    invalid = create_app(settings(model_uri="file:///does-not-exist"))
    with TestClient(invalid) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 503
    app = app_for(bundle_directory)
    with TestClient(app) as client:
        app.state.gridcast["draining"] = True
        assert client.get("/ready").status_code == 503
        assert client.post("/v2/forecasts", json=payload(336, 1)).status_code == 503

    warming = app_for(bundle_directory)
    warming.state.gridcast["predictor"].predict = lambda *_: (_ for _ in ()).throw(
        RuntimeError("warm-up failed")
    )  # type: ignore[method-assign]
    with TestClient(warming) as client:
        assert client.get("/ready").status_code == 503


def test_v1_and_request_body_limits(bundle_directory: Path) -> None:
    app = app_for(bundle_directory, max_body_bytes=20)
    with TestClient(app) as client:
        too_large = client.post("/v2/forecasts", content=b"x" * 21)
        assert too_large.status_code == 413
        assert too_large.headers["X-Request-ID"]
        assert (
            'gridcast_http_requests_total{method="POST",'
            'route="/v2/forecasts",status="413"} 1.0' in client.get("/metrics").text
        )
        bad_length = client.post(
            "/v2/forecasts", content=b"{}", headers={"Content-Length": "invalid"}
        )
        assert bad_length.status_code == 400
    app = app_for(bundle_directory)
    with TestClient(app) as client:
        response = client.post("/v1/forecasts", json=payload(336, 1))
        assert response.status_code == 200
        assert response.headers["Deprecation"] == "true"
        assert "GMT" in response.headers["Sunset"]
        assert response.headers["Link"] == '</v2/forecasts>; rel="successor-version"'


@pytest.mark.parametrize(
    "bad_payload",
    [
        payload(336, 1) | {"history": payload(336, 1)["history"][:-1]},
        payload(336, 1) | {"horizon_hours": 169},
        {
            "history": [
                item
                if index != 1
                else item | {"timestamp": "2024-01-01T02:00:00+00:00"}
                for index, item in enumerate(payload(336, 1)["history"])
            ],
            "horizon_hours": 1,
        },
    ],
)
def test_invalid_requests_are_422_and_do_not_trip_breaker(
    bundle_directory: Path, bad_payload: dict[str, object]
) -> None:
    app = app_for(bundle_directory)
    with TestClient(app) as client:
        response = client.post("/v2/forecasts", json=bad_payload)
        assert response.status_code == 422
        if isinstance(response.json()["detail"], list):
            assert response.json()["detail"][0]["loc"][0] == "body"
        assert app.state.gridcast["breaker"].state == "closed"


def test_overload_timeout_errors_and_circuit_fallbacks(bundle_directory: Path) -> None:
    app = app_for(bundle_directory, max_in_flight=1)
    app.state.gridcast["limiter"] = ConcurrencyLimiter(1)
    app.state.gridcast["limiter"]._in_flight = 1
    with TestClient(app) as client:
        overloaded = client.post("/v2/forecasts", json=payload(336, 1))
        assert overloaded.status_code == 503
        assert overloaded.headers["Retry-After"] == "1"

    timeout_app = app_for(
        bundle_directory, request_timeout_s=0.001, fault_latency_ms=10
    )
    with TestClient(timeout_app) as client:
        timed_out = client.post("/v2/forecasts", json=payload(336, 1)).json()
        assert timed_out["degraded"] is True
        assert timed_out["fallback_reason"] == "timeout"

    failing = app_for(bundle_directory, breaker_failures=2)
    predictor = failing.state.gridcast["predictor"]

    def fail(*_: Any) -> Any:
        raise RuntimeError("broken predictor")

    with TestClient(failing) as client:
        predictor.predict = fail  # type: ignore[method-assign]
        for _ in range(2):
            assert (
                client.post("/v2/forecasts", json=payload(336, 1)).json()[
                    "fallback_reason"
                ]
                == "error"
            )
        opened = client.post("/v2/forecasts", json=payload(336, 1)).json()
        assert opened["fallback_reason"] == "circuit_open"
        metrics = client.get("/metrics").text
        assert "gridcast_fallbacks_total" in metrics
        assert "gridcast_predictions_total" in metrics


def test_model_value_error_falls_back_and_counts_as_breaker_failure(
    bundle_directory: Path,
) -> None:
    app = app_for(bundle_directory)
    predictor = app.state.gridcast["predictor"]

    def fail(*_: Any) -> Any:
        raise ValueError("model output was non-finite")

    with TestClient(app) as client:
        predictor.predict = fail  # type: ignore[method-assign]
        response = client.post("/v2/forecasts", json=payload(336, 1))
        assert response.status_code == 200
        assert response.json()["fallback_reason"] == "error"
        assert app.state.gridcast["breaker"]._failures == 1


def test_invalid_history_is_422_even_when_circuit_is_open(
    bundle_directory: Path,
) -> None:
    app = app_for(bundle_directory)
    app.state.gridcast["breaker"]._opened_at = time.monotonic()
    malformed = payload(336, 1)
    malformed["history"][1]["timestamp"] = "2024-01-01T02:30:00+00:00"  # type: ignore[index]
    with TestClient(app) as client:
        response = client.post("/v2/forecasts", json=malformed)
    assert response.status_code == 422


def test_unmatched_route_has_bounded_metrics_label(bundle_directory: Path) -> None:
    app = app_for(bundle_directory)
    with TestClient(app) as client:
        assert client.get("/scanner/random-id").status_code == 404
        metrics = client.get("/metrics").text
    assert 'route="unmatched"' in metrics
    assert "scanner/random-id" not in metrics


def test_timeout_holds_slot_until_worker_finishes(bundle_directory: Path) -> None:
    app = app_for(bundle_directory, max_in_flight=1, request_timeout_s=0.01)
    predictor = app.state.gridcast["predictor"]
    original_predict = predictor.predict

    def slow_predict(*args: Any) -> Any:
        time.sleep(0.1)
        return original_predict(*args)

    with TestClient(app) as client:
        predictor.predict = slow_predict  # type: ignore[method-assign]
        first = client.post("/v2/forecasts", json=payload(336, 1))
        assert first.status_code == 200
        assert first.json()["fallback_reason"] == "timeout"
        assert app.state.gridcast["limiter"].in_flight == 1
        assert client.post("/v2/forecasts", json=payload(336, 1)).status_code == 503
        time.sleep(0.15)
        predictor.predict = original_predict  # type: ignore[method-assign]
        accepted = client.post("/v2/forecasts", json=payload(336, 1))
    assert accepted.status_code == 200


def test_admission_rejects_before_body_read_and_records_metrics() -> None:
    controller = AdmissionController(1)
    assert controller.try_acquire()
    metrics = create_metrics()
    downstream_called = False
    sent: list[dict[str, object]] = []

    async def downstream(*_: object) -> None:
        nonlocal downstream_called
        downstream_called = True

    async def receive() -> dict[str, object]:
        raise AssertionError("rejected body must not be read")

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    middleware = ForecastAdmissionMiddleware(
        downstream, controller=controller, metrics=metrics
    )
    asyncio.run(
        middleware(
            {"type": "http", "method": "POST", "path": "/v2/forecasts"},
            receive,
            send,
        )
    )

    assert not downstream_called
    assert sent[0]["status"] == 503
    assert (b"retry-after", b"1") in sent[0]["headers"]  # type: ignore[operator]
    assert (b"x-request-id",) == (sent[0]["headers"][2][0],)  # type: ignore[index]
    exposition = metrics.registry.get_sample_value(
        "gridcast_overload_rejections_total", {"stage": "admission"}
    )
    assert exposition == 1
    assert (
        metrics.registry.get_sample_value(
            "gridcast_http_requests_total",
            {"route": "/v2/forecasts", "method": "POST", "status": "503"},
        )
        == 1
    )


def test_admission_bypasses_health_and_releases_slots_on_all_outcomes() -> None:
    controller = AdmissionController(1)
    metrics = create_metrics()
    sent: list[dict[str, object]] = []
    calls: list[str] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    async def normal(scope: dict[str, object], *_: object) -> None:
        calls.append(scope["path"])  # type: ignore[arg-type]

    middleware = ForecastAdmissionMiddleware(
        normal, controller=controller, metrics=metrics
    )
    assert controller.try_acquire()
    asyncio.run(
        middleware({"type": "http", "method": "GET", "path": "/ready"}, receive, send)
    )
    assert calls == ["/ready"]
    controller.release()

    asyncio.run(
        middleware(
            {"type": "http", "method": "POST", "path": "/v1/forecasts"},
            receive,
            send,
        )
    )
    assert controller.admitted == 0

    async def exception(*_: object) -> None:
        raise RuntimeError("downstream failed")

    failing = ForecastAdmissionMiddleware(
        exception, controller=controller, metrics=metrics
    )
    with pytest.raises(RuntimeError, match="downstream failed"):
        asyncio.run(
            failing(
                {"type": "http", "method": "POST", "path": "/v2/forecasts"},
                receive,
                send,
            )
        )
    assert controller.admitted == 0

    async def disconnected(*_: object) -> None:
        raise asyncio.CancelledError

    cancelled = ForecastAdmissionMiddleware(
        disconnected, controller=controller, metrics=metrics
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            cancelled(
                {"type": "http", "method": "POST", "path": "/v2/forecasts"},
                receive,
                send,
            )
        )
    assert controller.admitted == 0
