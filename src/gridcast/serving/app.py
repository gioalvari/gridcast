"""FastAPI application factory for resilient GridCast model serving.

Forecast requests flow through client-history validation, limiter admission,
circuit-breaker admission, a deadline-bound primary prediction, then a
seasonal-naive fallback when primary serving is unavailable.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from email.utils import format_datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from gridcast import __version__
from gridcast.columns import Col
from gridcast.data import generate_synthetic_load
from gridcast.provenance import file_sha256
from gridcast.serving.bundle import LoadedBundle, load_bundle
from gridcast.serving.metrics import ServingMetrics, create_metrics
from gridcast.serving.predictor import (
    ForecastResult,
    InvalidHistoryError,
    Predictor,
    fallback_forecast,
    population_stability_index,
    validate_history,
)
from gridcast.serving.registry import resolve_model_uri
from gridcast.serving.resilience import (
    CircuitBreaker,
    ConcurrencyLimiter,
    FaultInjector,
    OverloadedError,
)
from gridcast.serving.settings import ServingSettings

LOGGER = logging.getLogger(__name__)
MAX_HISTORY_HOURS = 24 * 7 * 8
FORECAST_PATHS = frozenset({"/v1/forecasts", "/v2/forecasts"})


class AdmissionController:
    """Keep a bounded count of forecast requests before their bodies are read."""

    def __init__(self, max_admitted: int) -> None:
        """Initialize a controller with a positive request capacity."""
        self.max_admitted = max_admitted
        self._admitted = 0

    @property
    def admitted(self) -> int:
        """Return the number of requests currently admitted for parsing or serving."""
        return self._admitted

    def try_acquire(self) -> bool:
        """Reserve a request slot without waiting."""
        if self._admitted >= self.max_admitted:
            return False
        self._admitted += 1
        return True

    def release(self) -> None:
        """Release one request slot after the ASGI response completes."""
        self._admitted -= 1


class ForecastAdmissionMiddleware:
    """Reject excess forecast requests before body parsing or routing work."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        controller: AdmissionController,
        metrics: ServingMetrics,
    ) -> None:
        """Wrap an ASGI app with forecast-only pre-body admission control."""
        self.app = app
        self.controller = controller
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reserve a slot around the complete downstream ASGI exchange."""
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in FORECAST_PATHS
        ):
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        started = time.perf_counter()
        route = scope["path"]
        if not self.controller.try_acquire():
            self.metrics.overload_rejections.labels("admission").inc()
            self.metrics.http_requests.labels(route, "POST", 503).inc()
            self.metrics.http_duration.labels(route).observe(
                time.perf_counter() - started
            )
            await _send_json(
                send,
                503,
                {"detail": "service overloaded"},
                request_id=request_id,
                retry_after="1",
            )
            return

        self.metrics.admitted_requests.set(self.controller.admitted)
        try:
            await self.app(scope, receive, send)
        finally:
            self.controller.release()
            self.metrics.admitted_requests.set(self.controller.admitted)


class HistoryRecord(BaseModel):
    """One timestamped observed load provided to the serving endpoint."""

    model_config = ConfigDict(extra="forbid")
    timestamp: datetime = Field(
        description="Timezone-aware hourly observation timestamp."
    )
    load_mw: float = Field(gt=0, description="Observed positive electrical load in MW.")

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject naive client timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value


class ForecastRequest(BaseModel):
    """Validated request contract shared by v1 and v2 forecasts."""

    model_config = ConfigDict(extra="forbid")
    history: list[HistoryRecord] = Field(
        min_length=336,
        max_length=MAX_HISTORY_HOURS,
        description="Sorted, unique, regular hourly positive load history.",
    )
    horizon_hours: int = Field(
        default=168, ge=1, le=168, description="Number of hourly forecasts requested."
    )


class ForecastPointV2(BaseModel):
    """Calibrated point and quantile forecast for one hour."""

    timestamp: datetime
    p10_mw: float
    p50_mw: float
    p90_mw: float
    point_mw: float


class ForecastResponseV2(BaseModel):
    """Version 2 probabilistic serving response."""

    api_version: str = "v2"
    model_version: str
    origin: datetime
    horizon_hours: int
    degraded: bool
    fallback_reason: str | None
    forecasts: list[ForecastPointV2]


class ForecastPointV1(BaseModel):
    """Point-only compatibility forecast for one hour."""

    timestamp: datetime
    load_mw: float


class ForecastResponseV1(BaseModel):
    """Deprecated version 1 point-only serving response."""

    api_version: str = "v1"
    model_version: str
    origin: datetime
    horizon_hours: int
    degraded: bool
    forecasts: list[ForecastPointV1]


def create_app(
    settings: ServingSettings | None = None, *, predictor: Predictor | None = None
) -> FastAPI:
    """Create an independently instrumented FastAPI production serving application."""
    settings = settings or ServingSettings.from_env()
    metrics = create_metrics()
    state: dict[str, Any] = {
        "predictor": predictor,
        "ready": predictor is not None,
        "draining": False,
        "breaker": CircuitBreaker(settings.breaker_failures, settings.breaker_reset_s),
        "limiter": ConcurrencyLimiter(settings.max_in_flight),
        "admission": AdmissionController(settings.max_admitted),
        "faults": FaultInjector(settings.fault_error_rate, settings.fault_latency_ms),
        "executor": None,
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["executor"] = ThreadPoolExecutor(
            max_workers=settings.max_in_flight,
            thread_name_prefix="gridcast-predict",
        )
        if state["predictor"] is None:
            try:
                path = resolve_model_uri(settings.model_uri, settings.model_cache_dir)
                state["predictor"] = Predictor(load_bundle(path))
            except Exception:
                LOGGER.exception("model bundle could not be loaded")
        active = state["predictor"]
        if active is not None:
            try:
                _warm_up(active)
                state["ready"] = True
                _record_model_info(metrics, active.bundle)
            except Exception:
                state["ready"] = False
                LOGGER.exception("model warm-up failed")
        try:
            yield
        finally:
            state["draining"] = True
            state["ready"] = False
            executor: ThreadPoolExecutor | None = state["executor"]
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="GridCast Forecast Serving",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.gridcast = state
    app.state.metrics = metrics

    @app.middleware("http")
    async def observe_requests(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        started = time.perf_counter()

        def finalize(response: Response, route: str) -> Response:
            duration = time.perf_counter() - started
            metrics.http_requests.labels(
                route, request.method, response.status_code
            ).inc()
            metrics.http_duration.labels(route).observe(duration)
            response.headers["X-Request-ID"] = request_id
            LOGGER.info(
                json.dumps(
                    {
                        "route": route,
                        "status": response.status_code,
                        "duration_ms": round(duration * 1000, 3),
                        "request_id": request_id,
                        "model_version": _model_version(state),
                        "degraded": response.headers.get("X-GridCast-Degraded")
                        == "true",
                    }
                )
            )
            return response

        if request.method in {"POST", "PUT", "PATCH"}:
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    body_length = int(length)
                except ValueError:
                    response = JSONResponse(
                        {"detail": "invalid Content-Length"}, status_code=400
                    )
                    return finalize(response, _early_route_label(app, request))
                if body_length > settings.max_body_bytes:
                    response = JSONResponse(
                        {"detail": "request body too large"}, status_code=413
                    )
                    return finalize(response, _early_route_label(app, request))
        response = await call_next(request)
        return finalize(response, _route_label(request))

    @app.get("/live")
    async def live() -> dict[str, str]:
        """Return liveness independent of model-load state."""
        return {"status": "live"}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        """Return readiness only after load and warm-up complete."""
        if not state["ready"] or state["draining"]:
            raise HTTPException(status_code=503, detail="model is not ready")
        return {"status": "ready", "model_version": _model_version(state) or "unknown"}

    @app.get("/health")
    async def health() -> dict[str, object]:
        """Expose non-sensitive process and serving state."""
        breaker: CircuitBreaker = state["breaker"]
        limiter: ConcurrencyLimiter = state["limiter"]
        return {
            "model_version": _model_version(state),
            "circuit_state": breaker.state,
            "in_flight": limiter.in_flight,
            "environment": settings.environment,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics() -> str:
        """Return isolated Prometheus exposition text."""
        return generate_latest(metrics.registry).decode("utf-8")

    @app.post(
        "/v2/forecasts",
        response_model=ForecastResponseV2,
        openapi_extra=_forecast_request_body(),
    )
    async def forecast_v2(request: Request, response: Response) -> Response:
        """Serve calibrated point and interval forecasts."""
        payload = await _parse_forecast_request(request, settings.max_body_bytes)
        result = await _forecast(payload, "v2", state, metrics, settings)
        model = ForecastResponseV2(
            model_version=result.model_version,
            origin=result.timestamps.iloc[0],
            horizon_hours=len(result.point),
            degraded=result.degraded,
            fallback_reason=result.fallback_reason,
            forecasts=[
                ForecastPointV2(
                    timestamp=timestamp,
                    p10_mw=float(p10),
                    p50_mw=float(p50),
                    p90_mw=float(p90),
                    point_mw=float(point),
                )
                for timestamp, p10, p50, p90, point in zip(
                    result.timestamps,
                    result.p10,
                    result.p50,
                    result.p90,
                    result.point,
                    strict=True,
                )
            ],
        )
        response.headers["X-Model-Version"] = result.model_version
        response.headers["X-GridCast-Degraded"] = str(result.degraded).lower()
        return Response(
            content=model.model_dump_json(),
            headers=dict(response.headers),
            media_type="application/json",
        )

    @app.post(
        "/v1/forecasts",
        response_model=ForecastResponseV1,
        openapi_extra=_forecast_request_body(),
    )
    async def forecast_v1(request: Request, response: Response) -> Response:
        """Serve deprecated point-only compatibility forecasts."""
        payload = await _parse_forecast_request(request, settings.max_body_bytes)
        result = await _forecast(payload, "v1", state, metrics, settings)
        response.headers.update(
            {
                "X-Model-Version": result.model_version,
                "X-GridCast-Degraded": str(result.degraded).lower(),
                "Deprecation": "true",
                "Sunset": _sunset_header(settings.v1_sunset),
                "Link": '</v2/forecasts>; rel="successor-version"',
            }
        )
        model = ForecastResponseV1(
            model_version=result.model_version,
            origin=result.timestamps.iloc[0],
            horizon_hours=len(result.point),
            degraded=result.degraded,
            forecasts=[
                ForecastPointV1(timestamp=timestamp, load_mw=float(point))
                for timestamp, point in zip(
                    result.timestamps, result.point, strict=True
                )
            ],
        )
        return Response(
            content=model.model_dump_json(),
            headers=dict(response.headers),
            media_type="application/json",
        )

    app.add_middleware(
        ForecastAdmissionMiddleware,
        controller=state["admission"],
        metrics=metrics,
    )
    _install_openapi_schema(app)
    return app


def _forecast_request_body() -> dict[str, object]:
    """Keep manually parsed forecast bodies visible in the OpenAPI contract."""
    return {
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ForecastRequest"}
                }
            },
            "required": True,
        },
        "responses": {
            "422": {
                "description": "Validation Error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/HTTPValidationError"}
                    }
                },
            }
        },
    }


def _install_openapi_schema(app: FastAPI) -> None:
    """Restore component definitions for the manually parsed request body."""
    original_openapi = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = original_openapi()
            components = schema.setdefault("components", {}).setdefault("schemas", {})
            request_schema = ForecastRequest.model_json_schema(
                ref_template="#/components/schemas/{model}"
            )
            history = request_schema["properties"]["history"]
            horizon = request_schema["properties"]["horizon_hours"]
            history["items"]["$ref"] = "#/components/schemas/HistoryRecord"
            horizon["minimum"] = float(horizon["minimum"])
            horizon["maximum"] = float(horizon["maximum"])
            request_definitions = request_schema["$defs"]
            request_definitions["HistoryRecord"]["properties"]["load_mw"][
                "exclusiveMinimum"
            ] = 0.0
            components.update(request_schema.pop("$defs", {}))
            components["ForecastRequest"] = request_schema
            components.update(_validation_error_schemas())
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


def _validation_error_schemas() -> dict[str, object]:
    """Provide FastAPI's standard 422 models for manual validation failures."""
    return {
        "HTTPValidationError": {
            "properties": {
                "detail": {
                    "items": {"$ref": "#/components/schemas/ValidationError"},
                    "title": "Detail",
                    "type": "array",
                }
            },
            "title": "HTTPValidationError",
            "type": "object",
        },
        "ValidationError": {
            "properties": {
                "ctx": {"title": "Context", "type": "object"},
                "input": {"title": "Input"},
                "loc": {
                    "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                    "title": "Location",
                    "type": "array",
                },
                "msg": {"title": "Message", "type": "string"},
                "type": {"title": "Error Type", "type": "string"},
            },
            "required": ["loc", "msg", "type"],
            "title": "ValidationError",
            "type": "object",
        },
    }


async def _parse_forecast_request(
    request: Request, max_body_bytes: int
) -> ForecastRequest:
    """Read a bounded body and validate it through Pydantic's JSON parser."""
    body = await request.body()
    if len(body) > max_body_bytes:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        return ForecastRequest.model_validate_json(body)
    except ValidationError as error:
        errors = []
        for item in error.errors():
            errors.append({**item, "loc": ("body", *item["loc"])})
        raise RequestValidationError(errors) from error


def _request_id(scope: Scope) -> str:
    """Use the caller request ID where supplied, otherwise generate one."""
    for name, value in scope.get("headers", []):
        if name.lower() == b"x-request-id":
            return str(value.decode("latin-1"))
    return str(uuid.uuid4())


async def _send_json(
    send: Send,
    status: int,
    body: dict[str, str],
    *,
    request_id: str,
    retry_after: str | None = None,
) -> None:
    """Send a small JSON ASGI response without invoking request-body receive."""
    content = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(content)).encode()),
        (b"x-request-id", request_id.encode("latin-1")),
    ]
    if retry_after is not None:
        headers.append((b"retry-after", retry_after.encode()))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": content})


async def _forecast(
    payload: ForecastRequest,
    api_version: str,
    state: dict[str, Any],
    metrics: ServingMetrics,
    settings: ServingSettings,
) -> ForecastResult:
    history = pd.DataFrame(
        {
            Col.TIMESTAMP: [record.timestamp for record in payload.history],
            Col.TARGET: [record.load_mw for record in payload.history],
        }
    )
    active: Predictor | None = state["predictor"]
    if active is None or not state["ready"] or state["draining"]:
        raise HTTPException(status_code=503, detail="model is not ready")
    try:
        validate_history(history, active.bundle.manifest.min_history_hours)
    except InvalidHistoryError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    limiter: ConcurrencyLimiter = state["limiter"]
    acquired_slot = False
    release_slot = True
    try:
        limiter.acquire()
        acquired_slot = True
        metrics.in_flight.set(limiter.in_flight)
        result, release_slot = await _run_forecast(
            active,
            history,
            payload.horizon_hours,
            api_version,
            state,
            metrics,
            settings,
        )
        return result
    except OverloadedError as error:
        metrics.overload_rejections.labels("execution").inc()
        raise HTTPException(
            status_code=503, detail="service overloaded", headers={"Retry-After": "1"}
        ) from error
    finally:
        if acquired_slot and release_slot:
            limiter.release()
        metrics.in_flight.set(limiter.in_flight)


async def _run_forecast(
    predictor: Predictor,
    history: pd.DataFrame,
    horizon_hours: int,
    api_version: str,
    state: dict[str, Any],
    metrics: ServingMetrics,
    settings: ServingSettings,
) -> tuple[ForecastResult, bool]:
    breaker: CircuitBreaker = state["breaker"]
    metrics.circuit_state.set(_breaker_value(breaker.state))
    if not breaker.allow():
        return (
            _fallback(
                history, horizon_hours, predictor, "circuit_open", api_version, metrics
            ),
            True,
        )
    worker: Future[ForecastResult] | None = None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.request_timeout_s
    try:
        await asyncio.wait_for(
            state["faults"].inject(), timeout=settings.request_timeout_s
        )
        executor: ThreadPoolExecutor | None = state["executor"]
        if executor is None:
            raise RuntimeError("prediction executor is not initialized")
        worker = executor.submit(predictor.predict, history, horizon_hours)
        result = await asyncio.wait_for(
            asyncio.wrap_future(worker), timeout=max(0.0, deadline - loop.time())
        )
    except TimeoutError:
        breaker.record_failure()
        if worker is not None:
            _hold_slot_until_worker_finishes(worker, state["limiter"], metrics)
            return (
                _fallback(
                    history, horizon_hours, predictor, "timeout", api_version, metrics
                ),
                False,
            )
        return (
            _fallback(
                history, horizon_hours, predictor, "timeout", api_version, metrics
            ),
            True,
        )
    except Exception:
        breaker.record_failure()
        LOGGER.exception("primary forecast failed")
        return (
            _fallback(history, horizon_hours, predictor, "error", api_version, metrics),
            True,
        )
    breaker.record_success()
    metrics.predictions.labels(api_version, result.model_version, "primary").inc()
    metrics.circuit_state.set(_breaker_value(breaker.state))
    try:
        feature_values = history[Col.TARGET].iloc[-168:].to_numpy(dtype=np.float64)
        metrics.feature_psi.observe(
            population_stability_index(
                feature_values,
                predictor.bundle.manifest.drift_reference.bin_edges,
                predictor.bundle.manifest.drift_reference.proportions,
            )
        )
    except ValueError:
        LOGGER.warning("could not compute feature PSI")
    return result, True


def _hold_slot_until_worker_finishes(
    worker: Future[ForecastResult], limiter: ConcurrencyLimiter, metrics: ServingMetrics
) -> None:
    """Keep admission occupied while a timed-out worker continues executing.

    ``wait_for`` cannot stop a thread already running model code.  The slot and
    abandoned-work gauge therefore remain occupied until the worker completes,
    preventing timed-out CPU work from exceeding ``max_in_flight``.
    """
    metrics.abandoned_predictions.inc()

    def release(_: Future[ForecastResult]) -> None:
        limiter.release()
        metrics.in_flight.set(limiter.in_flight)
        metrics.abandoned_predictions.dec()

    worker.add_done_callback(release)


def _fallback(
    history: pd.DataFrame,
    horizon_hours: int,
    predictor: Predictor,
    reason: str,
    api_version: str,
    metrics: ServingMetrics,
) -> ForecastResult:
    result = fallback_forecast(
        history, horizon_hours, predictor.bundle.manifest.model_version, reason
    )
    metrics.fallbacks.labels(reason).inc()
    metrics.predictions.labels(api_version, result.model_version, "fallback").inc()
    return result


def _warm_up(predictor: Predictor) -> None:
    history = generate_synthetic_load(
        periods=predictor.bundle.manifest.min_history_hours
    )
    predictor.predict(history, 1)


def _record_model_info(metrics: ServingMetrics, bundle: LoadedBundle) -> None:
    digest = file_sha256(bundle.directory / "manifest.json") or "unknown"
    metrics.model_info.labels(bundle.manifest.model_version, digest).set(1)


def _model_version(state: dict[str, Any]) -> str | None:
    predictor: Predictor | None = state["predictor"]
    return None if predictor is None else predictor.bundle.manifest.model_version


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    return route.path if route is not None else "unmatched"


def _early_route_label(app: FastAPI, request: Request) -> str:
    """Return a bounded label before routing has resolved the request scope."""
    known_paths = {
        route_path
        for route in app.routes
        if isinstance((route_path := getattr(route, "path", None)), str)
    }
    return request.url.path if request.url.path in known_paths else "unmatched"


def _breaker_value(state: str) -> int:
    return {"closed": 0, "half_open": 1, "open": 2}[state]


def _sunset_header(value: str) -> str:
    parsed = date.fromisoformat(value)
    return format_datetime(
        datetime.combine(parsed, datetime_time.min, UTC), usegmt=True
    )
