"""Per-application Prometheus metrics for online serving."""

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


@dataclass(frozen=True)
class ServingMetrics:
    """All metrics attached to exactly one FastAPI serving application."""

    registry: CollectorRegistry
    http_requests: Counter
    http_duration: Histogram
    predictions: Counter
    fallbacks: Counter
    circuit_state: Gauge
    in_flight: Gauge
    model_info: Gauge
    feature_psi: Histogram
    overload_rejections: Counter
    abandoned_predictions: Gauge


def create_metrics() -> ServingMetrics:
    """Create isolated Prometheus collectors without global registry state."""
    registry = CollectorRegistry()
    return ServingMetrics(
        registry=registry,
        http_requests=Counter(
            "gridcast_http_requests_total",
            "HTTP requests completed.",
            ["route", "method", "status"],
            registry=registry,
        ),
        http_duration=Histogram(
            "gridcast_http_request_duration_seconds",
            "HTTP request duration.",
            ["route"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=registry,
        ),
        predictions=Counter(
            "gridcast_predictions_total",
            "Forecast requests served.",
            ["api_version", "model_version", "path"],
            registry=registry,
        ),
        fallbacks=Counter(
            "gridcast_fallbacks_total",
            "Fallback forecasts served.",
            ["reason"],
            registry=registry,
        ),
        circuit_state=Gauge(
            "gridcast_circuit_breaker_state",
            "Circuit state: closed=0, half_open=1, open=2.",
            registry=registry,
        ),
        in_flight=Gauge(
            "gridcast_in_flight_requests",
            "Admitted forecast requests.",
            registry=registry,
        ),
        model_info=Gauge(
            "gridcast_model_info",
            "Loaded model identity.",
            ["model_version", "bundle_sha256"],
            registry=registry,
        ),
        feature_psi=Histogram(
            "gridcast_feature_psi",
            "Population stability index for lag_168h.",
            buckets=(0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0),
            registry=registry,
        ),
        overload_rejections=Counter(
            "gridcast_overload_rejections_total",
            "Forecast requests rejected because all slots are occupied.",
            registry=registry,
        ),
        abandoned_predictions=Gauge(
            "gridcast_abandoned_predictions",
            "Timed-out primary predictions still running in the worker pool.",
            registry=registry,
        ),
    )
