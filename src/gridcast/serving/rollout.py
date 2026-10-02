"""Progressive delivery controller backed by Prometheus and Envoy admin APIs.

Promotion changes traffic only.  Re-pointing the stable deployment at the
candidate bundle remains a separate, explicit deployment action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import sleep as system_sleep
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class RolloutPolicy:
    """Traffic steps, observation timing, and canary service-level objectives."""

    steps: tuple[int, ...] = (5, 25, 50, 100)
    bake_seconds: float = 60.0
    min_requests: int = 100
    max_wait_seconds: float = 300.0
    max_error_rate: float = 0.01
    max_p95_latency_ms: float = 250.0
    max_fallback_rate: float = 0.01
    max_prediction_divergence_pct: float | None = None

    def __post_init__(self) -> None:
        """Validate a policy before it can change production traffic."""
        if (
            not self.steps
            or any(step < 0 or step > 100 for step in self.steps)
            or tuple(sorted(self.steps)) != self.steps
            or self.bake_seconds <= 0
            or self.min_requests < 1
            or self.max_wait_seconds < self.bake_seconds
        ):
            raise ValueError("invalid rollout traffic steps or timing")
        thresholds = (self.max_error_rate, self.max_fallback_rate)
        if any(value < 0 or value > 1 for value in thresholds):
            raise ValueError("rate thresholds must be between zero and one")
        if self.max_p95_latency_ms <= 0 or (
            self.max_prediction_divergence_pct is not None
            and self.max_prediction_divergence_pct < 0
        ):
            raise ValueError("invalid rollout SLO threshold")


@dataclass(frozen=True)
class CanaryMetrics:
    """Aggregated canary observations for one evaluation window."""

    requests: float
    errors: float
    fallbacks: float
    p95_latency_ms: float
    canary_mean_prediction: float | None = None
    stable_mean_prediction: float | None = None

    @property
    def error_rate(self) -> float:
        """Return errors plus degraded responses as a fraction of requests."""
        return (self.errors + self.fallbacks) / self.requests

    @property
    def fallback_rate(self) -> float:
        """Return degraded responses as a fraction of requests."""
        return self.fallbacks / self.requests


class MetricsSource(Protocol):
    """Source of aggregate canary observations."""

    def snapshot(self, window_seconds: float) -> CanaryMetrics:
        """Return observations for the trailing window."""


class TrafficController(Protocol):
    """Mechanism for setting the percentage routed to the canary."""

    def set_canary_weight(self, percent: int) -> None:
        """Set the current canary traffic percentage."""


@dataclass(frozen=True)
class RolloutStep:
    """One evaluated traffic step in a rollout report."""

    weight: int
    metrics: CanaryMetrics | None
    passed: bool
    reason: str | None = None


@dataclass(frozen=True)
class RolloutReport:
    """Serializable outcome of a progressive delivery run."""

    outcome: str
    steps: tuple[RolloutStep, ...]
    failing_check: str | None = None
    note: str = (
        "Promotion sets canary traffic to 100%; separately re-point the stable "
        "deployment at the accepted model version."
    )

    def as_dict(self) -> dict[str, Any]:
        """Convert the immutable report to JSON-safe primitives."""
        return asdict(self)


class PrometheusMetricsSource:
    """Read rollout SLO inputs from Prometheus's instant-query API."""

    def __init__(
        self,
        url: str,
        *,
        canary_job: str = "predict-canary",
        stable_job: str = "predict-stable",
        client: httpx.Client | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._canary_job = canary_job
        self._stable_job = stable_job
        self._client = client or httpx.Client(timeout=10.0)

    def snapshot(self, window_seconds: float) -> CanaryMetrics:
        """Query trailing request, error, latency, fallback, and mean values."""
        window = f"{max(1, round(window_seconds))}s"
        canary = f'job="{self._canary_job}"'
        requests = self._query(
            f'sum(increase(gridcast_http_requests_total{{{canary},route=~"/v[12]/forecasts"}}[{window}]))'
        )
        errors = self._query(
            f'sum(increase(gridcast_http_requests_total{{{canary},status=~"5..",route=~"/v[12]/forecasts"}}[{window}]))'
        )
        fallbacks = self._query(
            f"sum(increase(gridcast_fallbacks_total{{{canary}}}[{window}]))"
        )
        latency = self._query(
            "histogram_quantile(0.95, sum by (le) "
            f'(rate(gridcast_http_request_duration_seconds_bucket{{{canary},route=~"/v[12]/forecasts"}}[{window}])))'
        )
        return CanaryMetrics(
            requests=requests,
            errors=errors,
            fallbacks=fallbacks,
            p95_latency_ms=latency * 1000,
            # Serving metrics expose prediction counts, not prediction values.
            # An external outcome/quality metric can supply these optional means.
            canary_mean_prediction=None,
            stable_mean_prediction=None,
        )

    def _query(self, expression: str) -> float:
        response = self._client.get(
            f"{self._url}/api/v1/query", params={"query": expression}
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {payload}")
        result = payload.get("data", {}).get("result", [])
        if not result:
            return 0.0
        return float(result[0]["value"][1])


class EnvoyTrafficController:
    """Update the Envoy runtime key consumed by the first canary route."""

    runtime_key = "routing.canary_percent"

    def __init__(self, url: str, *, client: httpx.Client | None = None) -> None:
        self._url = url.rstrip("/")
        self._client = client or httpx.Client(timeout=10.0)

    def set_canary_weight(self, percent: int) -> None:
        """Set an Envoy integer runtime percentage without restarting Envoy."""
        if not 0 <= percent <= 100:
            raise ValueError("canary traffic must be between zero and 100")
        response = self._client.post(
            f"{self._url}/runtime_modify",
            params={self.runtime_key: str(percent)},
        )
        response.raise_for_status()


def evaluate(policy: RolloutPolicy, metrics: CanaryMetrics) -> str | None:
    """Return the first failed SLO, or ``None`` when all SLOs pass."""
    if metrics.error_rate > policy.max_error_rate:
        return f"error_rate={metrics.error_rate:.4f}>{policy.max_error_rate:.4f}"
    if metrics.p95_latency_ms > policy.max_p95_latency_ms:
        return (
            f"p95_latency_ms={metrics.p95_latency_ms:.2f}>"
            f"{policy.max_p95_latency_ms:.2f}"
        )
    if metrics.fallback_rate > policy.max_fallback_rate:
        return (
            f"fallback_rate={metrics.fallback_rate:.4f}>{policy.max_fallback_rate:.4f}"
        )
    if policy.max_prediction_divergence_pct is not None:
        canary_mean = metrics.canary_mean_prediction
        stable_mean = metrics.stable_mean_prediction
        if canary_mean is None or stable_mean in (None, 0):
            return "prediction_divergence_pct unavailable from serving metrics"
        divergence = abs(canary_mean - stable_mean) / abs(stable_mean) * 100
        if divergence > policy.max_prediction_divergence_pct:
            return (
                f"prediction_divergence_pct={divergence:.2f}>"
                f"{policy.max_prediction_divergence_pct:.2f}"
            )
    return None


def run_rollout(
    policy: RolloutPolicy,
    metrics: MetricsSource,
    traffic: TrafficController,
    *,
    sleep: Callable[[float], None] = system_sleep,
    dry_run: bool = False,
) -> RolloutReport:
    """Progress traffic, enforce SLOs, and immediately roll back on a breach."""
    if dry_run:
        return RolloutReport(
            outcome="dry_run",
            steps=tuple(RolloutStep(weight, None, True) for weight in policy.steps),
        )
    completed: list[RolloutStep] = []
    for weight in policy.steps:
        traffic.set_canary_weight(weight)
        waited = 0.0
        while True:
            sleep(policy.bake_seconds)
            waited += policy.bake_seconds
            observed = metrics.snapshot(policy.bake_seconds)
            if observed.requests >= policy.min_requests:
                break
            if waited >= policy.max_wait_seconds:
                traffic.set_canary_weight(0)
                reason = (
                    f"insufficient_requests={observed.requests:.0f}<"
                    f"{policy.min_requests} after {waited:.0f}s"
                )
                completed.append(RolloutStep(weight, observed, False, reason))
                return RolloutReport("rolled_back", tuple(completed), reason)
        failing_check = evaluate(policy, observed)
        passed = failing_check is None
        completed.append(RolloutStep(weight, observed, passed, failing_check))
        if failing_check is not None:
            traffic.set_canary_weight(0)
            return RolloutReport("rolled_back", tuple(completed), failing_check)
    return RolloutReport("promoted", tuple(completed))


def policy_from_dict(raw: dict[str, Any]) -> RolloutPolicy:
    """Construct a validated policy from a JSON configuration document."""
    raw = raw.copy()
    if "steps" in raw:
        raw["steps"] = tuple(raw["steps"])
    return RolloutPolicy(**raw)
