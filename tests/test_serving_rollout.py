import json

import httpx
import pytest

from gridcast.serving.rollout import (
    CanaryMetrics,
    EnvoyTrafficController,
    PrometheusMetricsSource,
    RolloutPolicy,
    policy_from_dict,
    run_rollout,
)


class FakeMetrics:
    def __init__(self, observations: list[CanaryMetrics]) -> None:
        self.observations = iter(observations)

    def snapshot(self, _: float) -> CanaryMetrics:
        return next(self.observations)


class FakeTraffic:
    def __init__(self) -> None:
        self.weights: list[int] = []

    def set_canary_weight(self, percent: int) -> None:
        self.weights.append(percent)


def sample(**changes: float) -> CanaryMetrics:
    values = {"requests": 10, "errors": 0, "fallbacks": 0, "p95_latency_ms": 10}
    values.update(changes)
    return CanaryMetrics(**values)


def policy(**changes: object) -> RolloutPolicy:
    values: dict[str, object] = {
        "steps": (5, 100),
        "bake_seconds": 1,
        "min_requests": 10,
        "max_wait_seconds": 2,
        "max_error_rate": 0.05,
        "max_p95_latency_ms": 50,
        "max_fallback_rate": 0.05,
    }
    values.update(changes)
    return RolloutPolicy(**values)  # type: ignore[arg-type]


def test_run_rollout_promotes_happy_canary() -> None:
    traffic = FakeTraffic()
    report = run_rollout(
        policy(), FakeMetrics([sample(), sample()]), traffic, sleep=lambda _: None
    )
    assert report.outcome == "promoted"
    assert traffic.weights == [5, 100]


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (sample(errors=1), "error_rate"),
        (sample(p95_latency_ms=51), "p95_latency_ms"),
        (sample(fallbacks=1), "fallback_rate"),
    ],
)
def test_run_rollout_rolls_back_slo_breaches(
    metrics: CanaryMetrics, expected: str
) -> None:
    traffic = FakeTraffic()
    configured = policy(max_error_rate=0.2) if expected == "fallback_rate" else policy()
    report = run_rollout(
        configured, FakeMetrics([metrics]), traffic, sleep=lambda _: None
    )
    assert report.outcome == "rolled_back"
    assert expected in (report.failing_check or "")
    assert traffic.weights == [5, 0]


def test_run_rollout_fails_after_insufficient_traffic() -> None:
    traffic = FakeTraffic()
    report = run_rollout(
        policy(min_requests=11),
        FakeMetrics([sample(), sample()]),
        traffic,
        sleep=lambda _: None,
    )
    assert report.outcome == "rolled_back"
    assert "insufficient_requests" in (report.failing_check or "")
    assert traffic.weights == [5, 0]


def test_run_rollout_dry_run_does_not_change_traffic() -> None:
    traffic = FakeTraffic()
    report = run_rollout(policy(), FakeMetrics([]), traffic, dry_run=True)
    assert report.outcome == "dry_run"
    assert traffic.weights == []


def test_prometheus_client_builds_queries_and_parses_values() -> None:
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(str(request.url))
        return httpx.Response(
            200, json={"status": "success", "data": {"result": [{"value": [0, "2"]}]}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = PrometheusMetricsSource("http://prom", client=client).snapshot(10)
    assert result.requests == 2
    assert result.p95_latency_ms == 2000
    assert any("gridcast_fallbacks_total" in query for query in queries)
    assert any("job%3D%22predict-canary%22" in query for query in queries)


def test_envoy_runtime_modify_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    EnvoyTrafficController("http://envoy:9901", client=client).set_canary_weight(25)
    assert requests[0].url.path == "/runtime_modify"
    assert dict(requests[0].url.params) == {
        "routing.canary_percent": "25",
    }


def test_report_is_json_serializable() -> None:
    report = run_rollout(
        policy(), FakeMetrics([sample(), sample()]), FakeTraffic(), sleep=lambda _: None
    )
    assert json.loads(json.dumps(report.as_dict()))["outcome"] == "promoted"


def test_policy_validation_and_conversion() -> None:
    assert policy_from_dict({"steps": [10, 100]}).steps == (10, 100)
    with pytest.raises(ValueError, match="invalid rollout"):
        RolloutPolicy(steps=(100, 5))
    with pytest.raises(ValueError, match="rate thresholds"):
        RolloutPolicy(max_error_rate=2)


def test_prediction_divergence_rolls_back() -> None:
    traffic = FakeTraffic()
    report = run_rollout(
        policy(max_prediction_divergence_pct=10),
        FakeMetrics([sample(canary_mean_prediction=12, stable_mean_prediction=10)]),
        traffic,
        sleep=lambda _: None,
    )
    assert "prediction_divergence_pct" in (report.failing_check or "")


def test_prometheus_empty_and_failed_responses() -> None:
    empty = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"status": "success", "data": {"result": []}}
            )
        )
    )
    assert (
        PrometheusMetricsSource("http://prom", client=empty).snapshot(1).requests == 0
    )
    failed = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"status": "error", "data": {}})
        )
    )
    with pytest.raises(RuntimeError, match="Prometheus query failed"):
        PrometheusMetricsSource("http://prom", client=failed).snapshot(1)


def test_envoy_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match="between zero"):
        EnvoyTrafficController("http://envoy").set_canary_weight(101)
