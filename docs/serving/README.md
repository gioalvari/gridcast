# GridCast production serving

This directory records the serving contract and the operational decisions around
`gridcast.serving.app:create_app`. The generated API contract is
[`openapi.json`](openapi.json); CI rejects unreviewed changes and breaking PR
changes relative to the base branch.

```mermaid
flowchart LR
 C[Client] --> E[Envoy: stable/canary/shadow]
 E --> A[FastAPI serving]
 A --> V[Validate body + hourly history]
 V --> L[Bounded limiter]
 L --> B[Circuit breaker]
 B --> P[Primary LightGBM bundle + deadline]
 P -->|failure, timeout, open| F[Seasonal-naive fallback]
 P --> R[Registry: file:// or s3://]
 F --> M[Prometheus metrics + structured logs]
 P --> M
 M --> O[Prometheus SLO checks]
 O --> W[Rollout controller adjusts/rolls back weights]
```

`/live` answers process liveness. `/ready` answers only after the native bundle
has resolved, verified, loaded, and warmed. `/health` is a bounded operational
state view; `/metrics` exposes only `gridcast_` Prometheus collectors.

The deprecated `/v1/forecasts` endpoint supplies `Deprecation`, `Sunset`, and
successor `Link` headers. New callers must use `/v2/forecasts`, which returns
point and calibrated interval values and explicitly says whether it degraded.

## Reading order

- [Serving standard](serving-standard.md): reusable team requirements.
- [Production readiness review](production-readiness-review.md): current facts.
- [Runbook](runbook.md) and [threat model](threat-model.md): operate safely.
- [ADRs](adr/): the decisions that constrain compatible changes.
- [Load-test plan template](load-test-plan-template.md) and
  [review checklist](code-review-checklist.md): evidence expected from PRs.
