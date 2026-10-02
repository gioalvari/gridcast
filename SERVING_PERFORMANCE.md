# GridCast serving performance record

**Run date:** 2026-10-02. **Base commit:** `f4a7f18` plus the uncommitted
`feature/production-serving` changes. **Host:** Apple M4 Pro, Rancher Desktop
moby VM (`docker info`: 2 vCPU, 6,212,427,776 bytes memory). The VM is shared by
the load generator, Envoy, LocalStack, Prometheus, Grafana and both prediction
services. Each prediction service is limited to **1 CPU and 1 GiB**. Image
`gridcast-serving:latest`, ID
`sha256:b33d223bda52b3e8efc682543d260c319ba0b5b986aabc8ffa8f37ddddad49c3`.
Stable is bundle `0.1.0`, canary is bundle `0.2.0` (both synthetic).

## Protocol

- `make stack-up`, canary weight set to 0% so **all traffic hits one 1-vCPU
  stable task**.
- Open-loop k6 `constant-arrival-rate` (`load/baseline.js`), 60 s per rate.
  Each request carries 336 hourly observations and asks for a 168-hour
  forecast through Envoy (`POST /v2/forecasts`).
- Both prediction services are restarted and given 20 s before each rate, so
  in-process counters, breaker state and abandoned work do not leak between
  runs.
- Summaries: `load/results/rate-<RATE>.json` (gitignored; regenerate with
  `docker compose -f deploy/compose.yaml run --rm -e RATE=<r> -e DURATION=60s
  -e RESULT_NAME=rate-<r> k6 run /load/baseline.js`).

## Results

| Offered req/s | Achieved req/s | p50 ms | p95 ms | p99 ms | HTTP failed | Schema checks | Dropped iterations | Model fallbacks |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 25 | 25.0 | 9.1 | 17.1 | 62.7 | 0.00% | 100.00% | 0 | none |
| 50 | 50.0 | 9.1 | 43.0 | 176.0 | 0.00% | 100.00% | 0 | none |
| 75 | 74.3 | 413.9 | 1003.6 | 2001.0 | 14.94% | 85.06% | 42 | timeout: 3 |
| 100 | 96.8 | 880.0 | 1999.2 | 2006.1 | 58.51% | 41.49% | 99 | timeout: 34, circuit_open: 103 |

Smoke (`make loadtest-smoke`, 5 req/s for 20 s) passed all k6 thresholds.

**Sustainable capacity for this task size is about 50 req/s** with p95 under
50 ms and no errors. Saturation starts between 50 and 75 req/s.

### What saturation looks like, and what it taught us

Envoy stats after the 75 and 100 req/s runs showed thousands of `504`
(gateway's 1 s route timeout) and `503` (service overload rejection), with
**zero outlier ejections** and a single retry, so the gateway behaved as
designed.

The important finding is *where* the time goes. Only a few requests reached the
model's 0.5 s deadline (`timeout` fallbacks: 3 at 75 req/s). Most failures are
requests that waited on the single event loop for JSON parsing and Pydantic
validation of 336 records *before* reaching the concurrency limiter, then hit
Envoy's 1 s timeout. The limiter protects model execution but not request
parsing, so on 1 vCPU the overload signal arrives too late. At 100 req/s, the
timeouts also opened the circuit breaker, and 103 requests were served by the
seasonal-naive fallback instead of failing. That is the intended degraded
mode.

Follow-ups, in priority order:

1. Admission control before body parsing (uvicorn `--limit-concurrency` or an
   ASGI middleware that counts requests in flight) so overload is rejected in
   microseconds.
2. Make the gateway timeout and the app deadline consistent (the app should
   give up first so it can return a degraded answer instead of a 504).
3. Scale horizontally rather than vertically: one uvicorn worker per vCPU, and
   autoscale on request count per target (already modelled in
   `infra/terraform`).

## Runtime routing and rollout

Envoy uses `runtime_fraction` on a canary route followed by a stable route. It
does not use the older `weighted_clusters.runtime_key_prefix`. Weights change
live through `POST /runtime_modify?routing.canary_percent=<n>`, without a
restart. Shadow traffic is controlled separately by `routing.shadow_percent`.
POST retries are limited to `connect-failure,reset`; 5xx responses are not
retried because the forecast may already have run.

| Scenario | Command | Outcome | Evidence |
|---|---|---|---|
| Healthy canary | `make rollout` | **Promoted** 5→25→50→100%. Canary samples 14/50/146/302, 0 errors, 0 fallbacks, p95 23.3/21.3/41.8/22.2 ms | `artifacts/rollout/good.json` |
| Faulty canary (`CANARY_FAULT_ERROR_RATE=1`) | `make rollout-bad-canary` | **Rolled back at the 5% step**: `error_rate=0.6000>0.0500` over 10 canary requests (6 fallbacks). Weight reset to 0% automatically | `artifacts/rollout/bad.json` |

`make rollout-bad-canary` succeeds only if the rollback was caused by an SLO
breach. A rollback for insufficient traffic fails the target, so CI cannot pass
just because no requests arrived.

Promotion sets canary traffic to 100%. A separate deployment then re-points
stable at the accepted version.

## Cost model (formula, not a price quote)

Assume a Fargate task of **1 vCPU / 1 GiB** in `eu-central-1` running at the
measured sustainable **50 forecasts/s**, with 30% headroom, so 35/s planned.
Let `P_vcpu` be USD per vCPU-hour and `P_gib` be USD per GiB-hour from the AWS
Fargate pricing page, checked on the deployment date.

```
task_hours_per_million = 1,000,000 / (35 * 3600)      = 7.94 h
compute_cost_per_million = 7.94 * (1 * P_vcpu + 1 * P_gib)
```

This excludes the ALB (fixed hourly charge plus LCU), CloudWatch/ADOT, S3,
NAT, data transfer and savings plans. At low traffic, the fixed ALB and the
minimum task count dominate, not the per-forecast cost.

## Limitations

- The local VM is not Fargate. The load generator shares 2 vCPUs with the
  service, so absolute numbers are pessimistic and noisy. Rerun on the
  deployment hardware before planning capacity.
- Bundles and histories are synthetic. LightGBM inference cost does not depend
  on the data source, but the model quality figures do not apply.
- Single gateway, single task per variant, and no soak, stress-ramp or chaos run
  is recorded here (`load/stress.js`, `load/soak.js` and `load/chaos.sh` exist
  but have not been run for this record).
- LocalStack is not S3: IAM, KMS and network latency are not exercised.
