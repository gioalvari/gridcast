# GridCast serving performance record

**Run date:** 2026-10-02. **Host:** Apple M4 Pro, Rancher Desktop moby VM
(`docker info`: 2 vCPU, 6,212,427,776 bytes of memory). The VM is shared by the
load generator, Envoy, LocalStack, Prometheus, Grafana and both prediction
services. Each prediction service is limited to **1 CPU and 1 GiB**. Bundles:
stable `0.1.0` and canary `0.2.0`, both synthetic.

Images compared:

- **without admission control**: the serving code at commit `c0b11a1`, image
  `sha256:d3ae6184…`;
- **with admission control**: the same image and dependencies plus the outer
  ASGI admission limit and pydantic-core JSON parsing, image
  `sha256:2015dc78…`.

## Protocol

- `make stack-up`, then the canary weight is set to 0%, so **all traffic hits
  one 1-vCPU stable task**.
- Open-loop k6 `constant-arrival-rate` (`load/baseline.js`), 60 s per run.
  Each request carries 336 hourly observations and asks for a 168-hour
  forecast through Envoy (`POST /v2/forecasts`).
- Before each run the prediction services are recreated with the image and
  limits under test and given 20 s, so counters, breaker state and abandoned
  work do not leak between runs.
- Image A/B runs are **interleaved** (old→new, then new→old) so that drift in
  host load affects both versions equally.
- Summaries: `load/results/<name>.json` (gitignored). `ok_req_duration` is a k6
  trend restricted to HTTP 200 responses.

## Results

### Capacity: the code change does not move it

| Rate | Version | Run | p50 ms | p95 ms | p99 ms | HTTP failed | Dropped |
|---:|---|---:|---:|---:|---:|---:|---:|
| 75 | without admission | 1 / 2 | 8.1 / 8.4 | 21.2 / 27.4 | 67.1 / 85.0 | 0.00% / 0.00% | 0 / 0 |
| 75 | with admission 64/32 | 1 / 2 | 7.9 / 8.8 | 19.5 / 36.5 | 60.9 / 97.3 | 0.00% / 0.00% | 0 / 0 |
| 100 | without admission | 1 / 2 | 15.8 / 251.7 | 582.4 / 509.0 | 664.9 / 626.0 | 1.40% / 1.02% | 20 / 16 |
| 100 | with admission 64/32 | 1 / 2 | 401.5 / 399.1 | 617.2 / 626.7 | 726.6 / 701.8 | 3.33% / 3.19% | 25 / 21 |

**One 1-vCPU task sustains 75 req/s with no errors and p95 under 40 ms.
Saturation is between 75 and 100 req/s.** Both versions behave the same up to
saturation.

An in-process micro-benchmark agrees: one sequential request costs a median of
6.2 ms without the change and 7.5 ms with it. The parts of a 336-record request
(median, 35 samples) are:

| Step | ms |
|---|---:|
| `json.loads` | 0.08 |
| dict → `ForecastRequest` | 0.19 |
| `ForecastRequest.model_validate_json` | 0.18 |
| DataFrame + `validate_history` | 0.58 |
| `Predictor.predict` (168 h) | 2.28 |
| response model + JSON | 0.30 |

Parsing is not the bottleneck; the model and DataFrame work are.

### Admission control only helps when it is sized to capacity

Same image (with admission), changing only `GRIDCAST_MAX_ADMITTED` /
`GRIDCAST_MAX_IN_FLIGHT`:

| Rate | Limits | Run | p50 ms | p95 ms | p99 ms | HTTP failed | Successful-request p95 ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 75 | 16 / 8 | 1 | 7.3 | 19.8 | 81.1 | 0.22% | 19.5 |
| 75 | 8 / 4 | 1 | 6.9 | 13.5 | 54.1 | 0.62% | 12.8 |
| 100 | none (no admission) | 1 / 2 | 15.8 / 251.7 | 582.4 / 509.0 | 664.9 / 626.0 | 1.40% / 1.02% | 583.3 / 510.5 |
| 100 | 64 / 32 | 1 / 2 | 401.5 / 399.1 | 617.2 / 626.7 | 726.6 / 701.8 | 3.33% / 3.19% | 618.6 / 629.3 |
| 100 | **16 / 8** | 1 / 2 | 6.5 / 7.3 | **26.6 / 46.8** | 71.1 / 99.7 | **0.08% / 0.13%** | 26.6 / 46.4 |
| 100 | 8 / 4 | 1 / 2 | 6.7 / 6.7 | 19.4 / 17.9 | 53.3 / 60.9 | 0.98% / 1.12% | 19.2 / 17.4 |
| 150 | 64 / 32 | 1 | 440.7 | 579.1 | 634.1 | 27.73% | 590.9 |
| 150 | 16 / 8 | 1 | 69.6 | 169.6 | 205.2 | 34.99% | 180.7 |
| 150 | 8 / 4 | 1 | 24.1 | 69.8 | 100.8 | 24.62% | 79.5 |

What this shows:

- **A generous limit is worse than no limit.** With 64 admitted requests and
  ~10 ms of CPU per request on one core, a request can queue for ~640 ms
  (Little's law: queue length ≈ throughput × waiting time). That is exactly the
  ~620 ms p95 measured.
- **A limit sized to capacity protects latency.** At 100 req/s, just past
  saturation, 16/8 keeps p95 at 27–47 ms and rejects 0.1% of requests,
  compared with ~550 ms without admission control.
- **A limit that is too tight sheds load too early.** 8/4 gives the lowest
  latency but rejects about 1% of traffic at 100 req/s and 0.6% at 75 req/s,
  where the service is not yet saturated.
- **Far beyond capacity, no limit creates CPU.** At 150 req/s every
  configuration fails 25–35% of requests. Small limits turn those failures into
  fast 503s, so the requests that are served stay fast.

Deployments therefore use **16 admitted / 8 executing per 1-vCPU task**
(`deploy/compose.yaml`, `infra/terraform/main.tf`); the sizing rule is in
ADR 0005. All configurations were measured once or twice on a shared VM, so
treat differences of a few milliseconds as noise.

### Earlier measurement, superseded

The first record of this report measured 15% failures at 75 req/s and blamed
JSON parsing on the event loop. The interleaved A/B run above contradicts both
claims: the same code serves 75 req/s without errors, and the micro-benchmark
puts parsing at well under 1 ms. The earlier run was most likely taken on a
loaded host. It is kept here only to explain the change.

### A stack defect found while measuring

During the A/B runs every request failed in under 1 ms. Envoy's
c-ares DNS resolver had stopped resolving the compose service names, so both
clusters had zero members, even though `getent hosts predict-stable` worked
inside the Envoy container. The clusters now use the `getaddrinfo` resolver,
IPv4 only, with a 2 s refresh (`deploy/envoy/envoy.yaml`). After a stable
container is recreated, the gateway routes to it again within seconds.

## Runtime routing and rollout

Envoy uses `runtime_fraction` on a canary route followed by a stable route. It
does not use the older `weighted_clusters.runtime_key_prefix`. Weights change
live through `POST /runtime_modify?routing.canary_percent=<n>`, without a
restart. Shadow traffic is controlled separately by `routing.shadow_percent`.
POST retries are limited to `connect-failure,reset`; 5xx responses are not
retried because the forecast may already have run.

| Scenario | Command | Outcome | Evidence |
|---|---|---|---|
| Healthy canary | `make rollout` | **Promoted** 5→25→50→100%. Canary samples 14/56/112/86, 0 errors, 0 fallbacks, p95 9.8/19.7/11.0/16.9 ms | `artifacts/rollout/good.json` |
| Faulty canary (`CANARY_FAULT_ERROR_RATE=1`) | `make rollout-bad-canary` | **Rolled back at the 5% step**: `error_rate=1.0000>0.0500` over 12 canary requests. Weight reset to 0% automatically | `artifacts/rollout/bad.json` |

`make rollout-bad-canary` succeeds only if the rollback was caused by an SLO
breach. A rollback for insufficient traffic fails the target, so CI cannot pass
just because no requests arrived.

Promotion sets canary traffic to 100%. A separate deployment then re-points
stable at the accepted version.

Both were re-run after the Envoy DNS change with the 16/8 limits, along with
`make loadtest-smoke` (101 requests, 0% failed, p95 14.8 ms).

## Cost model (formula, not a price quote)

Assume a Fargate task of **1 vCPU / 1 GiB** in `eu-central-1` running at the
measured error-free **75 forecasts/s**, with 30% headroom, so 52.5/s planned.
Let `P_vcpu` be USD per vCPU-hour and `P_gib` be USD per GiB-hour from the AWS
Fargate pricing page, checked on the deployment date.

```
task_hours_per_million   = 1,000,000 / (52.5 * 3600) = 5.29 h
compute_cost_per_million = 5.29 * (1 * P_vcpu + 1 * P_gib)
```

This excludes the ALB (fixed hourly charge plus LCU), CloudWatch/ADOT, S3,
NAT, data transfer and savings plans. At low traffic the fixed ALB cost and the
minimum task count dominate, not the per-forecast cost.

## Limitations

- The local VM is not Fargate. The load generator shares 2 vCPUs with the
  service, so absolute numbers are pessimistic and noisy, and each
  configuration was run once or twice. Rerun on the deployment hardware before
  planning capacity.
- Bundles and histories are synthetic. LightGBM inference cost does not depend
  on the data source, but the model quality figures do not apply.
- Single gateway and single task per variant. No soak, stress-ramp or chaos run
  is recorded here (`load/stress.js`, `load/soak.js` and `load/chaos.sh` exist
  but have not been run for this record).
- LocalStack is not S3: IAM, KMS and network latency are not exercised.
