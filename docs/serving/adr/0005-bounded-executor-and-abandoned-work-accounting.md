# ADR 0005: Bounded executor and abandoned-work accounting

## Context
Python thread cancellation cannot stop native model work after an async timeout.
Allowing unlimited replacement requests would amplify CPU exhaustion.

## Decision
Bound executor workers and admission at `GRIDCAST_MAX_IN_FLIGHT`. A timed-out
worker retains its slot until completion; `gridcast_abandoned_predictions`
reports the work that outlived its request.

An outer ASGI admission limit (`GRIDCAST_MAX_ADMITTED`, at least the execution
limit) rejects excess forecast requests before their bodies are read or parsed.
The app deadline is 0.5 s and deliberately remains below Envoy's 1 s route
timeout, so the app can return a degraded result before the gateway gives up.

## Consequences
Some callers receive quick 503 overload responses instead of unbounded latency.
Capacity estimates must include slow native work, not just completed requests.

The limits only protect latency when they are sized to capacity. By Little's
law, admitted requests ≈ throughput × acceptable queueing time. One 1-vCPU task
serves roughly 100 forecasts/s at ~10 ms CPU each, so a 150 ms budget gives
about 16 admitted requests. Measured at 100 req/s
(`SERVING_PERFORMANCE.md`): limits of 64/32 gave p95 ≈ 620 ms, which is worse
than no limit at all; 16/8 gave p95 27–47 ms with 0.1% rejections; 8/4 gave
p95 ≈ 19 ms but rejected ~1% of traffic, even at 75 req/s. Deployments therefore
set `GRIDCAST_MAX_IN_FLIGHT=8` and `GRIDCAST_MAX_ADMITTED=16` per vCPU
explicitly, and re-derive them whenever model cost or task size changes. The
code defaults (32/64) are deliberately generous and are not meant for
production.

## Alternatives considered
Cancelling futures is ineffective once running. An unbounded executor hides
overload until the process becomes unstable.
