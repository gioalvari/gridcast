# Model-serving load-test plan template

## Change and objective

- Change/image digest/model version:
- Contract version and representative request fixtures:
- Expected peak RPS, concurrency, and p95/p99 SLO:
- Environment and resource limits:

## Scenarios

| Scenario | Traffic | Required assertion |
|---|---|---|
| Baseline | normal valid requests | p95/p99, error and fallback limits |
| Peak | expected peak with realistic history sizes | no queue growth; autoscale/limit behavior |
| Saturation | beyond capacity | quick bounded 503s, no crash or memory leak |
| Primary fault | timeout/error injection | explicit fallback and breaker metrics |
| Canary | weighted candidate traffic | SLO comparison and rollback threshold |

Record k6 script revision, duration, seed, container/image digest, model manifest
hash, raw report location, and charts. State whether fixtures are synthetic or
production-representative. Promotion requires threshold-gated evidence, not an
interactive manual curl.
