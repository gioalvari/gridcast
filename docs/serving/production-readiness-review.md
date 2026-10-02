# GridCast serving production-readiness review

| Area | Status | Evidence / gap |
|---|---|---|
| Native signed-format loading | Done | `src/gridcast/serving/bundle.py`, SHA-256 manifest tests |
| Contract/versioning | Done | `docs/serving/openapi.json`, v1 headers in `app.py`, `openapi-contract` CI |
| Liveness/readiness | Done | `/live`, `/ready`, Docker health check, serving tests |
| Deadline, limiter, breaker | Done | `app.py`, `resilience.py`, resilience tests |
| Safe degradation | Done | `fallback_forecast`, `gridcast_fallbacks_total`, v2 `degraded` response |
| Metrics/logging | Partial | Prometheus and JSON request logs exist; no deployed dashboard/retention policy |
| Local rollout evidence | Done | `deploy/`, `load/`, `rollout-bad-canary`, `serving-e2e` CI |
| Container hardening | Done | non-root serving target, read-only CI run, `/tmp` only |
| Dependency/image audit | Done | CI runs pip-audit and Trivy (HIGH/CRITICAL, no ignore file); runtime image drops pip/setuptools/wheel and applies Debian security updates |
| AWS deployment | Not done | `infra/terraform` is static reference only, never applied |
| AWS progressive rollout | Partial | CodeDeploy/ADOT reference is defined; no account validation or real alarms |
| Authentication/authorization | Not done | Demo exposes no authN/authZ; use API Gateway/ALB auth or mTLS before public use |
| Model provenance | Partial | checksum + immutable registry design; no signed model attestation or production registry |
| Capacity/load SLO | Partial | k6 smoke gates CI; local capacity measured (~50 req/s per 1 vCPU task, `SERVING_PERFORMANCE.md`); pre-parse admission control and Fargate re-measurement still open |
| Incident response | Partial | runbook exists; no declared pager ownership or scheduled exercise |

**Conclusion:** suitable as an engineered local/demo serving layer, not approved
for public or AWS production until authentication, account deployment validation,
operational dashboards/on-call ownership, and representative load evidence exist.
