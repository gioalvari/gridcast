# Model-serving standard

Every team-owned model service must meet these requirements before a production
rollout. A check is evidence, not a promise.

## Contract and compatibility

- Publish an OpenAPI snapshot; make schemas, error codes, and examples explicit.
- Use path versions for incompatible behavior. Deprecate with a published sunset
  date and a successor link; never silently mutate a released response.
- Validate content type, bounded body size, numerical ranges, chronology, and
  tenant/request authorization before expensive work.

## Runtime semantics

- `/live` is process liveness only. `/ready` means artifacts verified, loaded,
  warmed, and able to accept work. `/health` must be non-sensitive and bounded.
- Set an end-to-end request deadline. Bound concurrency with non-queuing
  admission; return retryable overload responses rather than accumulating work.
- Specify a deterministic fallback, its quality limits, when it may run, and how
  consumers discover degradation. Account for work that survives cancellation.

## Observability

- Metric names have a service prefix. Counters end `_total`; durations use
  `_seconds`; labels use bounded enumerations only—never request IDs, model
  hashes, user IDs, timestamps, or exception text.
- Emit request ID, route, status, duration, model version, deployment revision,
  degraded/fallback state, and safe error class in structured logs. Do not log
  raw feature history or secrets.
- Define SLO dashboards and alerts for availability, p95/p99 latency, saturation,
  errors, fallback ratio, and model/data drift.

## Security and release

- Run non-root, read-only root filesystem, minimal image, pinned dependencies,
  and no unsafe model deserialization. Verify artifact integrity and provenance.
- Grant runtime identity only the model prefix and decrypt key it needs. Secrets
  are references, never source-controlled values. Authenticate callers; apply
  WAF/rate limits at the edge for Internet-facing services.
- Use staged rollout with measurable automated rollback. Attach deployment,
  image digest, model version, and schema version to evidence.

## Required evidence and operations

- Attach load-test results for representative, saturation, and failure behavior.
  Thresholds must gate promotion; synthetic demos do not prove production load.
- Maintain an alert runbook and an on-call escalation owner. Rehearse rollback
  and retain a known-good model/image reference.
