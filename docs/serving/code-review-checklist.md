# Model-serving PR checklist

- [ ] Contract and OpenAPI snapshot changed intentionally; breaking change has version/migration.
- [ ] Input/body limits, deadline, limiter, and cancellation behavior remain bounded.
- [ ] Fallback semantics and degraded response are correct and tested.
- [ ] Metric names/units/labels are bounded; logs contain request/deployment context without data/secrets.
- [ ] Artifact loader verifies integrity and never introduces pickle-like deserialization.
- [ ] Image runs non-root/read-only; IAM/secrets changes are least privilege.
- [ ] Rollout thresholds, rollback path, and runbook changes match behavior.
- [ ] Unit, integration, failure, and load-test evidence cover the change.
- [ ] Dependency, Trivy, OpenAPI, and infrastructure CI results are reviewed.
