# GridCast serving runbook

Use request ID, model version, image digest, deployment revision, and rollout
report together. Do not paste client histories or secrets into tickets.

| Alert | Meaning / panel | Diagnose | Mitigate | Escalate |
|---|---|---|---|---|
| Target 5xx | ALB `HTTPCode_Target_5XX_Count` | Compare target health, `/ready`, task logs, recent image/model changes | Stop CodeDeploy deployment; restore prior task definition/image | platform + model owner if persists 10 min |
| p95 latency | ALB `TargetResponseTime p95` > threshold | Check `gridcast_http_request_duration_seconds`, CPU, abandoned workers, request size | Reduce canary to 0, scale service, enforce edge limit | platform if saturation; model owner if predictor regression |
| Fallback rate | EMF `gridcast_fallbacks_total / gridcast_predictions_total` | Group by `reason`; inspect breaker state and registry/model load logs | Roll back canary; use stable model; do not suppress metric | model owner immediately if primary error/timeout |
| Overload | `gridcast_overload_rejections_total` rises | Check in-flight and abandoned gauges, CPU, request concurrency | Scale out within cap; reduce client concurrency; reject oversized traffic | platform if sustained |
| Not ready | ALB unhealthy targets / `/ready` 503 | Verify S3 URI/prefix, KMS decrypt, checksum/manifest logs, ephemeral `/tmp` | Roll back artifact/task; fix IAM only through review | platform + security for IAM/KMS |

## Commands and guardrails

**Local canary rollback:** the controller returns exit code 2 on a failed canary;
force zero canary only through the documented Envoy runtime control in the local
stack, then record `artifacts/rollout/bad.json`. Run `make stack-down` after CI
or a rehearsal.

**AWS reference:** CodeDeploy alarms auto-stop/roll back. An authorized operator
may stop the active deployment in the AWS console/CLI and select the last known
good task definition. Scale the ECS service via approved autoscaling bounds;
never broaden task IAM or disable checksum verification as an incident shortcut.

Escalate security events (unexpected model object, IAM denial pattern, suspected
credential exposure) to security immediately; preserve logs and image digest.
