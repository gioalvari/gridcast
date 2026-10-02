# Serving threat model

Scope: caller → Envoy/ALB → FastAPI → local/S3 model registry → metrics/logs.
The demo has no authentication and must not be exposed as a public production API.

| STRIDE | Threat | Current control | Required production control |
|---|---|---|---|
| Spoofing | Unauthenticated caller invokes forecasts | None in demo | API Gateway/ALB authentication or mTLS, scoped client identity |
| Tampering | Model bundle or transit is altered | SHA-256 manifest; TLS-only, versioned immutable S3/ECR reference | Signed provenance, S3 Object Lock where retention requires it, digest-pinned image |
| Repudiation | Caller/deployment denies an action | X-Request-ID and structured request logs | Retained centralized audit logs, authenticated principal/deployment identity |
| Information disclosure | History/errors reveal sensitive data | bounded generic errors; no raw history logging | classify data, TLS, redacted logs, least-privilege log access |
| Denial of service | Huge body, concurrency exhaustion, slow native work | body limit, non-queuing limiter, deadlines, breaker, abandoned-work accounting | edge rate limits, WAF, autoscaling, load-tested limits |
| Elevation of privilege | Container/model gains host or AWS powers | non-root, read-only rootfs, no pickle, prefix-only task IAM | admission/image policy, scoped OIDC deploy role, secret rotation |

Trust boundaries are the public edge, container runtime, model registry, and
observability backend. Treat model artifacts as executable-adjacent supply-chain
inputs even though native LightGBM avoids Python pickle execution.
