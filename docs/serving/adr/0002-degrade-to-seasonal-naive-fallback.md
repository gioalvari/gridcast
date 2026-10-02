# ADR 0002: Degrade to a seasonal-naive fallback

## Context
A primary prediction timeout, model exception, or open breaker should not turn
known-good client history into an opaque outage.

## Decision
Return a deterministic seasonal-naive forecast when the primary path fails after
request validation. Mark `degraded`, provide `fallback_reason` in v2, increment
`gridcast_fallbacks_total{reason}`, and expose a response header.

## Consequences
Availability is improved while clients can distinguish reduced quality. Fallback
rate is a rollout/CloudWatch rollback signal, not an acceptable steady state.

## Alternatives considered
Fail-fast 5xx sacrifices continuity. Serving cached old predictions risks
staleness and request mismatch. Retrying inside the request violates deadlines.
