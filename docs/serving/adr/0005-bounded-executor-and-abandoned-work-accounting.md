# ADR 0005: Bounded executor and abandoned-work accounting

## Context
Python thread cancellation cannot stop native model work after an async timeout.
Allowing unlimited replacement requests would amplify CPU exhaustion.

## Decision
Bound executor workers and admission at `GRIDCAST_MAX_IN_FLIGHT`. A timed-out
worker retains its slot until completion; `gridcast_abandoned_predictions`
reports the work that outlived its request.

## Consequences
Some callers receive quick 503 overload responses instead of unbounded latency.
Capacity estimates must include slow native work, not just completed requests.

## Alternatives considered
Cancelling futures is ineffective once running. An unbounded executor hides
overload until the process becomes unstable.
