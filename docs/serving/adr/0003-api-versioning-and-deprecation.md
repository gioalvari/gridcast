# ADR 0003: Path API versioning and deprecation

## Context
v2 adds calibrated intervals and fallback detail, while existing v1 point-only
clients need a bounded migration period.

## Decision
Use `/v1/forecasts` and `/v2/forecasts`. Keep v1 response-compatible, send RFC
deprecation metadata (`Deprecation`, `Sunset`, successor `Link`), and commit the
generated OpenAPI snapshot.

## Consequences
Breaking shape changes require a new path version and migration communication.
Schema regeneration plus oasdiff protects consumers in pull requests.

## Alternatives considered
Header/media-type negotiation obscures contracts in tooling. Silent v1 mutation
was rejected because operational clients cannot safely infer changed semantics.
