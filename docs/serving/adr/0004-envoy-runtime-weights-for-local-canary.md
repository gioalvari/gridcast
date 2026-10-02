# ADR 0004: Envoy runtime weights for local canary

## Context
The local compose stack needs realistic progressive delivery without cloud
resources or rebuilding routing configuration for each rollout step.

## Decision
Use Envoy runtime weights for stable/canary traffic and optional shadow traffic.
The rollout controller queries Prometheus evidence before changing weights.

## Consequences
Canary simulation is reproducible in CI and a failing canary can be exercised.
It is intentionally not an AWS deployment mechanism; ECS uses CodeDeploy
blue/green canaries instead.

## Alternatives considered
Docker Compose service recreation is not weighted routing. A Kubernetes mesh is
too large a dependency for local serving proof.
