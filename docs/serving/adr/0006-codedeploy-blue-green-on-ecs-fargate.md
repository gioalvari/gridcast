# ADR 0006: CodeDeploy blue/green on ECS Fargate

## Context
The AWS reference needs independent task-set validation and rollback alarms,
without introducing a Kubernetes control plane.

## Decision
Use ECS Fargate with a CodeDeploy controller, two ALB target groups, test and
production HTTPS listeners, and `ECSCanary10Percent5Minutes`. Alarms for target
5xx, p95 latency, and ADOT-exported fallback rate stop and roll back delivery.

## Consequences
Deployments require an AppSpec/task-definition revision and CodeDeploy-aware
automation. Two task sets temporarily increase rollout capacity/cost.

## Alternatives considered
ECS rolling deployment lacks the same test listener and traffic-shift controls.
Kubernetes/Argo Rollouts adds operational surface not justified here.
