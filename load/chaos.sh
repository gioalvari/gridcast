#!/bin/sh
set -eu
mkdir -p load/results
docker compose -f deploy/compose.yaml run --rm -e RESULT_NAME=chaos -e DURATION=75s k6 run /load/baseline.js &
k6_pid=$!
sleep 15
docker compose -f deploy/compose.yaml kill predict-canary
sleep 5
docker compose -f deploy/compose.yaml start predict-canary
sleep 15
CANARY_FAULT_LATENCY_MS=750 docker compose -f deploy/compose.yaml up -d --no-deps --force-recreate predict-canary
wait "$k6_pid"
curl -fsS 'http://localhost:9090/api/v1/query?query=sum(gridcast_fallbacks_total)' >load/results/chaos-fallbacks.json
