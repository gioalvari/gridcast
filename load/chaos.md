# Chaos exercise

`make chaos` starts a constant-rate k6 run, kills and restarts `predict-canary`,
then recreates it with `CANARY_FAULT_LATENCY_MS=750`. Results are written to
`load/results/chaos.json` and fallback counters to `load/results/chaos-fallbacks.json`.
This is a simple availability exercise, not a proof of fault tolerance: the current
fault injector degrades requests inside the canary; it does not emulate network loss.
