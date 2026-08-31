# Operational day-ahead contract

GridCast includes the unique operational-contract work originally explored in
GridLens. It asks not only whether a model predicts load accurately, but whether
every input genuinely existed when the forecast would have been issued and what
happens when a decision is made from that forecast.

## Italian contract

| Field | Value |
|---|---|
| Target | ENTSO-E Actual Total Load, Italy (`10YIT-GRTN-----B`) |
| Forecast origin | 10:00 Europe/Rome on D-1 |
| Delivery horizon | Every real local-market hour of day D |
| Weather vintage | ECMWF IFS run initialized at 00 UTC on D-1 |
| Conservative availability | Run initialization plus six hours |
| Storage and joins | UTC |

Inspect one immutable timing contract:

```bash
uv run gridcast day-ahead contract --delivery-date 2026-08-30
```

Check one archived weather vintage:

```bash
uv run gridcast day-ahead check-weather --delivery-date 2026-08-30
```

## Availability invariants

Every operational feature may carry:

- `valid_at`: when the value applies;
- `available_at`: when the value became usable;
- `forecast_origin`: when the prediction was issued;
- `run_initialized_at`: the immutable numerical-weather-model vintage.

GridCast rejects a feature or weather run when `available_at` is later than the
forecast origin. This prevents reanalysis or later-revised observations from
silently entering a day-ahead simulation.

## Daylight-saving time

Italian delivery dates are represented as real UTC intervals derived from
Europe/Rome boundaries. Spring transition days contain 23 hours, ordinary days
24, and autumn transition days 25. GridCast does not force these dates into an
artificial 24-row representation.

## Archived weather

The archived-weather client requests one explicit Open-Meteo Single Runs model
vintage for eight representative Italian nodes: Bari, Bologna, Cagliari, Milan,
Naples, Palermo, Rome, and Turin. Coverage validation requires every node to
contain every real delivery hour and requires the run to have been available by
forecast issuance.

## Decision-aware evaluation

For stylized procurement, shortage and surplus can have different penalties.
GridCast provides an asymmetric linear cost and the corresponding optimal
forecast quantile:

```text
optimal quantile = shortage cost / (shortage cost + surplus cost)
```

This makes it possible to demonstrate that the lowest-MAE schedule need not be
the lowest-cost schedule. Costs remain synthetic until a defensible market-cost
model and data license are established.

## Status and limitations

- The PJME benchmark remains the validated end-to-end model comparison.
- The Italian track currently provides contracts, leakage guards, ENTSO-E
  ingestion, archived-weather feasibility, and decision metrics.
- ENTSO-E was returning HTTP 503 during the latest live smoke test, so a
  canonical Italian training dataset has not been frozen.
- Exact archived weather vintages cover fewer independent annual cycles than
  the PJME dataset, so future model tuning must remain deliberately small.
