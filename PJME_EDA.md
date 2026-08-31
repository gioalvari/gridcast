# PJME exploratory analysis

This report summarizes the reproducible output of:

```bash
make data
make eda
```

## Dataset quality

| Check | Result |
|---|---:|
| Source observations | 145,366 |
| Normalized hourly observations | 145,392 |
| Coverage | 2002-01-01 01:00 to 2018-08-03 00:00 |
| Duplicate timestamps | 4 |
| Missing timestamps | 30 |
| Imputed short gaps | 30 |
| Missing load values | 0 |
| Non-positive load values | 0 |

The duplicate and missing wall-clock hours mostly align with daylight-saving
transitions. GridCast averages duplicate timestamps and linearly interpolates
only gaps of at most three consecutive hours. See
[DATA_SOURCES.md](DATA_SOURCES.md) for the source and timestamp caveats.

## Load distribution

| Statistic | Load (MW) |
|---|---:|
| Mean | 32,078.93 |
| Standard deviation | 6,464.28 |
| Minimum | 14,544.00 |
| 5th percentile | 22,622.00 |
| Median | 31,420.00 |
| 95th percentile | 44,187.00 |
| Maximum | 62,009.00 |

The maximum hourly load occurred at 17:00 on 2 August 2006.

## Observed patterns

- Average demand bottoms near 04:00 at approximately 25.4 GW.
- Demand rises quickly after 06:00 and peaks around 18:00-19:00 at 36.4 GW.
- Weekday evening demand is materially higher than weekend demand.
- Sunday has the lowest overnight and daytime profile.
- Variability increases during afternoon and early-evening hours.
- The daily history shows strong recurring annual seasonality and occasional
  extreme peaks, motivating weather and holiday features.

These observations support daily and weekly seasonal-naive baselines and the
planned lag features at 24, 48, and 168 hours. They also show why a final test
period must include complete seasons rather than a short random sample.
