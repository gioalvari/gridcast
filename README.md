# GridCast

[![Quality](https://github.com/gioalvari/gridcast/actions/workflows/quality.yml/badge.svg)](https://github.com/gioalvari/gridcast/actions/workflows/quality.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-17242B.svg)](LICENSE)

GridCast is a reproducible energy forecasting workbench focused on honest
temporal validation, uncertainty, and operationally meaningful evaluation.
It now also contains the former GridLens operational day-ahead work, so there is
one canonical energy-forecasting repository rather than two overlapping ones.

![Frozen-test model comparison](docs/assets/frozen-test-mae.svg)

On the frozen 52-week test, the combined LightGBM model reduces MAE by **19.07%**
relative to the weekly seasonal-naive baseline. The gain is **3.07%** relative
to the stronger daily seasonal-naive baseline and **2.38%** relative to base
LightGBM. The daily baseline still wins the most individual weeks, so the result
is reported as an aggregate improvement rather than universal superiority.

## Why this project

Forecasting examples often use random train/test splits, which leak future
information into model evaluation. GridCast uses strict chronological
walk-forward validation, explicit daily and weekly baselines, a frozen final
year, and validation-only conformal calibration.

## Quick start

Requirements: Python 3.11+ and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
make install
make full
make demo
```

The demo writes the following reproducible artifacts to `artifacts/demo/`:

- `load.csv`: generated hourly load data
- `forecasts.csv`: timestamped out-of-sample predictions by fold
- `metrics.csv`: MAE, RMSE, and MASE for each fold
- `summary.json`: aggregate metrics and experiment configuration

Run a customized backtest:

```bash
uv run gridcast demo \
  --periods 3360 \
  --initial-window 2016 \
  --horizon 168 \
  --step 168 \
  --seed 42 \
  --output-dir artifacts/demo
```

## Real PJM data

Download and normalize the public PJME dataset, then create the EDA report:

```bash
make data
make weather
make eda
```

The data command:

- downloads `PJME_hourly.csv` from the CC0 Kaggle dataset;
- records duplicates and missing hours in a JSON quality report;
- averages duplicate wall-clock timestamps;
- interpolates only short gaps of at most three hours;
- validates positive loads and a complete hourly frequency;
- writes a Snappy-compressed Parquet cache under the ignored `data/` folder.

The EDA command produces descriptive statistics, daily averages, hourly and
weekly profiles, and two publication-ready PNG charts in `artifacts/eda/`.
See [PJME_EDA.md](PJME_EDA.md) for the current findings and
[DATA_SOURCES.md](DATA_SOURCES.md) for provenance and timestamp caveats.

## ENTSO-E data

GridCast can also ingest authenticated actual total load from the ENTSO-E
Transparency Platform. The token is never stored in the repository.

```bash
uv run gridcast data entsoe --start 2024-01-01 --end 2025-01-01
```

Locally, the command reads the token from the macOS Keychain service
`gridcast-entsoe-api-token`; in CI it reads the `ENTSOE_API_TOKEN` secret. A
manual GitHub Actions workflow is included for bounded smoke tests. See
[DATA_SOURCES.md](DATA_SOURCES.md) for scope, licensing, and security details.

## Operational day-ahead contracts

The Italian track makes data availability explicit: each forecast has a 10:00
Europe/Rome D-1 origin, a DST-safe 23/24/25-hour delivery day, and one immutable
ECMWF run that must have been published before issuance.

```bash
uv run gridcast day-ahead contract --delivery-date 2026-08-30
uv run gridcast day-ahead check-weather --delivery-date 2026-08-30
```

It also adds asymmetric shortage/surplus decision costs and their cost-optimal
forecast quantile. See
[Operational day-ahead contract](docs/day-ahead-contract.md) and
[GridLens consolidation](docs/gridlens-migration.md).

## Real-data benchmark

Run the chronological PJME benchmark after preparing the data:

```bash
make benchmark
```

The benchmark compares persistence, daily seasonal naive, weekly seasonal
naive, LightGBM, and separate holiday, weather, and combined ablations. Target
and observed-weather features are delayed by at least 168 hours; federal
holidays are known in advance; and temperature climatology uses prior years
only. The entire weekly horizon is therefore generated without reading realized
values inside that horizon. The final 52 weeks are frozen as a test set; the
preceding 12 weeks form the development validation period.

Outputs under `artifacts/benchmark/` include the leaderboard, fold-level
metrics, timestamped predictions, configuration metadata, and diagnostic
charts. See [PJME_BENCHMARK.md](PJME_BENCHMARK.md) for results and limitations.

### Frozen-test benchmark

| Model | MAE (MW) | RMSE (MW) | MASE | MAE vs weekly naive |
|---|---:|---:|---:|---:|
| LightGBM + weather + holidays | **2,901.57** | **3,934.23** | **0.962** | **+19.07%** |
| LightGBM + weather | 2,913.06 | 3,964.45 | 0.966 | +18.74% |
| LightGBM base | 2,972.24 | 4,005.40 | 0.986 | +17.09% |
| Daily seasonal naive | 2,993.52 | 3,998.30 | 0.992 | +16.50% |
| Weekly seasonal naive | 3,585.07 | 4,856.16 | 1.189 | baseline |

## Probabilistic forecasting

Train P10, P50, and P90 models and calibrate the 80% interval using only the
validation folds:

```bash
make probabilistic
```

The raw quantile interval covers 57.59% of frozen-test observations. A
validation-only split-conformal correction increases coverage to 80.04%, close
to the 80% target, while increasing mean interval width from 5,842 to 8,985 MW.
An hour-conditional variant reduces mean hourly coverage error by 28.5%, from
2.40 to 1.72 percentage points, at the cost of 4.2% wider intervals and 81.40%
aggregate coverage.
A 12-week causal rolling experiment reaches 79.52% coverage but makes intervals
2.64% wider and does not improve week-level calibration. It is retained as a
documented negative result rather than replacing the static global method.
See [PJME_PROBABILISTIC.md](PJME_PROBABILISTIC.md) for pinball losses,
methodology, and limitations.

## Interactive dashboard

After generating the data and experiment artifacts, launch the portfolio UI:

```bash
make dashboard
```

The Streamlit dashboard opens at `http://localhost:8501` and provides:

- the complete PJME demand history and headline dataset statistics;
- the frozen-test leaderboard and model ablations;
- an interactive weekly comparison of actuals and point forecasts;
- calibrated probabilistic intervals and coverage diagnostics;
- optional local fit, latency, size, and memory measurements;
- a concise explanation of the anti-leakage evaluation contract.

## Local inference performance

Measure fit time, warm weekly prediction latency, throughput, serialized model
size, and indicative process-memory change:

```bash
make performance
```

On an Apple silicon development machine, both 300-tree LightGBM variants served
the complete 168-hour horizon in under one millisecond median in-process latency
after warmup. See [PERFORMANCE.md](PERFORMANCE.md) for the hardware-qualified
protocol, results, and limitations.

## Read-only API

Serve the generated artifacts through FastAPI:

```bash
make api
```

OpenAPI documentation is available at `http://localhost:8000/docs`. The API
exposes health, metadata, leaderboard, point-forecast, calibrated probabilistic
forecast, and optional local-performance endpoints. It returns `503` with setup
instructions when core artifacts are unavailable and `404` if the optional
performance run has not been generated.

Build the same service as a container:

```bash
make docker-build
docker run --rm -p 8000:8000 \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  gridcast:latest
```

## Methodology

- **Frequency:** hourly
- **Forecast horizon:** 168 hours (one week)
- **Baselines:** persistence, previous day, and previous week
- **Validation:** expanding training window with non-overlapping weekly folds
- **Metrics:** MAE, RMSE, and MASE
- **Data:** public PJME hourly load and Philadelphia ERA5 temperature; synthetic
  demand is retained only for the offline demo and tests

MASE scales absolute error by the in-sample seasonal-naive error. Values below
1.0 indicate an improvement over the in-sample seasonal-naive benchmark.

## Architecture

```text
PJM load + ERA5 temperature
            |
            v
Quality checks + leakage-safe features
            |
            v
Weekly walk-forward validation and frozen test
            |
            +--> point models and ablations
            |
            +--> P10 / P50 / P90 --> validation-only conformal calibration
                                      |
                                      v
                         Metrics, forecasts, and charts
```

## Roadmap

- Seasonal conformal calibration
- Italian day-ahead benchmark with archived weather vintages
- Decision regret and day-block confidence intervals
- Deployment of the API and dashboard to a public demo environment

## Development

```bash
make format  # apply Ruff formatting and safe fixes
make check   # lint, formatting check, and strict mypy
make test    # tests with branch coverage >= 90%
make full    # all checks and tests
```

## Limitations

The public PJME dataset ends in 2018, and a single Philadelphia weather point
cannot represent the complete PJM East footprint. ERA5 is reanalysis rather
than an archived operational forecast, so GridCast uses it only through delayed
values and prior-year climatology. Reported results are research benchmarks,
not evidence of production readiness.

## License

MIT
