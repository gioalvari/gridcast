# PJME forecasting benchmark

This report summarizes the reproducible output of:

```bash
make data
make weather
make benchmark
```

## Evaluation protocol

- Forecast horizon: 168 hours
- Validation: 12 weekly folds from 12 May to 4 August 2017
- Historical holdout: 52 weekly folds from 4 August 2017 to 3 August 2018
- LightGBM training history: rolling five-year window
- LightGBM objective: L1 regression
- Metrics: MAE, RMSE, and MASE scaled by the in-sample weekly naive error
- Leakage control: all target-derived features are delayed by at least 168 hours
- Weather proxy: Philadelphia ERA5 temperature, delayed by 168/336 hours
- Weather climatology: month-day-hour averages using prior years only
- Calendar: United States federal holiday, pre-holiday, and post-holiday flags

The complete historical holdout covers every season. Models are retrained at each
weekly origin using only observations preceding that origin.

## Historical holdout results

| Model | MAE (MW) | RMSE (MW) | MASE | MAE vs weekly naive |
|---|---:|---:|---:|---:|
| LightGBM + weather + holidays | 2,901.57 | 3,934.23 | 0.962 | +19.07% |
| LightGBM + weather | 2,913.06 | 3,964.45 | 0.966 | +18.74% |
| LightGBM | 2,972.24 | 4,005.40 | 0.986 | +17.09% |
| LightGBM + holidays | 2,977.34 | 4,029.76 | 0.987 | +16.95% |
| Daily seasonal naive | 2,993.52 | 3,998.30 | 0.992 | +16.50% |
| Weekly seasonal naive | 3,585.07 | 4,856.16 | 1.189 | baseline |
| Persistence | 4,499.88 | 5,697.31 | 1.492 | -25.52% |

The combined model has the lowest observed aggregate MAE: 19.07% below the
weekly naive baseline, 2.38% below base LightGBM, and 3.07% below daily naive.
The weather-only ablation captures most of this difference. Holidays alone
slightly worsen aggregate MAE, while combining holidays with weather is 0.39%
lower than weather-only. Dependence-aware uncertainty for these paired
differences is not yet reported, so the small margins do not establish reliable
superiority.

## Stability

| Model | Weekly wins | Median weekly MAE (MW) | Weekly MAE standard deviation (MW) |
|---|---:|---:|---:|
| Daily seasonal naive | 15 / 52 | 2,443.98 | 1,481.51 |
| LightGBM + weather + holidays | 14 / 52 | 2,412.02 | 1,707.18 |
| LightGBM + weather | 7 / 52 | 2,365.96 | 1,726.79 |
| Weekly seasonal naive | 5 / 52 | 3,153.14 | 1,939.66 |
| LightGBM + holidays | 5 / 52 | 2,420.17 | 1,710.67 |
| LightGBM | 5 / 52 | 2,337.68 | 1,673.79 |
| Persistence | 1 / 52 | 4,228.35 | 1,565.27 |

The combined model has the best aggregate MAE, but the daily naive model wins
the most individual weeks and has lower error variance. The combined model
beats the base LightGBM model in 34 of 52 folds. This is useful evidence for the
new features, but not enough to treat the current model as production-ready.

## Interpretation

- Daily seasonality is more reliable than weekly seasonality over this holdout
  year, indicating meaningful week-to-week level changes.
- Delayed weather and prior-year climatology provide a measurable improvement
  without exposing realized future weather.
- Holiday flags add value only in combination with weather in this experiment.
- All LightGBM variants still underestimate some demand peaks because they do
  not have archived day-ahead weather forecasts for the full holdout period.
- The small margin over daily naive and higher error variance justify adding
  exogenous features before increasing model complexity.

## Next experiment

The probabilistic extension is documented in
[PJME_PROBABILISTIC.md](PJME_PROBABILISTIC.md). For a newer dataset, archived
operational weather forecasts should replace delayed temperature proxies.
