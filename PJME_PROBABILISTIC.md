# PJME probabilistic forecasting

This report summarizes the reproducible output of:

```bash
make data
make weather
make probabilistic
```

## Evaluation protocol

- Forecast horizon: 168 hours
- Quantiles: P10, P50, and P90
- Raw interval: P10-P90 with 80% nominal coverage
- Validation: 12 weekly folds from 12 May to 4 August 2017
- Frozen test: 52 weekly folds from 4 August 2017 to 3 August 2018
- Quantile model: LightGBM with the combined calendar and weather features
- Calibration: one symmetric split-conformal correction learned only from the
  validation predictions
- Test isolation: no test observation contributes to model selection or
  conformal calibration

The implementation sorts each row's three quantile predictions before
evaluation, preventing invalid crossed intervals while preserving the three
predicted values.

## Frozen test results

| Metric | Result |
|---|---:|
| P10 pinball loss | 574.80 MW |
| P50 pinball loss | 1,445.89 MW |
| P90 pinball loss | 916.57 MW |
| P50 MAE | 2,891.77 MW |
| Raw P10-P90 coverage | 57.59% |
| Conformal coverage | 80.04% |
| Hour-conditional conformal coverage | 81.40% |
| Rolling prequential coverage | 79.52% |
| Target coverage | 80.00% |
| Raw mean interval width | 5,842.13 MW |
| Calibrated mean interval width | 8,984.87 MW |
| Hour-conditional mean interval width | 9,366.19 MW |
| Rolling prequential mean interval width | 9,221.88 MW |
| Conformal correction per bound | 1,571.37 MW |
| Global mean hourly coverage error | 2.40 percentage points |
| Conditional mean hourly coverage error | 1.72 percentage points |
| Global mean weekly coverage error | 19.32 percentage points |
| Rolling mean weekly coverage error | 19.34 percentage points |

## Interpretation

The independently trained quantiles are substantially under-dispersed: their
raw interval covers only 57.59% of observations despite an 80% nominal target.
The validation-only conformal correction raises frozen-test coverage to 80.04%,
showing that the calibration transfers well across the final year.

This reliability has a cost. The calibrated interval is 53.8% wider than the
raw interval. The plot for the latest test week also shows that a constant
symmetric correction can be unnecessarily wide overnight while remaining
valuable around volatile daytime peaks.

An hour-conditional extension estimates 24 independent corrections using only
the same validation folds. It reduces mean absolute hourly coverage error by
28.5%, from 2.40 to 1.72 percentage points. The trade-off is 81.40% aggregate
coverage and intervals that are 4.2% wider than the globally calibrated
interval. GridCast therefore preserves both methods rather than presenting the
conditional version as an unconditional improvement.

A causal rolling experiment retains the latest 12 completed weekly folds. The
first test week uses validation only; each subsequent week may use labels from
earlier completed test weeks, but never from itself or the future. Aggregate
coverage moves to 79.52%, while intervals become 2.64% wider than static global
calibration. Mean weekly coverage error is effectively unchanged and slightly
worse (19.34 versus 19.32 percentage points). Rolling calibration is therefore
retained as a transparent negative result, not selected as the default method.

## Limitations

- The interval has marginal coverage across the full test year; it does not
  guarantee 80% coverage for every hour, season, or individual week.
- The hour-conditional method uses only 84 validation observations per hour,
  making each finite-sample correction relatively coarse.
- Neither method guarantees coverage for each individual week or season.
- The rolling result is prequential rather than a fully untouched frozen-test
  estimate because labels from completed test weeks calibrate later weeks.
- ERA5 observations are used only through delayed values and prior-year
  climatology because archived operational forecasts do not span 2002-2018.
- Quantile sorting avoids crossing but does not jointly train a coherent
  conditional distribution.

## Next experiment

Evaluate seasonal groups against the static global and hourly baselines, with
explicit minimum sample sizes and no use of frozen-test labels.
