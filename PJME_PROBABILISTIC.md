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
| Target coverage | 80.00% |
| Raw mean interval width | 5,842.13 MW |
| Calibrated mean interval width | 8,984.87 MW |
| Conformal correction per bound | 1,571.37 MW |

## Interpretation

The independently trained quantiles are substantially under-dispersed: their
raw interval covers only 57.59% of observations despite an 80% nominal target.
The validation-only conformal correction raises frozen-test coverage to 80.04%,
showing that the calibration transfers well across the final year.

This reliability has a cost. The calibrated interval is 53.8% wider than the
raw interval. The plot for the latest test week also shows that a constant
symmetric correction can be unnecessarily wide overnight while remaining
valuable around volatile daytime peaks.

## Limitations

- The interval has marginal coverage across the full test year; it does not
  guarantee 80% coverage for every hour, season, or individual week.
- A single conformal correction ignores hour-dependent and seasonal error
  variance.
- ERA5 observations are used only through delayed values and prior-year
  climatology because archived operational forecasts do not span 2002-2018.
- Quantile sorting avoids crossing but does not jointly train a coherent
  conditional distribution.

## Next experiment

Calibrate residuals by forecast hour or season using rolling conformal windows,
then compare conditional coverage and interval width against this global
split-conformal baseline.
