# Decision-aware evaluation

GridCast complements statistical forecast metrics with stylized scheduling
costs. These are synthetic cost units, not electricity-market euros.

## Scenarios

| Scenario | Shortage cost | Surplus cost | Cost-optimal quantile |
|---|---:|---:|---:|
| Symmetric | 1 | 1 | P50 |
| Shortage heavy | 3 | 1 | P75 |
| Surplus heavy | 1 | 3 | P25 |

For point models, each prediction is treated as scheduled load and compared
with the hindsight-perfect zero-error schedule. For probabilistic forecasts,
P25 and P75 are linearly interpolated from P10/P50/P90 and compared with
scheduling at P50.

Generate the result tables and charts with:

```bash
make benchmark
make probabilistic
```

The point benchmark writes `decision_costs.csv` and `decision_costs.png`. The
probabilistic benchmark writes a separate `decision_costs.csv` comparing P50
with the cost-aware quantile. Results remain stylized decision sensitivity until
a defensible market-cost model is available.

## Historical holdout findings

Point-model rankings depend on the decision objective:

| Scenario | Lowest-cost point schedule | Mean cost | Comparison |
|---|---|---:|---:|
| Symmetric | LightGBM + weather + holidays | 2,901.57 | 3.17% below daily naive |
| Shortage heavy | Daily seasonal naive | 5,106.44 | 20.59% below combined LightGBM |
| Surplus heavy | LightGBM + weather + holidays | 5,448.63 | 20.66% below daily naive |

The daily seasonal-naive model has worse aggregate MAE than combined LightGBM,
yet it is the cheapest point schedule when shortage costs are three times
surplus costs. This demonstrates why point accuracy alone does not determine the
best operational decision.

Selecting a cost-aware quantile also improves the probabilistic model schedule:

| Scenario | P50 cost | Cost-aware schedule | Cost-aware cost | Savings |
|---|---:|---:|---:|---:|
| Symmetric | 2,891.77 | P50 | 2,891.77 | 0.00% |
| Shortage heavy | 6,136.33 | P75 | 5,239.61 | 14.61% |
| Surplus heavy | 5,430.77 | P25 | 4,165.73 | 23.29% |

P25 and P75 are linearly interpolated from the trained P10, P50, and P90
forecasts. These percentages are sensitivity results under synthetic linear
penalties, not claims about realized market savings.
