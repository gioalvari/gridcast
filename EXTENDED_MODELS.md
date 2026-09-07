# Extended point-model benchmark

GridCast optionally evaluates additional tabular boosters on the same leakage-safe
features and chronological folds used by the primary LightGBM benchmark.

## Models

| Model | Package | Role |
|---|---|---|
| LightGBM exogenous | `lightgbm` | Reference candidate |
| HistGradientBoosting exogenous | `scikit-learn` | Built-in histogram baseline |
| CatBoost exogenous | `catboost` | Optional symmetric-tree MAE candidate |
| XGBoost exogenous | `xgboost` | Optional histogram tree candidate |
| AutoML validation selection | GridCast | Chooses one fixed candidate from validation MAE |

All candidates use the same exogenous feature matrix and five-year training
window. They share nominal settings of 300 iterations, learning rate `0.05`, and
seed `42`, but retain fixed backend-specific topology, regularization, sampling,
and defaults; capacity and compute are not matched. HistGradientBoosting disables
random internal early stopping so all historical rows remain available. The
policy is not an ensemble: it selects the lowest aggregate MAE across the 12
validation folds, then exposes that already-evaluated candidate under a stable
policy ID. It does not inspect holdout targets during selection.

Install and run:

```bash
make install-extended
make benchmark-extended
```

Artifacts are written separately to `artifacts/benchmark-extended/`, so the
published primary benchmark and six-pair statistical family remain unchanged.

## Historical results

The validation-only policy selected CatBoost:

| Candidate | Validation MAE (MW) | Holdout MAE (MW) | Holdout MASE |
|---|---:|---:|---:|
| CatBoost exogenous | **3,821.19** | **2,842.20** | **0.942** |
| LightGBM exogenous | 3,908.83 | 2,901.57 | 0.962 |
| HistGradientBoosting exogenous | 3,948.95 | 2,895.70 | 0.960 |
| XGBoost exogenous | 4,165.32 | 2,916.48 | 0.967 |

CatBoost has the lowest observed task-trained holdout MAE and improves on
LightGBM by 59.37 MW, or 2.05%. However, an exploratory paired four-week circular
block bootstrap gives a marginal 95% interval of `[-6.94, 121.70] MW` and a
three-comparison Bonferroni-adjusted interval of `[-21.70, 134.34] MW`.
HistGradientBoosting and XGBoost also have adjusted intervals crossing zero.
Under the specified exploratory four-week circular-block protocol, none of the
three adjusted intervals excludes zero. Sensitivity with circular blocks of
2, 4, 6, 8, 13, and 26 weeks is written beside the primary comparison because
the conclusion can depend on block specification.

## Limitations

- This extension was designed after inspecting the historical holdout and is
  explicitly exploratory.
- The 12 validation weeks are a small selection sample; CatBoost selection may
  be unstable across years or regions.
- The candidates use fixed, non-optimized recipes and are not capacity- or
  compute-matched.
- The AutoML label means validation-only algorithm selection, not an external
  AutoML framework, stacking, or holdout-tuned ensemble.
- Runtime and memory were not benchmarked per backend in this study.
