# Time-series foundation models

GridCast evaluates Google TimesFM 2.5 as an optional zero-shot model. The
checkpoint and heavy PyTorch runtime are isolated from the standard project
installation.

## Model selection

| Model | Weights license | Included |
|---|---|---|
| TimesFM 2.5 200M | Apache-2.0 | Yes |
| TimesFM 3.0 | Non-commercial license | No |

TimesFM 3.0 source code is Apache-2.0, but its current pretrained weights do not
permit commercial or production use. GridCast therefore pins the Apache-2.0
TimesFM 2.5 checkpoint and its immutable Hugging Face revision.

## Protocol

- Checkpoint: `google/timesfm-2.5-200m-pytorch`
- Revision: `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`
- TimesFM package: 2.0.2
- PyTorch: 2.13.0
- Dependencies: resolved in `scripts/timesfm-requirements.txt` with hashes
- Mode: zero-shot, no PJME fine-tuning
- Context: latest 1,024 hourly target values at each origin
- Horizon: 168 hours
- Historical holdout: the same 52 weekly folds used by existing models
- Execution: CPU-only on Apple silicon, 52 internal microbatches per call
- TimesFM batch size: `per_core_batch_size=1`
- Forecast flags: input normalization, continuous quantile head, flip
  invariance, positive-series inference, and quantile-crossing correction
- Weights cache: approximately 882 MiB locally

Run the benchmark on macOS 14+ Apple silicon with Python 3.12.12 and an isolated
uv environment:

```bash
make timesfm
```

Regenerate the Apple-silicon dependency lock after intentionally changing a
foundation-model dependency with `make timesfm-lock`.

The first run downloads model weights from Hugging Face. Generated weights,
forecasts, and artifacts remain outside Git.

## Historical holdout results

| Model | Training on PJME | MAE (MW) | RMSE (MW) | MASE |
|---|---|---:|---:|---:|
| TimesFM 2.5 200M zero-shot | None | **1,926.88** | **2,740.02** | **0.639** |
| LightGBM + weather + holidays | Five-year rolling window | 2,901.57 | 3,934.23 | 0.962 |
| Daily seasonal naive | None | 2,993.52 | 3,998.30 | 0.992 |
| Weekly seasonal naive | None | 3,585.07 | 4,856.16 | 1.189 |

TimesFM reduces aggregate MAE by 33.59% relative to combined LightGBM and by
35.63% relative to daily seasonal naive. It beats those models in 39 and 44 of
52 weekly folds, respectively. Median weekly TimesFM MAE is 1,478.75 MW.

TimesFM's raw P10-P90 interval reaches 74.90% coverage against an 80% nominal
target, with a mean width of 5,455.59 MW. This is substantially better calibrated
than GridCast's uncalibrated LightGBM quantiles, but it still requires calibration
before operational interpretation.

The generated summary reports the complete first call, including any lazy PyTorch
compilation, separately from a repeated warm 52-fold inference. Model loading and
the initial weight download are excluded from both measurements. Warm inference
takes approximately 8.5 seconds on the documented Apple-silicon environment.
First-call timing also depends on the local compiler cache, so exact runtime claims
should be read from the generated artifact.

## Limitations

- TimesFM pretraining includes broad public and synthetic time-series corpora.
  GridCast cannot prove that PJM or closely related electricity series were absent,
  so benchmark contamination remains a material risk.
- The historical holdout was already inspected during earlier model development;
  it is not an untouched final test.
- The point output and quantiles are zero-shot; no validation-based selection or
  calibration has yet been applied to TimesFM.
- The 1,024-hour context omits explicit weather and calendar covariates.
- Runtime and memory depend on PyTorch, hardware, batch size, and local cache state.
- TimesFM 2.5 is not an officially supported Google product.

The result is strong evidence that foundation models deserve further evaluation,
not proof of production superiority. A defensible next step is a newer untouched
dataset with documented non-overlap and archived operational weather forecasts.
