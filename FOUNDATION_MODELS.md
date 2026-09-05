# Time-series foundation models

GridCast supports Google TimesFM 2.5 and TimesFM 3.0 as optional zero-shot
benchmarks. Their checkpoints and heavy PyTorch runtimes are isolated from the
standard project installation.

## Model selection

| Model | Weights license | Included |
|---|---|---|
| TimesFM 2.5 200M | Apache-2.0 | Yes |
| TimesFM 3.0 | `timesfm-non-commercial-license-v1.0` | Optional research only |

TimesFM 3.0 source code is Apache-2.0, but its current pretrained weights do not
permit commercial or production use. GridCast integrates those public weights
only as a separate research benchmark and never includes or redistributes them.

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

### TimesFM 3.0 research benchmark

The TimesFM 3 runner uses the official `timesfm==3.0.0` PyPI package and public
`google/timesfm-3.0-pytorch` checkpoint, pinned to immutable revision
`c71907076f28b1241d1fccc37efd183d0912cd13`. Read the checkpoint's separate
non-commercial license before running:

```bash
make timesfm3
```

The default protocol matches TimesFM 2.5: 1,024 target-only context hours, 168
forecast hours, and the same 52 historical holdout folds. It uses the official
evaluator defaults for symmetric averaging, positive forecasts, sorted P10-P90
quantiles, no z-normalization, and no padding. Regenerate its separate lock with
`make timesfm3-lock`.

No TimesFM 3 result is reported until a complete run succeeds. Its output
directory is `artifacts/foundation/timesfm-3.0/`, which remains outside Git.

The first run downloads model weights from Hugging Face. Generated weights,
forecasts, and artifacts remain outside Git.

## Historical holdout results

| Model | Training on PJME | MAE (MW) | RMSE (MW) | MASE |
|---|---|---:|---:|---:|
| TimesFM 3.0 zero-shot | None | **1,763.63** | **2,481.15** | **0.585** |
| TimesFM 2.5 200M zero-shot | None | **1,926.88** | **2,740.02** | **0.639** |
| LightGBM + weather + holidays | Five-year rolling window | 2,901.57 | 3,934.23 | 0.962 |
| Daily seasonal naive | None | 2,993.52 | 3,998.30 | 0.992 |
| Weekly seasonal naive | None | 3,585.07 | 4,856.16 | 1.189 |

TimesFM 2.5 reduces aggregate MAE by 33.59% relative to combined LightGBM and by
35.63% relative to daily seasonal naive. It beats those models in 39 and 44 of
52 weekly folds, respectively. Median weekly TimesFM MAE is 1,478.75 MW.

TimesFM 2.5's raw P10-P90 interval reaches 74.90% coverage against an 80% nominal
target, with a mean width of 5,455.59 MW. This is substantially better calibrated
than GridCast's uncalibrated LightGBM quantiles, but it still requires calibration
before operational interpretation.

TimesFM 3.0 reduces observed aggregate MAE by 8.47% relative to TimesFM 2.5 and
wins 32 of 52 paired weekly folds. Its raw P10-P90 interval reaches 80.11%
coverage with a mean width of 5,261.22 MW. Dependence-aware uncertainty for the
version-to-version difference crosses zero: the marginal 95% interval is
[-226.76, 567.98] MW and the Bonferroni-adjusted interval is
[-357.03, 708.19] MW. The
observed version improvement is therefore uncertain, and the non-commercial
license prevents production use. TimesFM 2.5's 974.69 MW improvement over
LightGBM has an adjusted interval of [317.80, 1,848.05] MW. See
[MODEL_COMPARISON.md](MODEL_COMPARISON.md).

The generated summary reports the complete first call, including any lazy PyTorch
compilation, separately from a repeated warm 52-fold inference. Model loading and
the initial weight download are excluded from both measurements. Warm inference
takes approximately 8.5 seconds on the documented Apple-silicon environment.
First-call timing also depends on the local compiler cache, so exact runtime claims
should be read from the generated artifact. In the documented environment,
TimesFM 3 first-call and warm inference took 10.01 and 10.29 seconds,
respectively, excluding model loading and download.

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
- TimesFM 3 weights are restricted to non-commercial, non-production research
  use under their separate license.

The result is strong evidence that foundation models deserve further evaluation,
not proof of production superiority. A defensible next step is a newer untouched
dataset with documented non-overlap and archived operational weather forecasts.
