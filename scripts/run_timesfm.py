#!/usr/bin/env python3
import argparse
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import distributions, version
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm
import torch
from huggingface_hub import hf_hub_download

from gridcast.foundation import (
    FoundationConfig,
    FoundationForecast,
    FoundationForecaster,
    run_foundation_benchmark,
    write_foundation_artifacts,
)
from gridcast.foundation_models import TIMESFM_2P5
from gridcast.provenance import file_sha256

MODEL_ID = TIMESFM_2P5.model_id
MODEL_REVISION = TIMESFM_2P5.model_revision
SUPPORTED_PYTHON = "3.12.12"


@dataclass(frozen=True)
class TimesFMSettings:
    """Fully resolved TimesFM compilation and forecast settings."""

    per_core_batch_size: int = 1
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = True
    torch_compile: bool = True


class TimesFMForecaster(FoundationForecaster):
    """TimesFM 2.5 adapter for the GridCast foundation-model protocol."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        settings: TimesFMSettings,
    ) -> None:
        """Load and compile the pinned Apache-2.0 TimesFM checkpoint."""
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_compile=settings.torch_compile,
        )
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=context_length,
                max_horizon=horizon,
                per_core_batch_size=settings.per_core_batch_size,
                normalize_inputs=settings.normalize_inputs,
                use_continuous_quantile_head=settings.use_continuous_quantile_head,
                force_flip_invariance=settings.force_flip_invariance,
                infer_is_positive=settings.infer_is_positive,
                fix_quantile_crossing=settings.fix_quantile_crossing,
            )
        )

    @property
    def device(self) -> str:
        """Return the device selected by the loaded TimesFM model."""
        return str(self._model.model.device)

    def forecast(
        self,
        inputs: list[np.ndarray],
        horizon: int,
    ) -> FoundationForecast:
        """Return zero-shot point and quantile forecasts."""
        point, quantiles = self._model.forecast(
            horizon=horizon,
            inputs=inputs.copy(),
        )
        return FoundationForecast(
            point=np.asarray(point, dtype=float),
            p10=np.asarray(quantiles[:, :, 1], dtype=float),
            p50=np.asarray(quantiles[:, :, 5], dtype=float),
            p90=np.asarray(quantiles[:, :, 9], dtype=float),
        )


def _installed_packages() -> dict[str, str]:
    return dict(
        sorted(
            (distribution.metadata["Name"].lower(), distribution.version)
            for distribution in distributions()
            if distribution.metadata["Name"]
        )
    )


def _validate_runtime() -> None:
    actual = (
        f"{platform.system()} {platform.machine()} / Python {platform.python_version()}"
    )
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or platform.python_version() != SUPPORTED_PYTHON
    ):
        msg = (
            "the locked TimesFM benchmark supports Apple silicon with Python "
            f"{SUPPORTED_PYTHON}; received {actual}"
        )
        raise RuntimeError(msg)


def _required_digest(path: Path) -> str:
    digest = file_sha256(path)
    if digest is None:
        raise FileNotFoundError(path)
    return digest


def main() -> int:
    """Run the pinned TimesFM zero-shot historical holdout benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/pjme_hourly.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/foundation/timesfm-2.5-200m"),
    )
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--holdout-folds", type=int, default=52)
    parser.add_argument("--per-core-batch-size", type=int, default=1)
    args = parser.parse_args()

    _validate_runtime()
    settings = TimesFMSettings(per_core_batch_size=args.per_core_batch_size)
    if settings.per_core_batch_size < 1:
        parser.error("--per-core-batch-size must be positive")
    config = FoundationConfig(
        model_name=TIMESFM_2P5.model_name,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        weights_license=TIMESFM_2P5.weights_license,
        context_length=args.context_length,
        horizon=args.horizon,
        holdout_folds=args.holdout_folds,
        model_parameters=asdict(settings),
    )
    data = pd.read_parquet(args.input)
    forecaster = TimesFMForecaster(config.context_length, config.horizon, settings)
    if forecaster.device != "cpu":
        msg = f"CPU benchmark requested but TimesFM selected {forecaster.device}"
        raise RuntimeError(msg)
    result = run_foundation_benchmark(data, forecaster, config)
    checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename="model.safetensors",
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    write_foundation_artifacts(
        result,
        config,
        data,
        args.output_dir,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "timesfm": version("timesfm"),
            "device": forecaster.device,
            "checkpoint_sha256": _required_digest(checkpoint),
            "dependency_lock_sha256": _required_digest(
                Path("scripts/timesfm-requirements.txt")
            ),
            "installed_packages": _installed_packages(),
        },
    )
    print(result.metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
