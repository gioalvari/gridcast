#!/usr/bin/env python3
import argparse
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import distributions, version
from pathlib import Path

# The public Xet endpoint can fail while reconstructing this 1.3 GB checkpoint.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from timesfm3 import ModelConfig, TimesFM3Evaluator

from gridcast.foundation import (
    FoundationConfig,
    FoundationForecast,
    FoundationForecaster,
    run_foundation_benchmark,
    write_foundation_artifacts,
)
from gridcast.foundation_models import TIMESFM_3
from gridcast.provenance import file_sha256

MODEL_ID = TIMESFM_3.model_id
MODEL_REVISION = TIMESFM_3.model_revision
WEIGHTS_LICENSE = TIMESFM_3.weights_license
EXPECTED_CHECKPOINT_SHA256 = TIMESFM_3.checkpoint_sha256 or ""
EXPECTED_CONFIG_SHA256 = TIMESFM_3.config_sha256 or ""
SUPPORTED_PYTHON = "3.12.12"


@dataclass(frozen=True)
class TimesFM3Settings:
    """Fully resolved TimesFM 3 benchmark settings."""

    per_core_batch_size: int = 1
    return_quantiles: bool = True
    use_symmetric_averaging: bool = True
    make_positive: bool = True
    sort_quantiles: bool = True
    use_znorm: bool = False
    padding_mode: str = "none"


class TimesFM3Forecaster(FoundationForecaster):
    """TimesFM 3 adapter for the GridCast foundation-model protocol."""

    def __init__(self, snapshot: Path, settings: TimesFM3Settings) -> None:
        """Load the pinned non-commercial checkpoint snapshot on CPU."""
        self._settings = settings
        self._model = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=str(snapshot),
                per_core_batch_size=settings.per_core_batch_size,
                device="cpu",
            )
        )

    @property
    def device(self) -> str:
        """Return the device selected by TimesFM 3."""
        return str(self._model.device)

    def forecast(
        self,
        inputs: list[np.ndarray],
        horizon: int,
    ) -> FoundationForecast:
        """Return zero-shot point and P10/P50/P90 forecasts."""
        outputs = list(
            self._model.predict_batch(
                contexts=inputs.copy(),
                horizon=horizon,
                return_quantiles=self._settings.return_quantiles,
                use_symmetric_averaging=self._settings.use_symmetric_averaging,
                make_positive=self._settings.make_positive,
                sort_quantiles=self._settings.sort_quantiles,
                use_znorm=self._settings.use_znorm,
                padding_mode=self._settings.padding_mode,
            )
        )
        if any(
            output.forecast is None or output.quantiles is None for output in outputs
        ):
            msg = "TimesFM 3 did not return point and quantile forecasts"
            raise ValueError(msg)
        point = np.stack([output.forecast for output in outputs])
        quantiles = np.stack([output.quantiles for output in outputs])
        return FoundationForecast(
            point=np.asarray(point, dtype=float),
            p10=np.asarray(quantiles[:, :, 0], dtype=float),
            p50=np.asarray(quantiles[:, :, 4], dtype=float),
            p90=np.asarray(quantiles[:, :, 8], dtype=float),
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
            "the locked TimesFM 3 benchmark supports Apple silicon with Python "
            f"{SUPPORTED_PYTHON}; received {actual}"
        )
        raise RuntimeError(msg)


def _required_digest(path: Path) -> str:
    digest = file_sha256(path)
    if digest is None:
        raise FileNotFoundError(path)
    return digest


def _verify_digest(path: Path, expected: str) -> str:
    digest = _required_digest(path)
    if digest != expected:
        raise RuntimeError(f"unexpected SHA-256 for {path.name}: {digest}")
    return digest


def main() -> int:
    """Run the pinned, non-commercial TimesFM 3 historical benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/pjme_hourly.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/foundation/timesfm-3.0"),
    )
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--holdout-folds", type=int, default=52)
    parser.add_argument("--per-core-batch-size", type=int, default=1)
    args = parser.parse_args()

    _validate_runtime()
    settings = TimesFM3Settings(per_core_batch_size=args.per_core_batch_size)
    if settings.per_core_batch_size < 1:
        parser.error("--per-core-batch-size must be positive")
    config = FoundationConfig(
        model_name=TIMESFM_3.model_name,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        weights_license=WEIGHTS_LICENSE,
        context_length=args.context_length,
        horizon=args.horizon,
        holdout_folds=args.holdout_folds,
        model_parameters=asdict(settings),
    )
    data = pd.read_parquet(args.input)
    config_path = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename="config.json",
            revision=MODEL_REVISION,
        )
    )
    checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename="model.safetensors",
            revision=MODEL_REVISION,
        )
    )
    if checkpoint.parent != config_path.parent:
        msg = "TimesFM 3 checkpoint and configuration snapshots do not match"
        raise RuntimeError(msg)
    checkpoint_sha256 = _verify_digest(checkpoint, EXPECTED_CHECKPOINT_SHA256)
    config_sha256 = _verify_digest(config_path, EXPECTED_CONFIG_SHA256)
    forecaster = TimesFM3Forecaster(checkpoint.parent, settings)
    if forecaster.device != "cpu":
        msg = f"CPU benchmark requested but TimesFM 3 selected {forecaster.device}"
        raise RuntimeError(msg)
    result = run_foundation_benchmark(data, forecaster, config)
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
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_config_sha256": config_sha256,
            "dependency_lock_sha256": _required_digest(
                Path("scripts/timesfm3-requirements.txt")
            ),
            "installed_packages": _installed_packages(),
        },
    )
    print(result.metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
