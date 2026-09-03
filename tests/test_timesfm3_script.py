import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_timesfm3.py"


@dataclass
class FakeModelConfig:
    """Capture TimesFM 3 model settings passed by the adapter."""

    checkpoint_path: str
    per_core_batch_size: int
    device: str


@dataclass
class FakeOutput:
    """One fake TimesFM 3 forecast output."""

    forecast: np.ndarray | None
    quantiles: np.ndarray | None


class FakeTimesFM3Evaluator:
    """Small stand-in for the optional TimesFM 3 runtime."""

    latest: "FakeTimesFM3Evaluator | None" = None

    def __init__(self, config: FakeModelConfig) -> None:
        self.config = config
        self.device = "cpu"
        self.arguments: dict[str, object] = {}
        FakeTimesFM3Evaluator.latest = self

    def predict_batch(self, **kwargs: object) -> list[FakeOutput]:
        self.arguments = kwargs
        contexts = kwargs["contexts"]
        assert isinstance(contexts, list)
        horizon = kwargs["horizon"]
        assert isinstance(horizon, int)
        quantiles = np.stack(
            [np.full(horizon, index, dtype=float) for index in range(9)],
            axis=1,
        )
        return [FakeOutput(np.ones(horizon, dtype=float), quantiles) for _ in contexts]


def _load_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    timesfm3 = ModuleType("timesfm3")
    timesfm3.ModelConfig = FakeModelConfig  # type: ignore[attr-defined]
    timesfm3.TimesFM3Evaluator = FakeTimesFM3Evaluator  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.hf_hub_download = lambda **_: "model.safetensors"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timesfm3", timesfm3)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    spec = importlib.util.spec_from_file_location(
        "gridcast_timesfm3_script", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timesfm3_adapter_forwards_settings_and_maps_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(monkeypatch)
    settings = module.TimesFM3Settings(per_core_batch_size=4)

    snapshot = Path("snapshot")
    forecaster = module.TimesFM3Forecaster(snapshot, settings)
    inputs = [np.arange(1024, dtype=np.float32)]
    forecast = forecaster.forecast(inputs, 168)
    evaluator = FakeTimesFM3Evaluator.latest

    assert evaluator is not None
    assert evaluator.config == FakeModelConfig(
        checkpoint_path=str(snapshot),
        per_core_batch_size=4,
        device="cpu",
    )
    assert forecaster.device == "cpu"
    assert len(inputs) == 1
    assert evaluator.arguments["contexts"] is not inputs
    assert evaluator.arguments["use_symmetric_averaging"] is True
    assert evaluator.arguments["make_positive"] is True
    assert evaluator.arguments["sort_quantiles"] is True
    assert evaluator.arguments["use_znorm"] is False
    assert evaluator.arguments["padding_mode"] == "none"
    assert np.all(forecast.point == 1.0)
    assert np.all(forecast.p10 == 0.0)
    assert np.all(forecast.p50 == 4.0)
    assert np.all(forecast.p90 == 8.0)


def test_timesfm3_runtime_rejects_unsupported_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(monkeypatch)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    with pytest.raises(RuntimeError, match="supports Apple silicon"):
        module._validate_runtime()


def test_timesfm3_rejects_unexpected_checkpoint_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"wrong")

    with pytest.raises(RuntimeError, match="unexpected SHA-256"):
        module._verify_digest(checkpoint, "a" * 64)


def test_timesfm3_main_pins_and_records_checkpoint_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script(monkeypatch)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config_path = snapshot / "config.json"
    checkpoint_path = snapshot / "model.safetensors"
    config_path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"weights")
    downloads: list[dict[str, object]] = []
    written_environment: dict[str, object] = {}

    def download(**kwargs: object) -> str:
        downloads.append(kwargs)
        return str(snapshot / str(kwargs["filename"]))

    def write_artifacts(*_: object, **kwargs: object) -> None:
        written_environment.update(cast(dict[str, object], kwargs["environment"]))

    class FakeRunner:
        def __init__(self, _: Path, __: object) -> None:
            self.device = "cpu"

    monkeypatch.setattr(module, "_validate_runtime", lambda: None)
    monkeypatch.setattr(module, "hf_hub_download", download)
    monkeypatch.setattr(module.pd, "read_parquet", lambda _: object())
    monkeypatch.setattr(
        module,
        "run_foundation_benchmark",
        lambda *_: SimpleNamespace(metrics={}),
    )
    monkeypatch.setattr(module, "write_foundation_artifacts", write_artifacts)
    monkeypatch.setattr(module, "TimesFM3Forecaster", FakeRunner)
    monkeypatch.setattr(module, "_installed_packages", lambda: {"timesfm": "3.0.0"})
    monkeypatch.setattr(module, "version", lambda _: "3.0.0")
    monkeypatch.setattr(
        module,
        "_verify_digest",
        lambda path, expected: expected,
    )
    monkeypatch.setattr(module, "_required_digest", lambda _: "c" * 64)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])

    assert module.main() == 0
    assert downloads == [
        {
            "repo_id": module.MODEL_ID,
            "filename": "config.json",
            "revision": module.MODEL_REVISION,
        },
        {
            "repo_id": module.MODEL_ID,
            "filename": "model.safetensors",
            "revision": module.MODEL_REVISION,
        },
    ]
    assert written_environment["checkpoint_sha256"] == (
        module.EXPECTED_CHECKPOINT_SHA256
    )
    assert written_environment["checkpoint_config_sha256"] == (
        module.EXPECTED_CONFIG_SHA256
    )
    assert written_environment["dependency_lock_sha256"] == "c" * 64
