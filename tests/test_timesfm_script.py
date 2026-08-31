import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_timesfm.py"


@dataclass
class FakeForecastConfig:
    """Capture TimesFM forecast settings passed by the adapter."""

    max_context: int
    max_horizon: int
    per_core_batch_size: int
    normalize_inputs: bool
    use_continuous_quantile_head: bool
    force_flip_invariance: bool
    infer_is_positive: bool
    fix_quantile_crossing: bool


class FakeTimesFMModel:
    """Small stand-in for the optional TimesFM runtime."""

    loaded: tuple[str, str | None, bool] | None = None

    def __init__(self) -> None:
        self.model = SimpleNamespace(device="cpu")
        self.forecast_config: FakeForecastConfig | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        torch_compile: bool = True,
    ) -> "FakeTimesFMModel":
        cls.loaded = (model_id, revision, torch_compile)
        return cls()

    def compile(self, forecast_config: FakeForecastConfig) -> None:
        self.forecast_config = forecast_config

    def forecast(
        self,
        *,
        horizon: int,
        inputs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        inputs.append(np.asarray([-1.0]))
        point = np.ones((1, horizon), dtype=float)
        quantiles = np.stack(
            [np.full((1, horizon), index, dtype=float) for index in range(10)],
            axis=2,
        )
        return point, quantiles


def _load_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    timesfm = ModuleType("timesfm")
    timesfm.ForecastConfig = FakeForecastConfig  # type: ignore[attr-defined]
    timesfm.TimesFM_2p5_200M_torch = FakeTimesFMModel  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.hf_hub_download = lambda **_: "model.safetensors"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timesfm", timesfm)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    spec = importlib.util.spec_from_file_location(
        "gridcast_timesfm_script", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timesfm_adapter_forwards_settings_and_maps_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(monkeypatch)
    settings = module.TimesFMSettings(per_core_batch_size=4)

    forecaster = module.TimesFMForecaster(1024, 168, settings)
    inputs = [np.arange(1024, dtype=np.float32)]
    forecast = forecaster.forecast(inputs, 168)

    assert forecaster.device == "cpu"
    assert len(inputs) == 1
    assert FakeTimesFMModel.loaded == (
        module.MODEL_ID,
        module.MODEL_REVISION,
        True,
    )
    assert forecaster._model.forecast_config == FakeForecastConfig(
        max_context=1024,
        max_horizon=168,
        per_core_batch_size=4,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
    assert np.all(forecast.point == 1.0)
    assert np.all(forecast.p10 == 1.0)
    assert np.all(forecast.p50 == 5.0)
    assert np.all(forecast.p90 == 9.0)


def test_timesfm_runtime_rejects_unsupported_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(monkeypatch)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    with pytest.raises(RuntimeError, match="supports Apple silicon"):
        module._validate_runtime()
