import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.columns import Col
from gridcast.data import generate_synthetic_load
from gridcast.models import LightGBMLoadForecaster, LightGBMQuantileForecaster
from gridcast.serving.bundle import BundleIntegrityError, load_bundle
from gridcast.serving.predictor import (
    Predictor,
    fallback_forecast,
    population_stability_index,
)
from gridcast.serving.resilience import CircuitBreaker, FaultInjector
from gridcast.serving.settings import ServingSettings


def test_bundle_round_trip_and_tampering(
    bundle_directory: Path, tmp_path: Path
) -> None:
    loaded = load_bundle(bundle_directory)
    assert loaded.manifest.model_version == "0.1.0"
    tampered = tmp_path / "tampered"
    shutil.copytree(bundle_directory, tampered)
    (tampered / "point.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="checksum"):
        load_bundle(tampered)


def test_predictor_is_ordered_and_leakage_safe(bundle_directory: Path) -> None:
    predictor = Predictor(load_bundle(bundle_directory))
    history = generate_synthetic_load(periods=336)
    result = predictor.predict(history, 12)
    assert len(result.point) == 12
    assert np.all(result.p10 <= result.p50)
    assert np.all(result.p50 <= result.p90)
    altered = history.copy()
    # Future placeholder values are deliberately not visible through delayed features.
    future = pd.DataFrame(
        {
            Col.TIMESTAMP: pd.date_range(
                history[Col.TIMESTAMP].iloc[-1] + pd.Timedelta(hours=1),
                periods=12,
                freq="h",
                tz="UTC",
            ),
            Col.TARGET: 999_999.0,
        }
    )
    assert np.allclose(result.point, predictor.predict(altered, 12).point)
    assert len(future) == 12


def test_fallback_psi_and_settings() -> None:
    history = generate_synthetic_load(periods=336)
    result = fallback_forecast(history, 2, "0.1.0", "error")
    assert result.degraded and np.array_equal(result.point, result.p10)
    assert (
        population_stability_index(np.array([1.0, 2.0]), [0.0, 1.5, 3.0], [0.5, 0.5])
        >= 0
    )
    with pytest.raises(ValueError, match="fault injection"):
        ServingSettings.from_env(
            {
                "GRIDCAST_MODEL_URI": "file:///bundle",
                "GRIDCAST_ENV": "production",
                "GRIDCAST_FAULT_ERROR_RATE": "0.1",
            }
        )


def test_breaker_and_fault_validation() -> None:
    now = [0.0]
    breaker = CircuitBreaker(2, 1.0, lambda: now[0])
    breaker.record_failure()
    breaker.record_failure()
    assert not breaker.allow()
    now[0] = 1.0
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"
    with pytest.raises(ValueError):
        FaultInjector(2.0, 0)


def test_boosters_require_fitting() -> None:
    with pytest.raises(RuntimeError):
        _ = LightGBMLoadForecaster().booster
    with pytest.raises(RuntimeError):
        _ = LightGBMQuantileForecaster(0.5).booster
