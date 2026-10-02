from pathlib import Path

import pytest

from gridcast.cli import main
from gridcast.data import generate_synthetic_load
from gridcast.serving.settings import ServingSettings


def test_settings_defaults_and_all_validation_paths() -> None:
    defaults = ServingSettings.from_env({"GRIDCAST_MODEL_URI": "file:///model"})
    assert defaults.max_in_flight == 32
    assert defaults.environment == "local"
    with pytest.raises(ValueError, match="required"):
        ServingSettings.from_env({})
    for name, value, message in (
        ("GRIDCAST_REQUEST_TIMEOUT_S", "x", "numeric"),
        ("GRIDCAST_MAX_IN_FLIGHT", "x", "integer"),
        ("GRIDCAST_BREAKER_FAILURES", "x", "integer"),
        ("GRIDCAST_BREAKER_RESET_S", "x", "numeric"),
        ("GRIDCAST_MAX_BODY_BYTES", "x", "integer"),
        ("GRIDCAST_FAULT_ERROR_RATE", "x", "numeric"),
        ("GRIDCAST_FAULT_LATENCY_MS", "x", "integer"),
    ):
        with pytest.raises(ValueError, match=message):
            ServingSettings.from_env(
                {"GRIDCAST_MODEL_URI": "file:///model", name: value}
            )
    for name, value in (
        ("GRIDCAST_REQUEST_TIMEOUT_S", "0"),
        ("GRIDCAST_MAX_IN_FLIGHT", "0"),
        ("GRIDCAST_BREAKER_FAILURES", "0"),
        ("GRIDCAST_BREAKER_RESET_S", "0"),
        ("GRIDCAST_MAX_BODY_BYTES", "0"),
        ("GRIDCAST_FAULT_ERROR_RATE", "2"),
        ("GRIDCAST_FAULT_LATENCY_MS", "-1"),
    ):
        with pytest.raises(ValueError, match="out of range"):
            ServingSettings.from_env(
                {"GRIDCAST_MODEL_URI": "file:///model", name: value}
            )
    with pytest.raises(ValueError, match="fault injection"):
        ServingSettings.from_env(
            {
                "GRIDCAST_MODEL_URI": "file:///model",
                "GRIDCAST_ENV": "production",
                "GRIDCAST_FAULT_LATENCY_MS": "1",
            }
        )


def test_serving_cli_package_verify_publish_and_errors(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assert (
        main(
            [
                "model",
                "package",
                "--synthetic",
                "--version",
                "1.0.0",
                "--output-dir",
                str(bundle),
                "--n-estimators",
                "1",
                "--calibration-hours",
                "168",
            ]
        )
        == 0
    )
    assert main(["model", "verify", str(bundle)]) == 0
    published = tmp_path / "published"
    assert main(["model", "publish", str(bundle), "--uri", published.as_uri()]) == 0
    data = tmp_path / "load.csv"
    generate_synthetic_load(periods=1681).to_csv(data, index=False)
    from_data = tmp_path / "from-data"
    assert (
        main(
            [
                "model",
                "package",
                "--data",
                str(data),
                "--version",
                "1.0.1",
                "--output-dir",
                str(from_data),
                "--n-estimators",
                "1",
                "--calibration-hours",
                "168",
            ]
        )
        == 0
    )
    with pytest.raises(FileExistsError):
        main(["model", "publish", str(bundle), "--uri", published.as_uri()])
    with pytest.raises(ValueError, match="canonical"):
        main(
            [
                "model",
                "package",
                "--data",
                str(tmp_path / "invalid.txt"),
                "--version",
                "1.0.2",
                "--output-dir",
                str(tmp_path / "bad"),
            ]
        )
