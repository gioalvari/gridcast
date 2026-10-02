from pathlib import Path

import pytest

from gridcast.data import generate_synthetic_load
from gridcast.serving.bundle import train_bundle, write_bundle


@pytest.fixture(scope="session")
def bundle_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one intentionally small, valid bundle for serving integration tests."""
    directory = tmp_path_factory.mktemp("serving-bundle") / "0.1.0"
    trained = train_bundle(
        generate_synthetic_load(periods=24 * 7 * 10),
        "0.1.0",
        n_estimators=3,
        calibration_hours=168,
    )
    return write_bundle(trained, directory)
