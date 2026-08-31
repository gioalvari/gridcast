import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.cli import main
from gridcast.columns import Col
from gridcast.data import generate_synthetic_load


def test_demo_writes_reproducible_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"

    exit_status = main(
        [
            "demo",
            "--periods",
            "504",
            "--initial-window",
            "336",
            "--horizon",
            "168",
            "--step",
            "168",
            "--seasonal-period",
            "168",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_status == 0
    assert {path.name for path in output_dir.iterdir()} == {
        "load.csv",
        "forecasts.csv",
        "metrics.csv",
        "summary.json",
    }
    report = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["model"] == "seasonal_naive"
    assert report["metrics"]["folds"] == 1


def test_eda_command_reads_parquet_and_writes_report(tmp_path: Path) -> None:
    input_path = tmp_path / "load.parquet"
    output_path = tmp_path / "eda"
    generate_synthetic_load(periods=24 * 14).to_parquet(input_path, index=False)

    exit_status = main(
        ["eda", "--input", str(input_path), "--output-dir", str(output_path)]
    )

    assert exit_status == 0
    assert pd.read_csv(output_path / "hourly_profile.csv").shape[0] == 24


def test_benchmark_command_writes_leaderboard(tmp_path: Path) -> None:
    input_path = tmp_path / "load.parquet"
    output_path = tmp_path / "benchmark"
    generate_synthetic_load(periods=24 * 49).to_parquet(input_path, index=False)

    exit_status = main(
        [
            "benchmark",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_path),
            "--validation-folds",
            "1",
            "--test-folds",
            "1",
            "--max-train-hours",
            str(24 * 21),
            "--n-estimators",
            "5",
            "--without-exogenous",
        ]
    )

    assert exit_status == 0
    leaderboard = pd.read_csv(output_path / "leaderboard.csv")
    assert set(leaderboard["split"]) == {"validation", "test"}


def test_probabilistic_command_writes_calibrated_metrics(tmp_path: Path) -> None:
    input_path = tmp_path / "load.parquet"
    weather_path = tmp_path / "weather.parquet"
    output_path = tmp_path / "probabilistic"
    data = generate_synthetic_load(periods=24 * 400, start="2016-01-01")
    data.to_parquet(input_path, index=False)
    data[[Col.TIMESTAMP]].assign(
        **{Col.TEMPERATURE: np.sin(np.arange(len(data)) / 100.0) * 10.0}
    ).to_parquet(weather_path, index=False)

    exit_status = main(
        [
            "probabilistic",
            "--input",
            str(input_path),
            "--weather",
            str(weather_path),
            "--output-dir",
            str(output_path),
            "--validation-folds",
            "1",
            "--test-folds",
            "1",
            "--max-train-hours",
            str(24 * 380),
            "--n-estimators",
            "3",
        ]
    )

    assert exit_status == 0
    metrics = pd.read_csv(output_path / "metrics.csv")
    assert set(metrics["split"]) == {"validation", "test"}
    assert metrics["calibrated_coverage"].between(0.0, 1.0).all()
