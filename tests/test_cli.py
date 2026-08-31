import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_performance_command_writes_measurements(tmp_path: Path) -> None:
    input_path = tmp_path / "load.parquet"
    weather_path = tmp_path / "weather.parquet"
    output_path = tmp_path / "performance"
    data = generate_synthetic_load(periods=24 * 400, start="2017-01-01")
    data.to_parquet(input_path, index=False)
    data[[Col.TIMESTAMP]].assign(**{Col.TEMPERATURE: 10.0}).to_parquet(
        weather_path, index=False
    )

    exit_status = main(
        [
            "performance",
            "--input",
            str(input_path),
            "--weather",
            str(weather_path),
            "--output-dir",
            str(output_path),
            "--max-train-hours",
            str(24 * 380),
            "--n-estimators",
            "3",
            "--warmup-runs",
            "1",
            "--repetitions",
            "2",
        ]
    )

    assert exit_status == 0
    measurements = pd.read_csv(output_path / "measurements.csv")
    assert len(measurements) == 3


def test_entsoe_command_passes_dates_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class Report:
        observations = 96
        resolution_minutes = 15

    def fake_ingest(
        raw_directory: Path,
        output_path: Path,
        report_path: Path,
        start: datetime,
        end: datetime,
        *,
        area: str,
        force: bool,
    ) -> Report:
        captured.update(
            raw=raw_directory,
            output=output_path,
            report=report_path,
            start=start,
            end=end,
            area=area,
            force=force,
        )
        return Report()

    monkeypatch.setattr("gridcast.cli.ingest_entsoe_actual_load", fake_ingest)

    exit_status = main(
        [
            "data",
            "entsoe",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output",
            str(tmp_path / "load.parquet"),
            "--report",
            str(tmp_path / "quality.json"),
            "--force",
        ]
    )

    assert exit_status == 0
    assert captured["start"] == datetime(2024, 1, 1, tzinfo=UTC)
    assert captured["end"] == datetime(2024, 1, 2, tzinfo=UTC)
    assert captured["force"] is True


def test_day_ahead_contract_command() -> None:
    assert (
        main(
            [
                "day-ahead",
                "contract",
                "--delivery-date",
                "2024-03-31",
            ]
        )
        == 0
    )
