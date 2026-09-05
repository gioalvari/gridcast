import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.cli import main
from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, VALIDATION_SPLIT, Col
from gridcast.data import generate_synthetic_load
from gridcast.foundation_models import TIMESFM_2P5, TIMESFM_3


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
            "--holdout-folds",
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
    assert set(leaderboard["split"]) == {
        VALIDATION_SPLIT,
        HISTORICAL_HOLDOUT_SPLIT,
    }


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
            "--holdout-folds",
            "1",
            "--max-train-hours",
            str(24 * 380),
            "--n-estimators",
            "3",
        ]
    )

    assert exit_status == 0
    metrics = pd.read_csv(output_path / "metrics.csv")
    assert set(metrics["split"]) == {
        VALIDATION_SPLIT,
        HISTORICAL_HOLDOUT_SPLIT,
    }
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


def test_comparison_command_writes_paired_artifacts(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.parquet"
    timesfm25_path = tmp_path / "timesfm25.parquet"
    timesfm3_path = tmp_path / "timesfm3.parquet"
    output_path = tmp_path / "comparison"
    models = {
        "seasonal_naive_24h": 4.0,
        "lightgbm_exogenous": 3.0,
        TIMESFM_2P5.model_name: 2.0,
        TIMESFM_3.model_name: 1.0,
    }

    def frame(model: str, error: float) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for fold in range(1, 5):
            cutoff = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=2 * fold)
            for step in range(2):
                rows.append(
                    {
                        Col.TIMESTAMP: cutoff + pd.Timedelta(hours=step + 1),
                        Col.TARGET: 100.0,
                        Col.PREDICTION: 100.0 - error,
                        Col.MODEL: model,
                        Col.SPLIT: HISTORICAL_HOLDOUT_SPLIT,
                        Col.FOLD: fold,
                        Col.CUTOFF: cutoff,
                    }
                )
        return pd.DataFrame(rows)

    pd.concat(
        [frame(model, models[model]) for model in models if "timesfm" not in model]
    ).to_parquet(benchmark_path, index=False)
    frame(TIMESFM_2P5.model_name, 2.0).to_parquet(timesfm25_path, index=False)
    frame(TIMESFM_3.model_name, 1.0).to_parquet(timesfm3_path, index=False)

    exit_status = main(
        [
            "comparison",
            "--benchmark",
            str(benchmark_path),
            "--timesfm25",
            str(timesfm25_path),
            "--timesfm3",
            str(timesfm3_path),
            "--output-dir",
            str(output_path),
            "--bootstrap-replicates",
            "100",
            "--block-length-folds",
            "2",
            "--expected-folds",
            "4",
            "--observations-per-fold",
            "2",
        ]
    )

    assert exit_status == 0
    comparisons = pd.read_csv(output_path / "paired_comparisons.csv")
    assert len(comparisons) == 6
    assert comparisons["mean_mae_improvement_mw"].gt(0.0).all()


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
