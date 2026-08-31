import json
from pathlib import Path

from gridcast.data import generate_synthetic_load
from gridcast.eda import create_eda_report


def test_create_eda_report_writes_tables_and_plots(tmp_path: Path) -> None:
    data = generate_synthetic_load(periods=24 * 14)

    summary = create_eda_report(data, tmp_path)

    assert summary["observations"] == 24 * 14
    assert json.loads((tmp_path / "summary.json").read_text())["load_mw"]["maximum"] > 0
    assert {path.name for path in tmp_path.iterdir()} == {
        "summary.json",
        "daily_load.csv",
        "hourly_profile.csv",
        "weekly_profile.csv",
        "load_history.png",
        "seasonal_profiles.png",
    }
