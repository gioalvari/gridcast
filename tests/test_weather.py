import json
from pathlib import Path

import pandas as pd
import pytest

from gridcast.columns import Col
from gridcast.weather import (
    ingest_temperature,
    prepare_temperature_data,
    validate_temperature_data,
)


def _write_weather(path: Path, periods: int = 4) -> None:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")
    payload = {
        "hourly": {
            "time": [timestamp.isoformat() for timestamp in timestamps],
            "temperature_2m": [float(index) for index in range(periods)],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_temperature_data_parses_hourly_response(tmp_path: Path) -> None:
    source = tmp_path / "weather.json"
    _write_weather(source)

    data, report = prepare_temperature_data(source)

    assert list(data.columns) == [Col.TIMESTAMP, Col.TEMPERATURE]
    assert data[Col.TEMPERATURE].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert report.observations == 4
    assert report.missing_temperatures == 0
    assert report.model == "ERA5"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"hourly": {}},
        {"hourly": {"time": ["2024-01-01"], "temperature_2m": []}},
        {"hourly": {"time": ["invalid"], "temperature_2m": [1.0]}},
        {"hourly": {"time": ["2024-01-01"], "temperature_2m": [None]}},
    ],
)
def test_prepare_temperature_data_rejects_invalid_payloads(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    source = tmp_path / "weather.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        prepare_temperature_data(source)


def test_validate_temperature_rejects_irregular_frequency() -> None:
    data = pd.DataFrame(
        {
            Col.TIMESTAMP: pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"]),
            Col.TEMPERATURE: [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="regular hourly"):
        validate_temperature_data(data)


def test_ingest_temperature_writes_parquet_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "weather.json"
    _write_weather(source)
    monkeypatch.setattr(
        "gridcast.weather.download_temperature_json",
        lambda destination, force=False: source,
    )
    output = tmp_path / "weather.parquet"
    report_path = tmp_path / "quality.json"

    report = ingest_temperature(source, output, report_path)

    assert report.observations == 4
    assert len(pd.read_parquet(output)) == 4
    assert json.loads(report_path.read_text())["model"] == "ERA5"
