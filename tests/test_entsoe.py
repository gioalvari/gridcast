import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from gridcast.columns import Col
from gridcast.entsoe import (
    ITALY_BIDDING_ZONE,
    download_actual_load_xml,
    get_entsoe_token,
    parse_actual_load_xml,
    prepare_actual_load,
)


def _load_document(
    start: str = "2024-01-01T00:00Z",
    resolution: str = "PT60M",
    quantities: list[float] | None = None,
) -> str:
    values = quantities or [100.0, 110.0, 120.0, 130.0]
    points = "".join(
        f"<Point><position>{position}</position><quantity>{value}</quantity></Point>"
        for position, value in enumerate(values, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <outBiddingZone_Domain.mRID>{ITALY_BIDDING_ZONE}</outBiddingZone_Domain.mRID>
    <Period>
      <timeInterval><start>{start}</start><end>2024-01-02T00:00Z</end></timeInterval>
      <resolution>{resolution}</resolution>
      {points}
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""


def test_token_prefers_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENTSOE_API_TOKEN", "environment-token")

    assert get_entsoe_token() == "environment-token"


def test_token_reports_missing_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    monkeypatch.setattr(
        "gridcast.entsoe.os.uname", lambda: type("U", (), {"sysname": "Linux"})()
    )

    with pytest.raises(RuntimeError, match="not configured"):
        get_entsoe_token()


def test_download_passes_token_only_through_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = _load_document().encode()
    captured: dict[str, object] = {}

    def fake_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        captured["args"] = args
        captured["input"] = kwargs["input"]
        return subprocess.CompletedProcess(args, 0, stdout=xml, stderr=b"")

    monkeypatch.setattr("gridcast.entsoe.get_entsoe_token", lambda: "secret-token")
    monkeypatch.setattr("gridcast.entsoe.subprocess.run", fake_run)
    destination = tmp_path / "load.xml"

    result = download_actual_load_xml(
        destination,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
        retries=1,
    )

    assert result == destination
    assert "secret-token" not in " ".join(captured["args"])
    assert b"secret-token" in bytes(captured["input"])
    assert destination.read_bytes() == xml


def test_download_validates_dates_retries_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "load.xml"
    destination.write_text("cached", encoding="utf-8")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    assert download_actual_load_xml(destination, start, start) == destination

    destination.unlink()
    with pytest.raises(ValueError, match="precede"):
        download_actual_load_xml(destination, start, start)
    with pytest.raises(ValueError, match="retries"):
        download_actual_load_xml(
            destination, start, datetime(2024, 1, 2, tzinfo=UTC), retries=0
        )

    monkeypatch.setattr("gridcast.entsoe.get_entsoe_token", lambda: "secret")
    monkeypatch.setattr(
        "gridcast.entsoe.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 22, b"html", b""),
    )
    with pytest.raises(RuntimeError, match="after retries"):
        download_actual_load_xml(
            destination, start, datetime(2024, 1, 2, tzinfo=UTC), retries=1
        )


def test_download_surfaces_api_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (
        b"<Acknowledgement_MarketDocument><Reason><text>invalid token</text>"
        b"</Reason></Acknowledgement_MarketDocument>"
    )
    monkeypatch.setattr("gridcast.entsoe.get_entsoe_token", lambda: "secret")
    monkeypatch.setattr(
        "gridcast.entsoe.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 22, payload, b""),
    )

    with pytest.raises(RuntimeError, match="invalid token"):
        download_actual_load_xml(
            tmp_path / "load.xml",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            retries=1,
        )


def test_parse_actual_load_supports_hourly_and_quarter_hour(tmp_path: Path) -> None:
    hourly_path = tmp_path / "hourly.xml"
    hourly_path.write_text(_load_document(), encoding="utf-8")
    quarter_path = tmp_path / "quarter.xml"
    quarter_path.write_text(_load_document(resolution="PT15M"), encoding="utf-8")

    hourly = parse_actual_load_xml(hourly_path)
    quarter = parse_actual_load_xml(quarter_path)

    assert hourly[Col.TARGET].tolist() == [100.0, 110.0, 120.0, 130.0]
    assert hourly[Col.TIMESTAMP].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert quarter[Col.TIMESTAMP].diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    assert str(hourly[Col.TIMESTAMP].dt.tz) == "UTC"


def test_prepare_actual_load_reports_quality(tmp_path: Path) -> None:
    source = tmp_path / "load.xml"
    source.write_text(_load_document(), encoding="utf-8")

    data, report = prepare_actual_load(
        [source],
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, 4),
    )

    assert len(data) == 4
    assert report.observations == 4
    assert report.resolution_minutes == 60
    assert report.missing_timestamps == 0
    assert report.area == ITALY_BIDDING_ZONE


def test_parser_rejects_acknowledgement_and_unsupported_resolution(
    tmp_path: Path,
) -> None:
    acknowledgement = tmp_path / "error.xml"
    acknowledgement.write_text(
        "<Acknowledgement_MarketDocument><Reason><text>bad token</text>"
        "</Reason></Acknowledgement_MarketDocument>",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="bad token"):
        parse_actual_load_xml(acknowledgement)

    unsupported = tmp_path / "unsupported.xml"
    unsupported.write_text(_load_document(resolution="P1D"), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        parse_actual_load_xml(unsupported)


def test_prepare_rejects_missing_intervals(tmp_path: Path) -> None:
    source = tmp_path / "load.xml"
    source.write_text(
        _load_document(quantities=[100.0, 110.0, 120.0]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing intervals"):
        prepare_actual_load(
            [source],
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 4, tzinfo=UTC),
        )
