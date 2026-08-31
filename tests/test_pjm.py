import json
import zipfile
from pathlib import Path
from types import TracebackType
from typing import Self

import pandas as pd
import pytest

from gridcast.columns import Col
from gridcast.pjm import (
    PJM_SOURCE_FILENAME,
    download_pjme_csv,
    ingest_pjme,
    prepare_pjme_data,
    validate_hourly_load,
)


def _write_source(path: Path, rows: list[tuple[str, object]]) -> None:
    pd.DataFrame(rows, columns=["Datetime", "PJME_MW"]).to_csv(path, index=False)


def test_prepare_pjme_regularizes_duplicates_and_short_gaps(tmp_path: Path) -> None:
    source_path = tmp_path / PJM_SOURCE_FILENAME
    _write_source(
        source_path,
        [
            ("2024-01-01 03:00:00", 130.0),
            ("2024-01-01 01:00:00", 100.0),
            ("2024-01-01 01:00:00", 120.0),
            ("2024-01-01 04:00:00", 140.0),
        ],
    )

    data, report = prepare_pjme_data(source_path)

    assert data[Col.TARGET].tolist() == [110.0, 120.0, 130.0, 140.0]
    assert report.source_rows == 4
    assert report.output_rows == 4
    assert report.duplicate_timestamps == 1
    assert report.missing_timestamps == 1
    assert report.imputed_observations == 1


def test_prepare_pjme_rejects_bad_sources(tmp_path: Path) -> None:
    source_path = tmp_path / PJM_SOURCE_FILENAME
    pd.DataFrame({"wrong": [1]}).to_csv(source_path, index=False)
    with pytest.raises(ValueError, match="missing required"):
        prepare_pjme_data(source_path)

    _write_source(source_path, [("invalid", 100.0)])
    with pytest.raises(ValueError, match="valid datetimes"):
        prepare_pjme_data(source_path)

    _write_source(source_path, [("2024-01-01", "invalid")])
    with pytest.raises(ValueError, match="numeric"):
        prepare_pjme_data(source_path)

    _write_source(source_path, [("2024-01-01", 0.0)])
    with pytest.raises(ValueError, match="positive"):
        prepare_pjme_data(source_path)

    _write_source(
        source_path,
        [("2024-01-01 00:00:00", 100.0), ("2024-01-01 05:00:00", 200.0)],
    )
    with pytest.raises(ValueError, match="4-hour gap"):
        prepare_pjme_data(source_path, max_gap_hours=3)
    with pytest.raises(ValueError, match="positive"):
        prepare_pjme_data(source_path, max_gap_hours=0)


def test_validate_hourly_load_rejects_irregular_data() -> None:
    data = pd.DataFrame(
        {
            Col.TIMESTAMP: pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 02:00"],
                format="%Y-%m-%d %H:%M",
            ),
            Col.TARGET: [100.0, 110.0],
        }
    )

    with pytest.raises(ValueError, match="regular hourly"):
        validate_hourly_load(data)


def test_download_extracts_expected_csv_and_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(PJM_SOURCE_FILENAME, "Datetime,PJME_MW\n2024-01-01,100\n")

    class Response:
        def __enter__(self) -> Self:
            self.file = archive_path.open("rb")
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            self.file.close()

        def read(self, size: int = -1) -> bytes:
            return self.file.read(size)

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    destination = tmp_path / "raw" / PJM_SOURCE_FILENAME

    assert download_pjme_csv(destination) == destination
    assert destination.read_text(encoding="utf-8").startswith("Datetime")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("cache should avoid download"),
    )
    assert download_pjme_csv(destination) == destination


def test_ingest_writes_parquet_and_quality_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / PJM_SOURCE_FILENAME
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h")
    _write_source(
        source_path,
        [
            (timestamp.isoformat(), 100.0 + index)
            for index, timestamp in enumerate(timestamps)
        ],
    )
    monkeypatch.setattr(
        "gridcast.pjm.download_pjme_csv", lambda destination, force=False: source_path
    )
    output_path = tmp_path / "processed" / "load.parquet"
    report_path = tmp_path / "artifacts" / "quality.json"

    report = ingest_pjme(source_path, output_path, report_path)

    assert report.output_rows == 4
    assert len(pd.read_parquet(output_path)) == 4
    assert json.loads(report_path.read_text(encoding="utf-8"))["output_rows"] == 4
