import json
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.columns import Col

PJM_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "robikscube/hourly-energy-consumption?datasetVersionNumber=3"
)
PJM_SOURCE_FILENAME = "PJME_hourly.csv"
PJM_SOURCE_TIMESTAMP = "Datetime"
PJM_SOURCE_TARGET = "PJME_MW"


@dataclass(frozen=True)
class DataQualityReport:
    """Quality statistics collected while normalizing PJM load data.

    Parameters
    ----------
    source_rows : int
        Number of rows in the source CSV.
    output_rows : int
        Number of rows in the regularized hourly dataset.
    duplicate_timestamps : int
        Extra source rows sharing a timestamp.
    missing_timestamps : int
        Hours absent between the first and last timestamp.
    missing_targets : int
        Source rows with an invalid or absent load value.
    imputed_observations : int
        Values interpolated during hourly regularization.
    non_positive_targets : int
        Source load values at or below zero.
    start : str
        Earliest timestamp in the normalized dataset.
    end : str
        Latest timestamp in the normalized dataset.
    """

    source_rows: int
    output_rows: int
    duplicate_timestamps: int
    missing_timestamps: int
    missing_targets: int
    imputed_observations: int
    non_positive_targets: int
    start: str
    end: str

    def to_dict(self) -> dict[str, int | str]:
        """Serialize the report to JSON-compatible values.

        Returns
        -------
        dict
            Data-quality fields and values.
        """
        return asdict(self)


def download_pjme_csv(
    destination: Path,
    *,
    force: bool = False,
    url: str = PJM_DATASET_URL,
) -> Path:
    """Download the public PJME CSV from the Kaggle dataset archive.

    Parameters
    ----------
    destination : pathlib.Path
        Local path for the extracted source CSV.
    force : bool, default=False
        Replace an existing source file when true.
    url : str, default=PJM_DATASET_URL
        Dataset archive URL. Primarily configurable for testing.

    Returns
    -------
    pathlib.Path
        Path to the downloaded CSV.

    Raises
    ------
    ValueError
        If the expected PJME file is absent from the archive.
    """
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "gridcast/0.1"})
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp_directory:
        archive_path = Path(temp_directory) / "pjm.zip"
        with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
            with archive_path.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file)

        with zipfile.ZipFile(archive_path) as archive:
            matching_names = [
                name
                for name in archive.namelist()
                if Path(name).name == PJM_SOURCE_FILENAME
            ]
            if len(matching_names) != 1:
                msg = f"archive must contain exactly one {PJM_SOURCE_FILENAME}"
                raise ValueError(msg)
            temporary_csv = Path(temp_directory) / PJM_SOURCE_FILENAME
            with archive.open(matching_names[0]) as source:
                with temporary_csv.open("wb") as output:
                    shutil.copyfileobj(source, output)
            temporary_csv.replace(destination)
    return destination


def prepare_pjme_data(
    source_path: Path,
    *,
    max_gap_hours: int = 3,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Parse and regularize the public PJME hourly load dataset.

    Duplicate wall-clock timestamps are averaged. Missing observations are
    linearly interpolated only when each consecutive gap is no longer than
    ``max_gap_hours``.

    Parameters
    ----------
    source_path : pathlib.Path
        Path to the source ``PJME_hourly.csv`` file.
    max_gap_hours : int, default=3
        Largest consecutive missing interval that may be interpolated.

    Returns
    -------
    tuple
        Normalized hourly data and its quality report.

    Raises
    ------
    ValueError
        If columns, values, or temporal gaps do not satisfy quality rules.
    """
    if max_gap_hours < 1:
        msg = "max_gap_hours must be positive"
        raise ValueError(msg)

    source = pd.read_csv(source_path)
    required = {PJM_SOURCE_TIMESTAMP, PJM_SOURCE_TARGET}
    missing_columns = required.difference(source.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        msg = f"source data are missing required columns: {names}"
        raise ValueError(msg)

    timestamps = pd.to_datetime(source[PJM_SOURCE_TIMESTAMP], errors="coerce")
    targets = pd.to_numeric(source[PJM_SOURCE_TARGET], errors="coerce")
    missing_targets = int(targets.isna().sum())
    if timestamps.isna().any():
        msg = "source timestamps must all be valid datetimes"
        raise ValueError(msg)
    if missing_targets:
        msg = "source load values must all be numeric and non-null"
        raise ValueError(msg)

    non_positive_targets = int(targets.le(0.0).sum())
    if non_positive_targets:
        msg = "source load values must all be positive"
        raise ValueError(msg)

    parsed = pd.DataFrame({Col.TIMESTAMP: timestamps, Col.TARGET: targets})
    parsed = parsed.sort_values(Col.TIMESTAMP, kind="stable")
    duplicate_timestamps = int(parsed[Col.TIMESTAMP].duplicated().sum())
    deduplicated = parsed.groupby(Col.TIMESTAMP, as_index=False)[Col.TARGET].mean()
    full_index = pd.date_range(
        start=deduplicated[Col.TIMESTAMP].iloc[0],
        end=deduplicated[Col.TIMESTAMP].iloc[-1],
        freq="h",
    )
    regular = deduplicated.set_index(Col.TIMESTAMP).reindex(full_index)
    missing_timestamps = int(regular[Col.TARGET].isna().sum())
    longest_gap = _longest_missing_run(regular[Col.TARGET].isna().to_numpy())
    if longest_gap > max_gap_hours:
        msg = (
            f"source contains a {longest_gap}-hour gap; "
            f"maximum allowed is {max_gap_hours}"
        )
        raise ValueError(msg)

    regular[Col.TARGET] = regular[Col.TARGET].interpolate(
        method="time", limit=max_gap_hours, limit_area="inside"
    )
    if regular[Col.TARGET].isna().any():
        msg = "hourly regularization left missing load values"
        raise ValueError(msg)
    normalized = regular.rename_axis(Col.TIMESTAMP).reset_index()
    validate_hourly_load(normalized)
    report = DataQualityReport(
        source_rows=len(source),
        output_rows=len(normalized),
        duplicate_timestamps=duplicate_timestamps,
        missing_timestamps=missing_timestamps,
        missing_targets=missing_targets,
        imputed_observations=missing_timestamps,
        non_positive_targets=non_positive_targets,
        start=normalized[Col.TIMESTAMP].iloc[0].isoformat(),
        end=normalized[Col.TIMESTAMP].iloc[-1].isoformat(),
    )
    return normalized, report


def ingest_pjme(
    raw_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    force: bool = False,
) -> DataQualityReport:
    """Download, normalize, and cache PJME load data as Parquet.

    Parameters
    ----------
    raw_path : pathlib.Path
        Cache location for the original CSV.
    output_path : pathlib.Path
        Destination for normalized Parquet data.
    report_path : pathlib.Path
        Destination for the JSON quality report.
    force : bool, default=False
        Replace the cached source download when true.

    Returns
    -------
    DataQualityReport
        Quality statistics from the ingestion run.
    """
    source_path = download_pjme_csv(raw_path, force=force)
    data, report = prepare_pjme_data(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False, compression="snappy")
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report


def validate_hourly_load(data: pd.DataFrame) -> None:
    """Validate the canonical GridCast hourly load schema.

    Parameters
    ----------
    data : pandas.DataFrame
        Candidate dataset with timestamp and load columns.

    Raises
    ------
    ValueError
        If schema, timestamps, frequency, or values are invalid.
    """
    required = {Col.TIMESTAMP, Col.TARGET}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        msg = f"data are missing required columns: {names}"
        raise ValueError(msg)
    timestamps = data[Col.TIMESTAMP]
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        msg = f"{Col.TIMESTAMP} must contain datetimes"
        raise ValueError(msg)
    if timestamps.isna().any() or timestamps.duplicated().any():
        msg = "timestamps must be unique and non-null"
        raise ValueError(msg)
    if not timestamps.is_monotonic_increasing:
        msg = "timestamps must be sorted in increasing order"
        raise ValueError(msg)
    if len(data) > 1 and not timestamps.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        msg = "timestamps must have a regular hourly frequency"
        raise ValueError(msg)
    target = pd.to_numeric(data[Col.TARGET], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all() or np.any(target <= 0.0):
        msg = f"{Col.TARGET} must contain only finite positive values"
        raise ValueError(msg)


def _longest_missing_run(missing: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in missing:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest
