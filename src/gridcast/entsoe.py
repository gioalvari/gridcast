import json
import os
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.columns import Col

ENTSOE_API_URL = "https://web-api.tp.entsoe.eu/api"
ENTSOE_TOKEN_ENV = "ENTSOE_API_TOKEN"
ENTSOE_KEYCHAIN_SERVICE = "gridcast-entsoe-api-token"
ITALY_BIDDING_ZONE = "10YIT-GRTN-----B"
ACTUAL_TOTAL_LOAD_DOCUMENT = "A65"
REALIZED_PROCESS = "A16"
_CURL_METADATA_DELIMITER = b"\nGRIDCAST_CURL_METADATA:"


@dataclass(frozen=True)
class EntsoeQualityReport:
    """Quality and provenance fields for ENTSO-E actual total load.

    Parameters
    ----------
    observations : int
        Number of normalized observations.
    duplicate_timestamps : int
        Duplicate source timestamps averaged during normalization.
    missing_timestamps : int
        Missing intervals in the requested range.
    resolution_minutes : int
        Normalized data resolution in minutes.
    start : str
        First normalized UTC timestamp.
    end : str
        Last normalized UTC timestamp.
    area : str
        ENTSO-E bidding-zone EIC code.
    """

    observations: int
    duplicate_timestamps: int
    missing_timestamps: int
    resolution_minutes: int
    start: str
    end: str
    area: str

    def to_dict(self) -> dict[str, int | str]:
        """Serialize the report to JSON-compatible values."""
        return asdict(self)


def get_entsoe_token() -> str:
    """Load the ENTSO-E token from environment or macOS Keychain.

    Returns
    -------
    str
        Non-empty ENTSO-E security token.

    Raises
    ------
    RuntimeError
        If no token is configured.
    """
    environment_token = os.environ.get(ENTSOE_TOKEN_ENV, "").strip()
    if environment_token:
        return environment_token
    if os.uname().sysname == "Darwin":
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                os.environ.get("USER", ""),
                "-s",
                ENTSOE_KEYCHAIN_SERVICE,
                "-w",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        keychain_token = result.stdout.strip()
        if result.returncode == 0 and keychain_token:
            return keychain_token
    msg = (
        f"ENTSO-E token not configured; set {ENTSOE_TOKEN_ENV} or add the "
        f"macOS Keychain service {ENTSOE_KEYCHAIN_SERVICE}"
    )
    raise RuntimeError(msg)


def download_actual_load_xml(
    destination: Path,
    start: datetime,
    end: datetime,
    *,
    area: str = ITALY_BIDDING_ZONE,
    force: bool = False,
    retries: int = 3,
) -> Path:
    """Download one ENTSO-E actual-total-load XML document securely.

    The authenticated URL is passed to curl through stdin rather than process
    arguments, preventing the token from appearing in process listings.

    Parameters
    ----------
    destination : pathlib.Path
        Cache path for the XML response.
    start : datetime
        Inclusive UTC period start.
    end : datetime
        Exclusive UTC period end.
    area : str, default=ITALY_BIDDING_ZONE
        ENTSO-E bidding-zone EIC code.
    force : bool, default=False
        Replace an existing cache when true.
    retries : int, default=3
        Maximum attempts for transient server errors.

    Returns
    -------
    pathlib.Path
        Cached XML path.
    """
    start = _as_utc(start)
    end = _as_utc(end)
    if destination.exists() and not force:
        return destination
    if start >= end:
        msg = "ENTSO-E period start must precede period end"
        raise ValueError(msg)
    if retries < 1:
        msg = "retries must be positive"
        raise ValueError(msg)

    token = get_entsoe_token()
    parameters = {
        "securityToken": token,
        "documentType": ACTUAL_TOTAL_LOAD_DOCUMENT,
        "processType": REALIZED_PROCESS,
        "outBiddingZone_Domain": area,
        "periodStart": _api_timestamp(start),
        "periodEnd": _api_timestamp(end),
    }
    authenticated_url = f"{ENTSOE_API_URL}?{urllib.parse.urlencode(parameters)}"
    curl_config = f'url = "{authenticated_url}"\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    status_code = "unknown"
    content_type = "unknown"
    try:
        for attempt in range(1, retries + 1):
            result = subprocess.run(
                [
                    "curl",
                    "--silent",
                    "--show-error",
                    "--fail-with-body",
                    "--retry",
                    "0",
                    "--write-out",
                    "\nGRIDCAST_CURL_METADATA:%{http_code}|%{content_type}",
                    "--config",
                    "-",
                ],
                input=curl_config.encode(),
                capture_output=True,
                check=False,
            )
            payload, status_code, content_type = _split_curl_response(result.stdout)
            _raise_api_acknowledgement(payload)
            if result.returncode == 0:
                destination.write_bytes(payload)
                return destination
            if attempt < retries:
                time.sleep(float(2 ** (attempt - 1)))
        media_type = content_type or "unknown"
        msg = (
            f"ENTSO-E request failed after retries (HTTP {status_code}, "
            f"content type {media_type})"
        )
        raise RuntimeError(msg)
    finally:
        token = authenticated_url = curl_config = ""


def parse_actual_load_xml(source: Path) -> pd.DataFrame:
    """Parse an ENTSO-E load document into timestamped megawatt values.

    Parameters
    ----------
    source : pathlib.Path
        ENTSO-E XML response.

    Returns
    -------
    pandas.DataFrame
        UTC timestamps, total load in MW, and bidding-zone code.

    Raises
    ------
    ValueError
        If no supported time-series points are present.
    """
    payload = source.read_bytes()
    root = ET.fromstring(payload)
    _raise_api_acknowledgement(payload)
    rows: list[dict[str, object]] = []
    for series in _children(root, "TimeSeries"):
        area = _first_text(series, "outBiddingZone_Domain.mRID") or ITALY_BIDDING_ZONE
        for period in _children(series, "Period"):
            start_text = _nested_text(period, ["timeInterval", "start"])
            resolution_text = _first_text(period, "resolution")
            if start_text is None or resolution_text is None:
                continue
            period_start = pd.Timestamp(start_text)
            resolution = _parse_resolution(resolution_text)
            for point in _children(period, "Point"):
                position_text = _first_text(point, "position")
                quantity_text = _first_text(point, "quantity")
                if position_text is None or quantity_text is None:
                    continue
                position = int(position_text)
                rows.append(
                    {
                        Col.TIMESTAMP: period_start + (position - 1) * resolution,
                        Col.TARGET: float(quantity_text),
                        Col.AREA: area,
                    }
                )
    if not rows:
        msg = "ENTSO-E response does not contain actual-load points"
        raise ValueError(msg)
    data = pd.DataFrame(rows).sort_values(Col.TIMESTAMP, ignore_index=True)
    data[Col.TIMESTAMP] = pd.to_datetime(data[Col.TIMESTAMP], utc=True)
    return data


def prepare_actual_load(
    sources: list[Path],
    start: datetime,
    end: datetime,
    *,
    area: str = ITALY_BIDDING_ZONE,
) -> tuple[pd.DataFrame, EntsoeQualityReport]:
    """Combine cached ENTSO-E XML chunks and validate regular resolution."""
    start = _as_utc(start)
    end = _as_utc(end)
    parsed = pd.concat([parse_actual_load_xml(source) for source in sources])
    parsed = parsed.loc[
        parsed[Col.TIMESTAMP].ge(pd.Timestamp(start))
        & parsed[Col.TIMESTAMP].lt(pd.Timestamp(end))
    ]
    duplicate_timestamps = int(parsed[Col.TIMESTAMP].duplicated().sum())
    normalized = parsed.groupby([Col.TIMESTAMP, Col.AREA], as_index=False).agg(
        {Col.TARGET: "mean"}
    )
    normalized = normalized.sort_values(Col.TIMESTAMP, ignore_index=True)
    differences = normalized[Col.TIMESTAMP].diff().dropna()
    if differences.empty:
        msg = "ENTSO-E data require at least two observations"
        raise ValueError(msg)
    resolution = differences.mode().iloc[0]
    expected_index = pd.date_range(
        start=start, end=end, freq=resolution, inclusive="left"
    )
    observed_index = pd.DatetimeIndex(normalized[Col.TIMESTAMP])
    missing_timestamps = len(expected_index.difference(observed_index))
    if missing_timestamps:
        msg = f"ENTSO-E data contain {missing_timestamps} missing intervals"
        raise ValueError(msg)
    target = normalized[Col.TARGET].to_numpy(dtype=float)
    if not np.isfinite(target).all() or np.any(target <= 0.0):
        msg = "ENTSO-E load values must be finite and positive"
        raise ValueError(msg)
    report = EntsoeQualityReport(
        observations=len(normalized),
        duplicate_timestamps=duplicate_timestamps,
        missing_timestamps=missing_timestamps,
        resolution_minutes=int(resolution / pd.Timedelta(minutes=1)),
        start=normalized[Col.TIMESTAMP].iloc[0].isoformat(),
        end=normalized[Col.TIMESTAMP].iloc[-1].isoformat(),
        area=area,
    )
    return normalized, report


def ingest_entsoe_actual_load(
    raw_directory: Path,
    output_path: Path,
    report_path: Path,
    start: datetime,
    end: datetime,
    *,
    area: str = ITALY_BIDDING_ZONE,
    force: bool = False,
) -> EntsoeQualityReport:
    """Download monthly ENTSO-E chunks and cache normalized load as Parquet."""
    start = _as_utc(start)
    end = _as_utc(end)
    sources: list[Path] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(_next_month(chunk_start), end)
        destination = raw_directory / (
            f"actual_load_{area}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.xml"
        )
        sources.append(
            download_actual_load_xml(
                destination,
                chunk_start,
                chunk_end,
                area=area,
                force=force,
            )
        )
        chunk_start = chunk_end
    data, report = prepare_actual_load(sources, start, end, area=area)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False, compression="snappy")
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report


def _api_timestamp(value: datetime) -> str:
    return _as_utc(value).strftime("%Y%m%d%H%M")


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


def _parse_resolution(value: str) -> pd.Timedelta:
    supported = {
        "PT15M": timedelta(minutes=15),
        "PT30M": timedelta(minutes=30),
        "PT60M": timedelta(hours=1),
        "PT1H": timedelta(hours=1),
    }
    if value not in supported:
        msg = f"unsupported ENTSO-E resolution: {value}"
        raise ValueError(msg)
    return pd.Timedelta(supported[value])


def _raise_api_acknowledgement(payload: bytes) -> None:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return
    if _local_name(root.tag) != "Acknowledgement_MarketDocument":
        return
    reason = next(
        (element.text for element in root.iter() if _local_name(element.tag) == "text"),
        "unspecified API error",
    )
    raise RuntimeError(f"ENTSO-E API rejected the request: {reason}")


def _split_curl_response(payload: bytes) -> tuple[bytes, str, str]:
    if _CURL_METADATA_DELIMITER not in payload:
        return payload, "unknown", "unknown"
    body, metadata = payload.rsplit(_CURL_METADATA_DELIMITER, 1)
    status_code, _, content_type = metadata.decode(errors="replace").partition("|")
    return body, status_code or "unknown", content_type or "unknown"


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if _local_name(child.tag) == name]


def _first_text(element: ET.Element, name: str) -> str | None:
    return next(
        (child.text for child in element.iter() if _local_name(child.tag) == name),
        None,
    )


def _nested_text(element: ET.Element, path: list[str]) -> str | None:
    current = element
    for name in path:
        match = next(
            (child for child in current if _local_name(child.tag) == name),
            None,
        )
        if match is None:
            return None
        current = match
    return current.text


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
