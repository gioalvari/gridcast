import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.columns import Col

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PHILADELPHIA_LATITUDE = 39.9526
PHILADELPHIA_LONGITUDE = -75.1652
WEATHER_START_DATE = "2002-01-01"
WEATHER_END_DATE = "2018-08-03"


@dataclass(frozen=True)
class WeatherQualityReport:
    """Quality and provenance fields for the temperature dataset.

    Parameters
    ----------
    observations : int
        Number of normalized hourly observations.
    missing_temperatures : int
        Missing temperatures in the API payload.
    start : str
        First weather timestamp.
    end : str
        Last weather timestamp.
    latitude : float
        Requested latitude.
    longitude : float
        Requested longitude.
    model : str
        Weather reanalysis model.
    timezone : str
        Timezone requested from the API.
    """

    observations: int
    missing_temperatures: int
    start: str
    end: str
    latitude: float
    longitude: float
    model: str
    timezone: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialize the report to JSON-compatible values.

        Returns
        -------
        dict
            Quality and provenance fields.
        """
        return asdict(self)


def download_temperature_json(
    destination: Path,
    *,
    force: bool = False,
    start_date: str = WEATHER_START_DATE,
    end_date: str = WEATHER_END_DATE,
) -> Path:
    """Download hourly ERA5 temperature for Philadelphia from Open-Meteo.

    Parameters
    ----------
    destination : pathlib.Path
        Cache path for the API JSON response.
    force : bool, default=False
        Replace an existing cache when true.
    start_date : str, default="2002-01-01"
        First requested date.
    end_date : str, default="2018-08-03"
        Last requested date.

    Returns
    -------
    pathlib.Path
        Path to the cached JSON response.
    """
    if destination.exists() and not force:
        return destination

    query = urllib.parse.urlencode(
        {
            "latitude": PHILADELPHIA_LATITUDE,
            "longitude": PHILADELPHIA_LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m",
            "timezone": "America/New_York",
            "models": "era5",
        }
    )
    request = urllib.request.Request(
        f"{OPEN_METEO_ARCHIVE_URL}?{query}",
        headers={"User-Agent": "gridcast/0.1"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        payload = response.read()
    destination.write_bytes(payload)
    return destination


def prepare_temperature_data(
    source_path: Path,
) -> tuple[pd.DataFrame, WeatherQualityReport]:
    """Parse and validate an Open-Meteo hourly temperature response.

    Parameters
    ----------
    source_path : pathlib.Path
        Cached Open-Meteo JSON response.

    Returns
    -------
    tuple
        Canonical hourly weather data and its quality report.

    Raises
    ------
    ValueError
        If the response is incomplete or temporally irregular.
    """
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        msg = "weather response must contain an hourly object"
        raise ValueError(msg)
    raw_timestamps = hourly.get("time")
    raw_temperatures = hourly.get("temperature_2m")
    if not isinstance(raw_timestamps, list) or not isinstance(raw_temperatures, list):
        msg = "weather response must contain time and temperature_2m arrays"
        raise ValueError(msg)
    if len(raw_timestamps) != len(raw_temperatures) or not raw_timestamps:
        msg = "weather time and temperature arrays must have equal non-zero lengths"
        raise ValueError(msg)

    timestamps = pd.to_datetime(raw_timestamps, errors="coerce")
    temperatures = pd.to_numeric(pd.Series(raw_temperatures), errors="coerce")
    missing_temperatures = int(temperatures.isna().sum())
    if np.any(timestamps.isna()):
        msg = "weather timestamps must all be valid datetimes"
        raise ValueError(msg)
    if missing_temperatures:
        msg = "weather temperatures must all be numeric and non-null"
        raise ValueError(msg)
    data = pd.DataFrame(
        {Col.TIMESTAMP: timestamps, Col.TEMPERATURE: temperatures.astype(float)}
    )
    validate_temperature_data(data)
    report = WeatherQualityReport(
        observations=len(data),
        missing_temperatures=missing_temperatures,
        start=data[Col.TIMESTAMP].iloc[0].isoformat(),
        end=data[Col.TIMESTAMP].iloc[-1].isoformat(),
        latitude=PHILADELPHIA_LATITUDE,
        longitude=PHILADELPHIA_LONGITUDE,
        model="ERA5",
        timezone="America/New_York",
    )
    return data, report


def ingest_temperature(
    raw_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    force: bool = False,
) -> WeatherQualityReport:
    """Download, validate, and cache hourly temperature as Parquet.

    Parameters
    ----------
    raw_path : pathlib.Path
        Cache location for the API response.
    output_path : pathlib.Path
        Destination for normalized Parquet weather data.
    report_path : pathlib.Path
        Destination for the JSON quality report.
    force : bool, default=False
        Replace the cached API response when true.

    Returns
    -------
    WeatherQualityReport
        Quality and provenance statistics.
    """
    source_path = download_temperature_json(raw_path, force=force)
    data, report = prepare_temperature_data(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False, compression="snappy")
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report


def validate_temperature_data(data: pd.DataFrame) -> None:
    """Validate canonical hourly temperature data.

    Parameters
    ----------
    data : pandas.DataFrame
        Candidate weather dataset.

    Raises
    ------
    ValueError
        If schema, timestamps, frequency, or temperatures are invalid.
    """
    required = {Col.TIMESTAMP, Col.TEMPERATURE}
    missing = required.difference(data.columns)
    if missing:
        names = ", ".join(sorted(missing))
        msg = f"weather data are missing required columns: {names}"
        raise ValueError(msg)
    timestamps = data[Col.TIMESTAMP]
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        msg = f"{Col.TIMESTAMP} must contain datetimes"
        raise ValueError(msg)
    if timestamps.isna().any() or timestamps.duplicated().any():
        msg = "weather timestamps must be unique and non-null"
        raise ValueError(msg)
    if not timestamps.is_monotonic_increasing:
        msg = "weather timestamps must be sorted in increasing order"
        raise ValueError(msg)
    if len(data) > 1 and not timestamps.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        msg = "weather timestamps must have a regular hourly frequency"
        raise ValueError(msg)
    temperature = pd.to_numeric(data[Col.TEMPERATURE], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(temperature).all():
        msg = f"{Col.TEMPERATURE} must contain only finite values"
        raise ValueError(msg)
