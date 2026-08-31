from datetime import date
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from gridcast.columns import Col
from gridcast.contracts import ForecastContract, WeatherRun

OPEN_METEO_SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
ITALIAN_WEATHER_NODES: dict[str, tuple[float, float]] = {
    "bari": (41.1171, 16.8719),
    "bologna": (44.4949, 11.3426),
    "cagliari": (39.2238, 9.1217),
    "milan": (45.4642, 9.1900),
    "naples": (40.8518, 14.2681),
    "palermo": (38.1157, 13.3615),
    "rome": (41.9028, 12.4964),
    "turin": (45.0703, 7.6869),
}
WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "wind_speed_10m",
    "cloud_cover",
)


class ArchivedWeatherClient:
    """Client for immutable Open-Meteo numerical-weather-model runs.

    Parameters
    ----------
    client : httpx.Client, optional
        Injected synchronous HTTP client, primarily for testing.
    """

    def __init__(self, client: httpx.Client | None = None) -> None:
        """Initialize the archived weather client."""
        self._client = client or httpx.Client(timeout=60.0)

    def fetch_run(
        self,
        run: WeatherRun,
        delivery_date: date,
        *,
        locations: dict[str, tuple[float, float]] | None = None,
        timezone_name: str = "Europe/Rome",
    ) -> pd.DataFrame:
        """Fetch one archived run for each configured weather node.

        Parameters
        ----------
        run : WeatherRun
            Immutable model vintage and availability metadata.
        delivery_date : date
            Local market delivery date.
        locations : dict, optional
            Named latitude/longitude nodes.
        timezone_name : str, default="Europe/Rome"
            Market timezone used to select the delivery day.

        Returns
        -------
        pandas.DataFrame
            Hourly weather features with validity and availability timestamps.
        """
        nodes = locations or ITALIAN_WEATHER_NODES
        parameters = {
            "latitude": ",".join(str(nodes[name][0]) for name in nodes),
            "longitude": ",".join(str(nodes[name][1]) for name in nodes),
            "run": run.initialized_at.strftime("%Y-%m-%dT%H:%M"),
            "hourly": ",".join(WEATHER_VARIABLES),
            "models": run.model,
            "timezone": "GMT",
        }
        response = self._client.get(
            OPEN_METEO_SINGLE_RUN_URL,
            params=parameters,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload if isinstance(payload, list) else [payload]
        if len(items) != len(nodes):
            msg = "Open-Meteo returned an unexpected number of locations"
            raise ValueError(msg)
        frames: list[pd.DataFrame] = []
        for location, item in zip(nodes, items, strict=True):
            if not isinstance(item, dict):
                msg = f"invalid Open-Meteo response for {location}"
                raise ValueError(msg)
            hourly = item.get("hourly")
            if not isinstance(hourly, dict) or "time" not in hourly:
                msg = f"missing hourly weather data for {location}"
                raise ValueError(msg)
            frame = pd.DataFrame(hourly)
            frame[Col.VALID_AT] = pd.to_datetime(frame.pop("time"), utc=True)
            local_date = (
                frame[Col.VALID_AT].dt.tz_convert(ZoneInfo(timezone_name)).dt.date
            )
            frame = frame.loc[local_date.eq(delivery_date)].copy()
            if frame.empty:
                msg = f"model run does not cover {delivery_date} for {location}"
                raise ValueError(msg)
            frame[Col.LOCATION] = location
            frame[Col.RUN_INITIALIZED_AT] = run.initialized_at
            frame[Col.AVAILABLE_AT] = run.available_at
            frame[Col.MODEL] = run.model
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def validate_weather_run_coverage(
    weather: pd.DataFrame,
    contract: ForecastContract,
    delivery_date: date,
    locations: set[str] | None = None,
) -> dict[str, int]:
    """Validate node coverage and feature availability for a delivery day.

    Parameters
    ----------
    weather : pandas.DataFrame
        Archived weather observations returned by ``ArchivedWeatherClient``.
    contract : ForecastContract
        Operational day-ahead contract.
    delivery_date : date
        Local delivery date.
    locations : set of str, optional
        Expected node names.

    Returns
    -------
    dict
        Hour counts keyed by weather node.
    """
    expected_locations = locations or set(ITALIAN_WEATHER_NODES)
    required = {
        Col.VALID_AT,
        Col.AVAILABLE_AT,
        Col.LOCATION,
        Col.RUN_INITIALIZED_AT,
    }
    missing = required.difference(weather.columns)
    if missing:
        names = ", ".join(sorted(missing))
        msg = f"weather run is missing required columns: {names}"
        raise ValueError(msg)
    origin = contract.forecast_origin(delivery_date)
    if weather[Col.AVAILABLE_AT].gt(origin).any():
        msg = "weather run contains values unavailable at forecast origin"
        raise ValueError(msg)
    counts = {
        str(location): int(count)
        for location, count in weather.groupby(Col.LOCATION)[Col.VALID_AT]
        .count()
        .items()
    }
    expected_hours = len(contract.delivery_intervals(delivery_date))
    if set(counts) != expected_locations or any(
        count != expected_hours for count in counts.values()
    ):
        msg = "weather run does not cover every delivery hour and location"
        raise ValueError(msg)
    return counts
