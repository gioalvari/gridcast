from datetime import UTC, date, datetime

import httpx
import pandas as pd
import pytest

from gridcast.columns import Col
from gridcast.contracts import ForecastContract
from gridcast.weather_runs import (
    ArchivedWeatherClient,
    validate_weather_run_coverage,
)


def _response(request: httpx.Request) -> httpx.Response:
    times = pd.date_range("2024-05-31T22:00:00Z", periods=24, freq="h")
    item = {
        "hourly": {
            "time": [timestamp.isoformat() for timestamp in times],
            "temperature_2m": list(range(24)),
        }
    }
    return httpx.Response(200, request=request, json=[item, item])


def test_archived_weather_client_returns_delivery_day_nodes() -> None:
    client = httpx.Client(transport=httpx.MockTransport(_response))
    contract = ForecastContract()
    delivery_date = date(2024, 6, 1)
    locations = {"rome": (41.9, 12.5), "milan": (45.5, 9.2)}

    weather = ArchivedWeatherClient(client).fetch_run(
        contract.weather_run(delivery_date),
        delivery_date,
        locations=locations,
    )
    counts = validate_weather_run_coverage(
        weather, contract, delivery_date, set(locations)
    )

    assert counts == {"milan": 24, "rome": 24}
    assert weather[Col.AVAILABLE_AT].le(contract.forecast_origin(delivery_date)).all()


def test_weather_coverage_rejects_future_and_incomplete_data() -> None:
    contract = ForecastContract()
    delivery_date = date(2024, 6, 1)
    intervals = contract.delivery_intervals(delivery_date)
    weather = pd.DataFrame(
        {
            Col.VALID_AT: intervals,
            Col.AVAILABLE_AT: [contract.forecast_origin(delivery_date)] * 24,
            Col.RUN_INITIALIZED_AT: [datetime(2024, 5, 31, tzinfo=UTC)] * 24,
            Col.LOCATION: ["rome"] * 24,
        }
    )
    with pytest.raises(ValueError, match="every delivery hour"):
        validate_weather_run_coverage(
            weather.iloc[:-1], contract, delivery_date, {"rome"}
        )

    weather[Col.AVAILABLE_AT] = contract.forecast_origin(delivery_date) + pd.Timedelta(
        minutes=1
    )
    with pytest.raises(ValueError, match="unavailable"):
        validate_weather_run_coverage(weather, contract, delivery_date, {"rome"})
