from datetime import UTC, date, datetime, time, timedelta

import pytest
from pydantic import ValidationError

from gridcast.contracts import (
    FeatureObservation,
    ForecastContract,
    LeakageError,
    WeatherRun,
)


def test_contract_builds_dst_safe_delivery_days() -> None:
    contract = ForecastContract()

    assert len(contract.delivery_intervals(date(2024, 3, 31))) == 23
    assert len(contract.delivery_intervals(date(2024, 10, 27))) == 25
    assert len(contract.delivery_intervals(date(2024, 6, 1))) == 24
    assert contract.forecast_origin(date(2024, 6, 1)) == datetime(
        2024, 5, 31, 8, tzinfo=UTC
    )


def test_contract_selects_weather_run_available_before_origin() -> None:
    contract = ForecastContract()
    delivery_date = date(2024, 6, 1)
    run = contract.weather_run(delivery_date)

    assert run.initialized_at == datetime(2024, 5, 31, 0, tzinfo=UTC)
    assert run.available_at == datetime(2024, 5, 31, 6, tzinfo=UTC)
    contract.assert_run_available(run, delivery_date)
    assert contract.summary(delivery_date)["delivery_hours"] == 24


def test_contract_rejects_weather_run_published_after_origin() -> None:
    contract = ForecastContract(issue_time_local=time(5, 0))
    delivery_date = date(2024, 6, 1)

    with pytest.raises(LeakageError, match="after forecast origin"):
        contract.assert_run_available(
            contract.weather_run(delivery_date), delivery_date
        )


def test_weather_run_validates_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        WeatherRun(
            initialized_at=datetime(2024, 1, 1),
            available_at=datetime(2024, 1, 1, 1),
            model="test",
        )
    with pytest.raises(ValidationError, match="before initialization"):
        WeatherRun(
            initialized_at=datetime(2024, 1, 1, 1, tzinfo=UTC),
            available_at=datetime(2024, 1, 1, tzinfo=UTC),
            model="test",
        )


def test_feature_observation_blocks_future_availability() -> None:
    origin = datetime(2024, 1, 1, 10, tzinfo=UTC)
    available = FeatureObservation(
        name="temperature",
        value=10.0,
        valid_at=origin + timedelta(hours=4),
        available_at=origin,
    )
    available.assert_available(origin)

    future = available.model_copy(
        update={"available_at": origin + timedelta(minutes=1)}
    )
    with pytest.raises(LeakageError, match="temperature"):
        future.assert_available(origin)
    with pytest.raises(ValueError, match="timezone-aware"):
        available.assert_available(datetime(2024, 1, 1, 10))
