from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LeakageError(ValueError):
    """Raised when data were unavailable at forecast issuance."""


class WeatherRun(BaseModel):
    """Immutable weather-model vintage and its publication timestamp.

    Parameters
    ----------
    initialized_at : datetime
        Time at which the numerical weather run was initialized.
    available_at : datetime
        Earliest time at which the completed run was usable.
    model : str
        Weather-model identifier.
    """

    model_config = ConfigDict(frozen=True)

    initialized_at: datetime
    available_at: datetime
    model: str

    @model_validator(mode="after")
    def validate_timestamps(self) -> "WeatherRun":
        """Require timezone-aware, chronologically valid timestamps."""
        if self.initialized_at.tzinfo is None or self.available_at.tzinfo is None:
            msg = "weather timestamps must be timezone-aware"
            raise ValueError(msg)
        if self.available_at < self.initialized_at:
            msg = "weather run cannot be available before initialization"
            raise ValueError(msg)
        return self


class FeatureObservation(BaseModel):
    """Feature value with explicit validity and availability timestamps.

    Parameters
    ----------
    name : str
        Feature identifier.
    value : float
        Observed or forecast feature value.
    valid_at : datetime
        Time represented by the value.
    available_at : datetime
        Earliest time the value could have been consumed.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    valid_at: datetime
    available_at: datetime

    def assert_available(self, forecast_origin: datetime) -> None:
        """Require the feature to be available by forecast issuance."""
        if forecast_origin.tzinfo is None:
            msg = "forecast origin must be timezone-aware"
            raise ValueError(msg)
        if self.available_at.tzinfo is None:
            msg = "feature availability must be timezone-aware"
            raise ValueError(msg)
        if self.available_at > forecast_origin:
            raise LeakageError(
                f"{self.name} became available at {self.available_at.isoformat()}, "
                f"after forecast origin {forecast_origin.isoformat()}"
            )


class ForecastContract(BaseModel):
    """Operational contract for Italian day-ahead load forecasts.

    Parameters
    ----------
    area_eic : str
        ENTSO-E bidding-zone EIC code.
    issue_time_local : time
        Day-ahead forecast issuance time in the configured timezone.
    timezone_name : str
        IANA timezone for the delivery market.
    weather_model : str
        Archived numerical-weather-model identifier.
    weather_run_hour_utc : int
        Initialization hour of the selected D-1 weather run.
    weather_publication_lag : timedelta
        Conservative delay before the weather run is considered available.
    """

    model_config = ConfigDict(frozen=True)

    area_eic: str = "10YIT-GRTN-----B"
    issue_time_local: time = time(10, 0)
    timezone_name: str = "Europe/Rome"
    weather_model: str = "ecmwf_ifs"
    weather_run_hour_utc: int = Field(default=0, ge=0, le=23)
    weather_publication_lag: timedelta = timedelta(hours=6)

    def forecast_origin(self, delivery_date: date) -> datetime:
        """Return the UTC instant at which the day-ahead forecast is issued."""
        local_origin = datetime.combine(
            delivery_date - timedelta(days=1),
            self.issue_time_local,
            ZoneInfo(self.timezone_name),
        )
        return local_origin.astimezone(UTC)

    def weather_run(self, delivery_date: date) -> WeatherRun:
        """Return the archived weather run selected for a delivery day."""
        initialized_at = datetime.combine(
            delivery_date - timedelta(days=1),
            time(self.weather_run_hour_utc),
            UTC,
        )
        return WeatherRun(
            initialized_at=initialized_at,
            available_at=initialized_at + self.weather_publication_lag,
            model=self.weather_model,
        )

    def delivery_intervals(self, delivery_date: date) -> list[datetime]:
        """Return every real UTC hour in one local Italian delivery day."""
        zone = ZoneInfo(self.timezone_name)
        local_start = datetime.combine(delivery_date, time.min, zone)
        local_end = datetime.combine(delivery_date + timedelta(days=1), time.min, zone)
        utc_start = local_start.astimezone(UTC)
        utc_end = local_end.astimezone(UTC)
        hours = int((utc_end - utc_start).total_seconds() // 3_600)
        return [utc_start + timedelta(hours=offset) for offset in range(hours)]

    def assert_run_available(self, run: WeatherRun, delivery_date: date) -> None:
        """Require the weather vintage to be public before forecast issuance."""
        forecast_origin = self.forecast_origin(delivery_date)
        if run.available_at > forecast_origin:
            raise LeakageError(
                f"weather run {run.initialized_at.isoformat()} became available at "
                f"{run.available_at.isoformat()}, after forecast origin "
                f"{forecast_origin.isoformat()}"
            )

    def summary(self, delivery_date: date) -> dict[str, object]:
        """Return the immutable timing contract for one delivery day."""
        run = self.weather_run(delivery_date)
        self.assert_run_available(run, delivery_date)
        return {
            "area_eic": self.area_eic,
            "delivery_date": delivery_date.isoformat(),
            "forecast_origin": self.forecast_origin(delivery_date).isoformat(),
            "weather_model": run.model,
            "weather_run_initialized_at": run.initialized_at.isoformat(),
            "weather_run_available_at": run.available_at.isoformat(),
            "delivery_hours": len(self.delivery_intervals(delivery_date)),
        }
