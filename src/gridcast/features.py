from enum import StrEnum

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from gridcast.columns import Col
from gridcast.pjm import validate_hourly_load
from gridcast.weather import validate_temperature_data


class Feature(StrEnum):
    """Leakage-safe model feature names."""

    HOUR_SIN = "hour_sin"
    HOUR_COS = "hour_cos"
    WEEK_SIN = "week_sin"
    WEEK_COS = "week_cos"
    YEAR_SIN = "year_sin"
    YEAR_COS = "year_cos"
    LAG_168 = "lag_168h"
    LAG_336 = "lag_336h"
    MEAN_24_AT_168 = "mean_24h_at_lag_168h"
    MEAN_168_AT_168 = "mean_168h_at_lag_168h"
    STD_168_AT_168 = "std_168h_at_lag_168h"
    IS_HOLIDAY = "is_federal_holiday"
    IS_PRE_HOLIDAY = "is_pre_holiday"
    IS_POST_HOLIDAY = "is_post_holiday"
    TEMPERATURE_LAG_168 = "temperature_lag_168h"
    TEMPERATURE_LAG_336 = "temperature_lag_336h"
    TEMPERATURE_CLIMATOLOGY = "temperature_prior_year_climatology"
    HEATING_DEGREES_CLIMATOLOGY = "heating_degrees_climatology"
    COOLING_DEGREES_CLIMATOLOGY = "cooling_degrees_climatology"


FEATURE_COLUMNS = [feature.value for feature in Feature]
BASE_FEATURE_COLUMNS = FEATURE_COLUMNS[:11]
EXOGENOUS_FEATURE_COLUMNS = FEATURE_COLUMNS[11:]
HOLIDAY_FEATURE_COLUMNS = [
    Feature.IS_HOLIDAY.value,
    Feature.IS_PRE_HOLIDAY.value,
    Feature.IS_POST_HOLIDAY.value,
]
WEATHER_FEATURE_COLUMNS = [
    Feature.TEMPERATURE_LAG_168.value,
    Feature.TEMPERATURE_LAG_336.value,
    Feature.TEMPERATURE_CLIMATOLOGY.value,
    Feature.HEATING_DEGREES_CLIMATOLOGY.value,
    Feature.COOLING_DEGREES_CLIMATOLOGY.value,
]
FEATURE_WARMUP_HOURS = 24 * 14
EXOGENOUS_WARMUP_HOURS = 24 * 366


def build_forecast_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build calendar and lag features safe for a one-week forecast origin.

    Every target-derived feature is delayed by at least 168 hours. Therefore,
    all rows in a 168-hour forecast horizon can be generated at the start of
    that horizon without reading any target inside it.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical, regular hourly load data.

    Returns
    -------
    pandas.DataFrame
        Numerical feature matrix aligned with ``data``.
    """
    validate_hourly_load(data)
    timestamps = data[Col.TIMESTAMP]
    target = data[Col.TARGET].astype(float)
    hour_angle = 2.0 * np.pi * timestamps.dt.hour / 24.0
    week_angle = (
        2.0
        * np.pi
        * (timestamps.dt.dayofweek * 24.0 + timestamps.dt.hour)
        / (24.0 * 7.0)
    )
    year_angle = 2.0 * np.pi * timestamps.dt.dayofyear / 365.25
    known_history = target.shift(24 * 7)

    return pd.DataFrame(
        {
            Feature.HOUR_SIN: np.sin(hour_angle),
            Feature.HOUR_COS: np.cos(hour_angle),
            Feature.WEEK_SIN: np.sin(week_angle),
            Feature.WEEK_COS: np.cos(week_angle),
            Feature.YEAR_SIN: np.sin(year_angle),
            Feature.YEAR_COS: np.cos(year_angle),
            Feature.LAG_168: known_history,
            Feature.LAG_336: target.shift(24 * 14),
            Feature.MEAN_24_AT_168: known_history.rolling(24).mean(),
            Feature.MEAN_168_AT_168: known_history.rolling(24 * 7).mean(),
            Feature.STD_168_AT_168: known_history.rolling(24 * 7).std(),
        },
        index=data.index,
    )


def build_exogenous_features(data: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Add ex-ante holidays and leakage-safe historical weather features.

    Weather observations are delayed by at least 168 hours. The climatology for
    each month-day-hour uses only matching observations from previous years.
    Contemporaneous realized temperature is never exposed to the model.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical regular hourly load data.
    weather : pandas.DataFrame
        Canonical regular hourly temperature data covering the load timestamps.

    Returns
    -------
    pandas.DataFrame
        Base features plus holidays and safe temperature proxies.

    Raises
    ------
    ValueError
        If weather does not cover every load timestamp.
    """
    base = build_forecast_features(data)
    validate_temperature_data(weather)
    aligned = data[[Col.TIMESTAMP]].merge(
        weather[[Col.TIMESTAMP, Col.TEMPERATURE]],
        on=Col.TIMESTAMP,
        how="left",
        validate="one_to_one",
    )
    if aligned[Col.TEMPERATURE].isna().any():
        msg = "weather must cover every load timestamp"
        raise ValueError(msg)

    timestamps = data[Col.TIMESTAMP]
    normalized_dates = timestamps.dt.normalize()
    holiday_calendar = USFederalHolidayCalendar()
    holidays = holiday_calendar.holidays(
        start=normalized_dates.iloc[0] - pd.Timedelta(days=1),
        end=normalized_dates.iloc[-1] + pd.Timedelta(days=1),
    )
    temperature = aligned[Col.TEMPERATURE].astype(float)
    climate_key = pd.MultiIndex.from_arrays(
        [timestamps.dt.month, timestamps.dt.day, timestamps.dt.hour]
    )
    climatology = temperature.groupby(climate_key).transform(
        lambda values: values.shift(1).expanding().mean()
    )

    exogenous = pd.DataFrame(
        {
            Feature.IS_HOLIDAY: normalized_dates.isin(holidays).astype(float),
            Feature.IS_PRE_HOLIDAY: (normalized_dates + pd.Timedelta(days=1))
            .isin(holidays)
            .astype(float),
            Feature.IS_POST_HOLIDAY: (normalized_dates - pd.Timedelta(days=1))
            .isin(holidays)
            .astype(float),
            Feature.TEMPERATURE_LAG_168: temperature.shift(24 * 7),
            Feature.TEMPERATURE_LAG_336: temperature.shift(24 * 14),
            Feature.TEMPERATURE_CLIMATOLOGY: climatology,
            Feature.HEATING_DEGREES_CLIMATOLOGY: np.maximum(18.0 - climatology, 0.0),
            Feature.COOLING_DEGREES_CLIMATOLOGY: np.maximum(climatology - 18.0, 0.0),
        },
        index=data.index,
    )
    return pd.concat([base, exogenous], axis=1)
