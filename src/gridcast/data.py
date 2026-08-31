import numpy as np
import pandas as pd

from gridcast.columns import Col


def generate_synthetic_load(
    periods: int = 24 * 140,
    seed: int = 42,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate hourly electricity demand with realistic seasonal structure.

    Parameters
    ----------
    periods : int, default=3360
        Number of hourly observations to generate.
    seed : int, default=42
        Seed used by the random number generator.
    start : str, default="2024-01-01"
        First timestamp in a format understood by pandas.

    Returns
    -------
    pandas.DataFrame
        Timestamped synthetic demand in megawatts.

    Raises
    ------
    ValueError
        If fewer than two weeks of observations are requested.
    """
    if periods < 24 * 14:
        msg = "periods must contain at least two weeks of hourly observations"
        raise ValueError(msg)

    timestamps = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    elapsed_hours = np.arange(periods, dtype=float)
    hour = timestamps.hour.to_numpy(dtype=float)
    weekday = timestamps.dayofweek.to_numpy(dtype=float)
    day_of_year = timestamps.dayofyear.to_numpy(dtype=float)

    daily_cycle = 700.0 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0)
    evening_peak = 450.0 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
    weekend_effect = np.where(weekday >= 5.0, -650.0, 0.0)
    annual_cycle = 400.0 * np.cos(2.0 * np.pi * (day_of_year - 15.0) / 365.25)
    trend = 0.08 * elapsed_hours

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=110.0, size=periods)
    load = 5_000.0 + daily_cycle + evening_peak + weekend_effect + annual_cycle
    load = load + trend + noise

    return pd.DataFrame(
        {
            Col.TIMESTAMP: timestamps,
            Col.TARGET: load,
        }
    )
