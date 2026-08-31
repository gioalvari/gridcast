import numpy as np
from numpy.typing import NDArray


class SeasonalNaiveForecaster:
    """Forecast by repeating observations from the previous seasonal cycle.

    Parameters
    ----------
    seasonal_period : int
        Number of observations in one seasonal cycle.
    """

    def __init__(self, seasonal_period: int) -> None:
        """Initialize the seasonal-naive forecaster.

        Parameters
        ----------
        seasonal_period : int
            Number of observations in one seasonal cycle.
        """
        if seasonal_period < 1:
            msg = "seasonal_period must be positive"
            raise ValueError(msg)
        self.seasonal_period = seasonal_period
        self._history: NDArray[np.float64] | None = None

    def fit(self, values: NDArray[np.float64]) -> "SeasonalNaiveForecaster":
        """Store the training history used to generate seasonal forecasts.

        Parameters
        ----------
        values : numpy.ndarray
            One-dimensional training target.

        Returns
        -------
        SeasonalNaiveForecaster
            The fitted model.
        """
        history = np.asarray(values, dtype=float)
        if history.ndim != 1:
            msg = "values must be one-dimensional"
            raise ValueError(msg)
        if len(history) < self.seasonal_period:
            msg = "training history must cover at least one seasonal period"
            raise ValueError(msg)
        if not np.isfinite(history).all():
            msg = "training history must contain only finite values"
            raise ValueError(msg)
        self._history = history[-self.seasonal_period :].copy()
        return self

    def predict(self, horizon: int) -> NDArray[np.float64]:
        """Predict future values recursively from the final seasonal cycle.

        Parameters
        ----------
        horizon : int
            Number of observations to forecast.

        Returns
        -------
        numpy.ndarray
            Forecast values with length ``horizon``.
        """
        if self._history is None:
            msg = "fit must be called before predict"
            raise RuntimeError(msg)
        if horizon < 1:
            msg = "horizon must be positive"
            raise ValueError(msg)

        season = self._history[-self.seasonal_period :]
        repetitions = int(np.ceil(horizon / self.seasonal_period))
        return np.tile(season, repetitions)[:horizon]
