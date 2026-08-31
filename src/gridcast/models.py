import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.typing import NDArray


class LightGBMLoadForecaster:
    """Deterministic gradient-boosted electricity-load forecaster.

    Parameters
    ----------
    n_estimators : int, default=300
        Number of boosting iterations.
    learning_rate : float, default=0.05
        Boosting shrinkage rate.
    random_state : int, default=42
        Seed controlling model reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """Initialize the LightGBM model.

        Parameters
        ----------
        n_estimators : int, default=300
            Number of boosting iterations.
        learning_rate : float, default=0.05
            Boosting shrinkage rate.
        random_state : int, default=42
            Seed controlling model reproducibility.
        """
        if n_estimators < 1:
            msg = "n_estimators must be positive"
            raise ValueError(msg)
        if learning_rate <= 0.0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model: LGBMRegressor | None = None

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "LightGBMLoadForecaster":
        """Fit the forecaster on complete feature rows.

        Parameters
        ----------
        features : pandas.DataFrame
            Training feature matrix.
        target : pandas.Series
            Training load target.

        Returns
        -------
        LightGBMLoadForecaster
            Fitted model.
        """
        complete = features.notna().all(axis=1) & target.notna()
        if not complete.any():
            msg = "training data do not contain complete feature rows"
            raise ValueError(msg)
        self._model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=31,
            min_child_samples=48,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        self._model.fit(features.loc[complete], target.loc[complete])
        return self

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Predict load for complete feature rows.

        Parameters
        ----------
        features : pandas.DataFrame
            Forecast feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted electricity load in megawatts.
        """
        if self._model is None:
            msg = "fit must be called before predict"
            raise RuntimeError(msg)
        if features.isna().any(axis=None):
            msg = "prediction features must be complete"
            raise ValueError(msg)
        prediction = self._model.predict(features)
        return np.asarray(prediction, dtype=float)


class LightGBMQuantileForecaster:
    """Gradient-boosted forecaster for one conditional target quantile.

    Parameters
    ----------
    quantile : float
        Quantile level strictly between zero and one.
    n_estimators : int, default=300
        Number of boosting iterations.
    learning_rate : float, default=0.05
        Boosting shrinkage rate.
    random_state : int, default=42
        Seed controlling model reproducibility.
    """

    def __init__(
        self,
        quantile: float,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """Initialize the quantile forecaster.

        Parameters
        ----------
        quantile : float
            Quantile level strictly between zero and one.
        n_estimators : int, default=300
            Number of boosting iterations.
        learning_rate : float, default=0.05
            Boosting shrinkage rate.
        random_state : int, default=42
            Seed controlling model reproducibility.
        """
        if not 0.0 < quantile < 1.0:
            msg = "quantile must be strictly between zero and one"
            raise ValueError(msg)
        if n_estimators < 1:
            msg = "n_estimators must be positive"
            raise ValueError(msg)
        if learning_rate <= 0.0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model: LGBMRegressor | None = None

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "LightGBMQuantileForecaster":
        """Fit the quantile model on complete feature rows.

        Parameters
        ----------
        features : pandas.DataFrame
            Training feature matrix.
        target : pandas.Series
            Training load target.

        Returns
        -------
        LightGBMQuantileForecaster
            Fitted model.
        """
        complete = features.notna().all(axis=1) & target.notna()
        if not complete.any():
            msg = "training data do not contain complete feature rows"
            raise ValueError(msg)
        self._model = LGBMRegressor(
            objective="quantile",
            alpha=self.quantile,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=31,
            min_child_samples=48,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        self._model.fit(features.loc[complete], target.loc[complete])
        return self

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Predict the configured conditional quantile.

        Parameters
        ----------
        features : pandas.DataFrame
            Forecast feature matrix.

        Returns
        -------
        numpy.ndarray
            Quantile predictions in megawatts.
        """
        if self._model is None:
            msg = "fit must be called before predict"
            raise RuntimeError(msg)
        if features.isna().any(axis=None):
            msg = "prediction features must be complete"
            raise ValueError(msg)
        prediction = self._model.predict(features)
        return np.asarray(prediction, dtype=float)
