from collections.abc import Callable
from importlib import import_module
from typing import Protocol, Self, cast

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMRegressor
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor


class PointForecaster(Protocol):
    """Structural interface for deterministic tabular load forecasters."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self:
        """Fit a fresh estimator on historical feature rows."""
        ...

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Predict one value for every supplied feature row."""
        ...


class RegressorBackend(Protocol):
    """Minimal interface implemented by supported regression backends."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> object:
        """Fit the backend estimator."""
        ...

    def predict(self, features: pd.DataFrame) -> object:
        """Return backend predictions."""
        ...


class MissingOptionalDependencyError(ImportError):
    """Raised when an explicitly requested optional estimator is unavailable."""


def _complete_training_rows(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    if not features.index.equals(target.index):
        raise ValueError("training feature and target indexes must match")
    complete = features.notna().all(axis=1) & target.notna()
    if not complete.any():
        raise ValueError("training data do not contain complete feature rows")
    clean_features = features.loc[complete]
    clean_target = target.loc[complete]
    features_finite = bool(np.isfinite(clean_features.to_numpy(dtype=np.float64)).all())
    target_finite = bool(np.isfinite(clean_target.to_numpy(dtype=np.float64)).all())
    if not features_finite or not target_finite:
        raise ValueError("training data must contain only finite values")
    return clean_features, clean_target


def _validate_prediction_features(features: pd.DataFrame) -> None:
    if features.isna().any(axis=None):
        raise ValueError("prediction features must be complete")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("prediction features must contain only finite values")


def _validated_prediction(
    values: object,
    expected_rows: int,
) -> NDArray[np.float64]:
    prediction = np.asarray(values, dtype=np.float64)
    if prediction.shape != (expected_rows,):
        raise ValueError(
            f"prediction shape must be ({expected_rows},), got {prediction.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("predictions must contain only finite values")
    return prediction


def _optional_regressor(
    module_name: str,
    class_name: str,
) -> Callable[..., RegressorBackend]:
    try:
        module = import_module(module_name)
    except ImportError as error:
        raise MissingOptionalDependencyError(
            f"{module_name} is required; install GridCast with `--extra extended`"
        ) from error
    estimator = getattr(module, class_name, None)
    if not isinstance(estimator, type):
        raise MissingOptionalDependencyError(
            f"{module_name}.{class_name} is not available"
        )
    return cast(Callable[..., RegressorBackend], estimator)


POINT_ESTIMATORS = (
    "lightgbm",
    "hist_gradient_boosting",
    "catboost",
    "xgboost",
)


def ensure_point_estimator_available(estimator: str) -> None:
    """Validate that one registered backend can be constructed in this environment."""
    if estimator not in POINT_ESTIMATORS:
        raise ValueError(
            f"unknown point estimator {estimator!r}; expected one of {POINT_ESTIMATORS}"
        )
    if estimator == "catboost":
        _optional_regressor("catboost", "CatBoostRegressor")
    elif estimator == "xgboost":
        _optional_regressor("xgboost", "XGBRegressor")


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
        clean_features, clean_target = _complete_training_rows(features, target)
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
        self._model.fit(clean_features, clean_target)
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
        _validate_prediction_features(features)
        return _validated_prediction(self._model.predict(features), len(features))

    @property
    def booster(self) -> Booster:
        """Return the fitted native LightGBM booster.

        Returns
        -------
        lightgbm.Booster
            Native model suitable for versioned bundle serialization.
        """
        if self._model is None:
            raise RuntimeError("fit must be called before accessing booster")
        return self._model.booster_


class CatBoostLoadForecaster:
    """Optional CatBoost MAE forecaster for numeric leakage-safe features."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """Initialize fixed CatBoost parameters without importing CatBoost."""
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model: RegressorBackend | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "CatBoostLoadForecaster":
        """Fit CatBoost on complete finite historical rows."""
        clean_features, clean_target = _complete_training_rows(features, target)
        estimator = _optional_regressor("catboost", "CatBoostRegressor")
        self._model = estimator(
            loss_function="MAE",
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=self.random_state,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        self._model.fit(clean_features, clean_target)
        return self

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Predict load with the fitted CatBoost model."""
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        _validate_prediction_features(features)
        return _validated_prediction(self._model.predict(features), len(features))


class XGBoostLoadForecaster:
    """Optional histogram XGBoost MAE forecaster."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """Initialize fixed XGBoost parameters without importing XGBoost."""
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model: RegressorBackend | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "XGBoostLoadForecaster":
        """Fit XGBoost on complete finite historical rows."""
        clean_features, clean_target = _complete_training_rows(features, target)
        estimator = _optional_regressor("xgboost", "XGBRegressor")
        self._model = estimator(
            objective="reg:absoluteerror",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=8,
            min_child_weight=10.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        self._model.fit(clean_features, clean_target)
        return self

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Predict load with the fitted XGBoost model."""
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        _validate_prediction_features(features)
        return _validated_prediction(self._model.predict(features), len(features))


class HistGradientBoostingLoadForecaster:
    """Scikit-learn histogram gradient-boosting MAE forecaster."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """Initialize fixed histogram gradient-boosting parameters."""
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model: HistGradientBoostingRegressor | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "HistGradientBoostingLoadForecaster":
        """Fit histogram gradient boosting on complete finite historical rows."""
        clean_features, clean_target = _complete_training_rows(features, target)
        self._model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            max_leaf_nodes=31,
            min_samples_leaf=48,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=self.random_state,
        )
        self._model.fit(clean_features, clean_target)
        return self

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Predict load with fitted histogram gradient boosting."""
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        _validate_prediction_features(features)
        return _validated_prediction(self._model.predict(features), len(features))


def create_point_forecaster(
    estimator: str,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> PointForecaster:
    """Create a fresh deterministic point forecaster by stable backend name."""
    if estimator == "lightgbm":
        return LightGBMLoadForecaster(n_estimators, learning_rate, random_state)
    if estimator == "hist_gradient_boosting":
        return HistGradientBoostingLoadForecaster(
            n_estimators, learning_rate, random_state
        )
    if estimator == "catboost":
        return CatBoostLoadForecaster(n_estimators, learning_rate, random_state)
    if estimator == "xgboost":
        return XGBoostLoadForecaster(n_estimators, learning_rate, random_state)
    raise ValueError(
        f"unknown point estimator {estimator!r}; expected one of {POINT_ESTIMATORS}"
    )


def resolved_point_model_parameters(
    n_estimators: int,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> dict[str, dict[str, bool | float | int | str]]:
    """Return all explicit backend settings used by the extended benchmark."""
    return {
        "lightgbm": {
            "objective": "regression_l1",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": 31,
            "min_child_samples": 48,
            "subsample": 0.9,
            "subsample_freq": 0,
            "colsample_bytree": 0.9,
            "reg_lambda": 0.1,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        },
        "hist_gradient_boosting": {
            "loss": "absolute_error",
            "max_iter": n_estimators,
            "learning_rate": learning_rate,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 48,
            "l2_regularization": 0.1,
            "early_stopping": False,
            "random_state": random_state,
        },
        "catboost": {
            "loss_function": "MAE",
            "iterations": n_estimators,
            "learning_rate": learning_rate,
            "depth": 8,
            "l2_leaf_reg": 3.0,
            "random_seed": random_state,
            "allow_writing_files": False,
            "thread_count": -1,
            "verbose": False,
        },
        "xgboost": {
            "objective": "reg:absoluteerror",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": 8,
            "min_child_weight": 10.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 0.1,
            "random_state": random_state,
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
        },
    }


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
        clean_features, clean_target = _complete_training_rows(features, target)
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
        self._model.fit(clean_features, clean_target)
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
        _validate_prediction_features(features)
        return _validated_prediction(self._model.predict(features), len(features))

    @property
    def booster(self) -> Booster:
        """Return the fitted native LightGBM booster.

        Returns
        -------
        lightgbm.Booster
            Native model suitable for versioned bundle serialization.
        """
        if self._model is None:
            raise RuntimeError("fit must be called before accessing booster")
        return self._model.booster_
