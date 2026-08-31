import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from gridcast.columns import Col


class MissingArtifactsError(FileNotFoundError):
    """Raised when dashboard artifacts have not been generated."""


@dataclass(frozen=True)
class DashboardPaths:
    """Filesystem locations consumed by the GridCast dashboard.

    Parameters
    ----------
    history : pathlib.Path
        Processed PJME load history.
    eda_summary : pathlib.Path
        EDA JSON summary.
    leaderboard : pathlib.Path
        Point benchmark leaderboard CSV.
    benchmark_forecasts : pathlib.Path
        Point benchmark forecasts.
    benchmark_fold_metrics : pathlib.Path
        Fold-level point metrics.
    probabilistic_forecasts : pathlib.Path
        Quantile and conformal forecasts.
    probabilistic_metrics : pathlib.Path
        Probabilistic metrics CSV.
    probabilistic_summary : pathlib.Path
        Probabilistic JSON summary.
    """

    history: Path = Path("data/processed/pjme_hourly.parquet")
    eda_summary: Path = Path("artifacts/eda/summary.json")
    leaderboard: Path = Path("artifacts/benchmark/leaderboard.csv")
    benchmark_forecasts: Path = Path("artifacts/benchmark/forecasts.parquet")
    benchmark_fold_metrics: Path = Path("artifacts/benchmark/fold_metrics.csv")
    probabilistic_forecasts: Path = Path("artifacts/probabilistic/forecasts.parquet")
    probabilistic_metrics: Path = Path("artifacts/probabilistic/metrics.csv")
    probabilistic_summary: Path = Path("artifacts/probabilistic/summary.json")


@dataclass(frozen=True)
class DashboardData:
    """Validated data loaded for all dashboard sections.

    Parameters
    ----------
    history : pandas.DataFrame
        Complete hourly load history.
    eda_summary : dict
        Dataset summary statistics.
    leaderboard : pandas.DataFrame
        Aggregate point benchmark metrics.
    benchmark_forecasts : pandas.DataFrame
        Point forecasts by model and fold.
    benchmark_fold_metrics : pandas.DataFrame
        Weekly point metrics.
    probabilistic_forecasts : pandas.DataFrame
        Quantile and calibrated interval forecasts.
    probabilistic_metrics : pandas.DataFrame
        Aggregate uncertainty metrics.
    probabilistic_summary : dict
        Probabilistic configuration and headline holdout metrics.
    """

    history: pd.DataFrame
    eda_summary: dict[str, object]
    leaderboard: pd.DataFrame
    benchmark_forecasts: pd.DataFrame
    benchmark_fold_metrics: pd.DataFrame
    probabilistic_forecasts: pd.DataFrame
    probabilistic_metrics: pd.DataFrame
    probabilistic_summary: dict[str, object]


MODEL_LABELS = {
    "timesfm_2_5_200m_zero_shot": "TimesFM 2.5 200M zero-shot",
    "lightgbm_exogenous": "LightGBM + weather + holidays",
    "lightgbm_weather": "LightGBM + weather",
    "lightgbm": "LightGBM base",
    "lightgbm_holidays": "LightGBM + holidays",
    "seasonal_naive_24h": "Daily seasonal naive",
    "seasonal_naive_168h": "Weekly seasonal naive",
    "persistence_1h": "Persistence",
}


def load_dashboard_data(paths: DashboardPaths | None = None) -> DashboardData:
    """Load and validate every artifact required by the dashboard.

    Parameters
    ----------
    paths : DashboardPaths, optional
        Custom artifact paths. Defaults to project-relative paths.

    Returns
    -------
    DashboardData
        Validated dashboard data.

    Raises
    ------
    MissingArtifactsError
        If one or more generated artifacts are absent.
    ValueError
        If an artifact does not satisfy its expected schema.
    """
    selected = paths or DashboardPaths()
    path_values = [
        selected.history,
        selected.eda_summary,
        selected.leaderboard,
        selected.benchmark_forecasts,
        selected.benchmark_fold_metrics,
        selected.probabilistic_forecasts,
        selected.probabilistic_metrics,
        selected.probabilistic_summary,
    ]
    missing = [str(path) for path in path_values if not path.exists()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        msg = (
            "dashboard artifacts are missing:\n"
            f"{details}\nRun `make data weather eda benchmark probabilistic`."
        )
        raise MissingArtifactsError(msg)

    history = pd.read_parquet(selected.history)
    leaderboard = pd.read_csv(selected.leaderboard)
    benchmark_forecasts = pd.read_parquet(selected.benchmark_forecasts)
    benchmark_fold_metrics = pd.read_csv(selected.benchmark_fold_metrics)
    probabilistic_forecasts = pd.read_parquet(selected.probabilistic_forecasts)
    probabilistic_metrics = pd.read_csv(selected.probabilistic_metrics)
    _require_columns(history, {Col.TIMESTAMP, Col.TARGET}, "history")
    _require_columns(
        leaderboard,
        {Col.SPLIT, Col.MODEL, "mae", "rmse", "mase"},
        "leaderboard",
    )
    _require_columns(
        benchmark_forecasts,
        {
            Col.TIMESTAMP,
            Col.TARGET,
            Col.PREDICTION,
            Col.MODEL,
            Col.SPLIT,
            Col.FOLD,
        },
        "benchmark forecasts",
    )
    _require_columns(
        benchmark_fold_metrics,
        {Col.MODEL, Col.SPLIT, Col.FOLD, "mae"},
        "benchmark fold metrics",
    )
    _require_columns(
        probabilistic_forecasts,
        {
            Col.TIMESTAMP,
            Col.TARGET,
            Col.P10,
            Col.P50,
            Col.P90,
            Col.P10_CALIBRATED,
            Col.P90_CALIBRATED,
            Col.P10_HOURLY_CALIBRATED,
            Col.P90_HOURLY_CALIBRATED,
            Col.P10_ROLLING_CALIBRATED,
            Col.P90_ROLLING_CALIBRATED,
            Col.SPLIT,
            Col.FOLD,
        },
        "probabilistic forecasts",
    )
    _require_columns(
        probabilistic_metrics,
        {
            Col.SPLIT,
            "raw_coverage",
            "calibrated_coverage",
            "hourly_calibrated_coverage",
            "rolling_calibrated_coverage",
            "raw_mean_width_mw",
            "calibrated_mean_width_mw",
            "hourly_calibrated_mean_width_mw",
            "rolling_calibrated_mean_width_mw",
        },
        "probabilistic metrics",
    )
    return DashboardData(
        history=history,
        eda_summary=_read_json(selected.eda_summary),
        leaderboard=leaderboard,
        benchmark_forecasts=benchmark_forecasts,
        benchmark_fold_metrics=benchmark_fold_metrics,
        probabilistic_forecasts=probabilistic_forecasts,
        probabilistic_metrics=probabilistic_metrics,
        probabilistic_summary=_read_json(selected.probabilistic_summary),
    )


def daily_history(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly load to daily mean and maximum values.

    Parameters
    ----------
    history : pandas.DataFrame
        Canonical hourly load history.

    Returns
    -------
    pandas.DataFrame
        Daily timestamps, mean load, and maximum load.
    """
    _require_columns(history, {Col.TIMESTAMP, Col.TARGET}, "history")
    return (
        history.assign(date=history[Col.TIMESTAMP].dt.floor("D"))
        .groupby("date", as_index=False)
        .agg(mean_load_mw=(Col.TARGET, "mean"), peak_load_mw=(Col.TARGET, "max"))
    )


def benchmark_week(
    forecasts: pd.DataFrame,
    split: str,
    fold: int,
    models: list[str],
) -> pd.DataFrame:
    """Select one benchmark fold and requested models.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Point benchmark forecasts.
    split : str
        Evaluation split.
    fold : int
        Fold number inside the split.
    models : list of str
        Model identifiers to retain.

    Returns
    -------
    pandas.DataFrame
        Chronologically sorted forecast rows.
    """
    selected = forecasts.loc[
        forecasts[Col.SPLIT].eq(split)
        & forecasts[Col.FOLD].eq(fold)
        & forecasts[Col.MODEL].isin(models)
    ]
    return selected.sort_values([Col.TIMESTAMP, Col.MODEL], ignore_index=True)


def probabilistic_week(forecasts: pd.DataFrame, split: str, fold: int) -> pd.DataFrame:
    """Select one probabilistic forecast fold.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Probabilistic forecasts.
    split : str
        Evaluation split.
    fold : int
        Fold number inside the split.

    Returns
    -------
    pandas.DataFrame
        Chronologically sorted quantile forecasts.
    """
    selected = forecasts.loc[
        forecasts[Col.SPLIT].eq(split) & forecasts[Col.FOLD].eq(fold)
    ]
    return selected.sort_values(Col.TIMESTAMP, ignore_index=True)


def display_model(model: str) -> str:
    """Return a readable model label.

    Parameters
    ----------
    model : str
        Internal model identifier.

    Returns
    -------
    str
        Human-readable display label.
    """
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object"
        raise ValueError(msg)
    return cast(dict[str, object], payload)


def _require_columns(
    data: pd.DataFrame, required: set[str | Col], artifact: str
) -> None:
    missing = {str(column) for column in required}.difference(data.columns)
    if missing:
        names = ", ".join(sorted(missing))
        msg = f"{artifact} is missing required columns: {names}"
        raise ValueError(msg)
