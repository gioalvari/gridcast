from datetime import datetime
from pathlib import Path
from typing import cast

from gridcast.api_models import (
    DatasetMetadata,
    ExperimentMetadata,
    LeaderboardEntry,
    MetadataResponse,
    PerformanceMeasurement,
    PerformanceResponse,
    PointForecast,
    PointForecastResponse,
    ProbabilisticForecast,
    ProbabilisticForecastResponse,
    ProbabilisticMetadata,
)
from gridcast.columns import Col
from gridcast.dashboard_data import DashboardData, benchmark_week, display_model
from gridcast.performance import load_performance_summary


class ForecastNotFoundError(LookupError):
    """Raised when a requested split, fold, or model does not exist."""


class GridCastService:
    """Read-only application service backed by validated GridCast artifacts.

    Parameters
    ----------
    data : DashboardData
        Validated experiment and forecast artifacts.
    """

    def __init__(self, data: DashboardData) -> None:
        """Initialize the read-only service.

        Parameters
        ----------
        data : DashboardData
            Validated experiment and forecast artifacts.
        """
        self.data = data

    async def metadata(self) -> MetadataResponse:
        """Return dataset and experiment metadata.

        Returns
        -------
        MetadataResponse
            Public metadata for the loaded artifacts.
        """
        eda = self.data.eda_summary
        load = eda.get("load_mw", {})
        benchmark = self._benchmark_summary()
        benchmark_config = self._object_dict(benchmark.get("config"))
        probabilistic = self.data.probabilistic_summary
        probabilistic_test = self._object_dict(probabilistic.get("test"))
        return MetadataResponse(
            dataset=DatasetMetadata(
                observations=self._int(eda.get("observations")),
                start=self._datetime(eda.get("start")),
                end=self._datetime(eda.get("end")),
                years=self._float(eda.get("years")),
                mean_load_mw=self._float(self._object_dict(load).get("mean")),
                maximum_load_mw=self._float(self._object_dict(load).get("maximum")),
            ),
            experiment=ExperimentMetadata(
                horizon_hours=self._int(benchmark_config.get("horizon")),
                validation_folds=self._int(benchmark_config.get("validation_folds")),
                test_folds=self._int(benchmark_config.get("test_folds")),
                validation_start=self._datetime(benchmark.get("validation_start")),
                test_start=self._datetime(benchmark.get("test_start")),
                test_end=self._datetime(benchmark.get("test_end")),
            ),
            probabilistic=ProbabilisticMetadata(
                quantiles=[
                    self._float(value)
                    for value in self._object_list(probabilistic.get("quantiles"))
                ],
                target_coverage=self._float(probabilistic.get("target_coverage")),
                conformal_correction_mw=self._float(
                    probabilistic.get("conformal_correction_mw")
                ),
                raw_coverage=self._float(probabilistic_test.get("raw_coverage")),
                calibrated_coverage=self._float(
                    probabilistic_test.get("calibrated_coverage")
                ),
                hourly_calibrated_coverage=self._float(
                    probabilistic_test.get("hourly_calibrated_coverage")
                ),
                global_hourly_coverage_mae=self._float(
                    probabilistic_test.get("global_hourly_coverage_mae")
                ),
                conditional_hourly_coverage_mae=self._float(
                    probabilistic_test.get("conditional_hourly_coverage_mae")
                ),
            ),
        )

    async def leaderboard(self, split: str) -> list[LeaderboardEntry]:
        """Return the ordered leaderboard for an evaluation split.

        Parameters
        ----------
        split : str
            ``validation`` or ``test``.

        Returns
        -------
        list of LeaderboardEntry
            Models ordered by ascending MAE.
        """
        selected = self.data.leaderboard.loc[
            self.data.leaderboard[Col.SPLIT].eq(split)
        ].sort_values("mae")
        if selected.empty:
            raise ForecastNotFoundError(f"split not found: {split}")
        return [
            LeaderboardEntry(
                split=str(row[Col.SPLIT]),
                model=str(row[Col.MODEL]),
                label=display_model(str(row[Col.MODEL])),
                folds=int(row["folds"]),
                observations=int(row["observations"]),
                mae_mw=float(row["mae"]),
                rmse_mw=float(row["rmse"]),
                mase=float(row["mase"]),
                improvement_vs_weekly_percent=float(
                    row["mae_improvement_vs_weekly_pct"]
                ),
            )
            for _, row in selected.iterrows()
        ]

    async def point_forecasts(
        self, split: str, fold: int, models: list[str] | None
    ) -> PointForecastResponse:
        """Return point forecasts for one split and fold.

        Parameters
        ----------
        split : str
            ``validation`` or ``test``.
        fold : int
            Fold number inside the split.
        models : list of str, optional
            Models to include. Defaults to all available models.

        Returns
        -------
        PointForecastResponse
            Selected timestamped point forecasts.
        """
        available = self.data.benchmark_forecasts.loc[
            self.data.benchmark_forecasts[Col.SPLIT].eq(split)
            & self.data.benchmark_forecasts[Col.FOLD].eq(fold)
        ]
        if available.empty:
            raise ForecastNotFoundError(f"forecast fold not found: {split}/{fold}")
        selected_models = models or list(
            dict.fromkeys(available[Col.MODEL].astype(str).tolist())
        )
        unknown = set(selected_models).difference(available[Col.MODEL].astype(str))
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ForecastNotFoundError(f"models not found in fold: {names}")
        selected = benchmark_week(
            self.data.benchmark_forecasts, split, fold, selected_models
        )
        forecasts = [
            PointForecast(
                timestamp=cast(datetime, row[Col.TIMESTAMP]),
                actual_mw=float(row[Col.TARGET]),
                prediction_mw=float(row[Col.PREDICTION]),
                model=str(row[Col.MODEL]),
                model_label=display_model(str(row[Col.MODEL])),
                split=str(row[Col.SPLIT]),
                fold=int(row[Col.FOLD]),
            )
            for _, row in selected.iterrows()
        ]
        return PointForecastResponse(
            split=split,
            fold=fold,
            models=selected_models,
            forecasts=forecasts,
        )

    async def probabilistic_forecasts(
        self, split: str, fold: int
    ) -> ProbabilisticForecastResponse:
        """Return calibrated probabilistic forecasts for one fold.

        Parameters
        ----------
        split : str
            ``validation`` or ``test``.
        fold : int
            Fold number inside the split.

        Returns
        -------
        ProbabilisticForecastResponse
            Selected timestamped quantiles and calibrated bounds.
        """
        selected = self.data.probabilistic_forecasts.loc[
            self.data.probabilistic_forecasts[Col.SPLIT].eq(split)
            & self.data.probabilistic_forecasts[Col.FOLD].eq(fold)
        ].sort_values(Col.TIMESTAMP)
        if selected.empty:
            raise ForecastNotFoundError(f"forecast fold not found: {split}/{fold}")
        forecasts = [
            ProbabilisticForecast(
                timestamp=cast(datetime, row[Col.TIMESTAMP]),
                actual_mw=float(row[Col.TARGET]),
                p10_mw=float(row[Col.P10]),
                p50_mw=float(row[Col.P50]),
                p90_mw=float(row[Col.P90]),
                p10_calibrated_mw=float(row[Col.P10_CALIBRATED]),
                p90_calibrated_mw=float(row[Col.P90_CALIBRATED]),
                p10_hourly_calibrated_mw=float(row[Col.P10_HOURLY_CALIBRATED]),
                p90_hourly_calibrated_mw=float(row[Col.P90_HOURLY_CALIBRATED]),
                split=str(row[Col.SPLIT]),
                fold=int(row[Col.FOLD]),
            )
            for _, row in selected.iterrows()
        ]
        return ProbabilisticForecastResponse(
            split=split,
            fold=fold,
            target_coverage=self._float(
                self.data.probabilistic_summary.get("target_coverage")
            ),
            forecasts=forecasts,
        )

    async def performance(self) -> PerformanceResponse:
        """Return optional environment-qualified performance measurements.

        Returns
        -------
        PerformanceResponse
            Host metadata and model performance measurements.

        Raises
        ------
        ForecastNotFoundError
            If the local performance benchmark has not been generated.
        """
        try:
            summary = load_performance_summary(
                Path("artifacts/performance/summary.json")
            )
        except FileNotFoundError as error:
            raise ForecastNotFoundError(
                "performance artifacts not found; run `make performance`"
            ) from error
        environment = self._object_dict(summary.get("environment"))
        measurements = [
            PerformanceMeasurement.model_validate(item)
            for item in self._object_list(summary.get("measurements"))
        ]
        return PerformanceResponse(
            environment={
                key: value
                for key, value in environment.items()
                if isinstance(value, (str, int))
            },
            measurements=measurements,
        )

    def _benchmark_summary(self) -> dict[str, object]:
        summary_path = self.data.probabilistic_summary.get("benchmark_summary")
        if isinstance(summary_path, dict):
            return cast(dict[str, object], summary_path)
        config = self._object_dict(self.data.probabilistic_summary.get("config"))
        test = self.data.benchmark_forecasts.loc[
            self.data.benchmark_forecasts[Col.SPLIT].eq("test")
        ]
        validation = self.data.benchmark_forecasts.loc[
            self.data.benchmark_forecasts[Col.SPLIT].eq("validation")
        ]
        return {
            "config": config,
            "validation_start": validation[Col.TIMESTAMP].min().isoformat(),
            "test_start": test[Col.TIMESTAMP].min().isoformat(),
            "test_end": test[Col.TIMESTAMP].max().isoformat(),
        }

    @staticmethod
    def _object_dict(value: object) -> dict[str, object]:
        return cast(dict[str, object], value) if isinstance(value, dict) else {}

    @staticmethod
    def _object_list(value: object) -> list[object]:
        return cast(list[object], value) if isinstance(value, list) else []

    @staticmethod
    def _float(value: object) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"expected numeric metadata, received {value!r}")

    @staticmethod
    def _int(value: object) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        raise ValueError(f"expected integer metadata, received {value!r}")

    @staticmethod
    def _datetime(value: object) -> datetime:
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        if isinstance(value, datetime):
            return value
        raise ValueError(f"expected datetime metadata, received {value!r}")
