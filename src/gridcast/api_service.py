import json
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from gridcast.api_models import (
    DatasetMetadata,
    DecisionResponse,
    ExperimentMetadata,
    FoundationResponse,
    FoundationRuntime,
    FoundationSummary,
    LeaderboardEntry,
    MetadataResponse,
    PerformanceMeasurement,
    PerformanceResponse,
    PointDecisionResult,
    PointForecast,
    PointForecastResponse,
    ProbabilisticForecast,
    ProbabilisticForecastResponse,
    ProbabilisticMetadata,
    QuantileDecisionResult,
    StatisticalComparisonResponse,
    StatisticalComparisonSummary,
)
from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, Col
from gridcast.dashboard_data import DashboardData, benchmark_week, display_model
from gridcast.foundation_models import (
    TIMESFM_2P5,
    TIMESFM_3,
    FoundationModelIdentity,
    validate_foundation_identity,
)
from gridcast.performance import load_performance_summary

POINT_DECISIONS_PATH = Path("artifacts/benchmark/decision_costs.csv")
QUANTILE_DECISIONS_PATH = Path("artifacts/probabilistic/decision_costs.csv")
FOUNDATION_SUMMARY_PATH = Path("artifacts/foundation/timesfm-2.5-200m/summary.json")
FOUNDATION3_SUMMARY_PATH = Path("artifacts/foundation/timesfm-3.0/summary.json")
COMPARISON_SUMMARY_PATH = Path("artifacts/model-comparison/summary.json")


class ForecastNotFoundError(LookupError):
    """Raised when a requested split, fold, or model does not exist."""


class InvalidArtifactError(ValueError):
    """Raised when a generated artifact does not satisfy its public contract."""


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
        probabilistic_holdout = self._object_dict(
            probabilistic.get("historical_holdout")
        )
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
                holdout_folds=self._int(benchmark_config.get("holdout_folds")),
                validation_start=self._datetime(benchmark.get("validation_start")),
                holdout_start=self._datetime(benchmark.get("holdout_start")),
                holdout_end=self._datetime(benchmark.get("holdout_end")),
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
                raw_coverage=self._float(probabilistic_holdout.get("raw_coverage")),
                calibrated_coverage=self._float(
                    probabilistic_holdout.get("calibrated_coverage")
                ),
                hourly_calibrated_coverage=self._float(
                    probabilistic_holdout.get("hourly_calibrated_coverage")
                ),
                global_hourly_coverage_mae=self._float(
                    probabilistic_holdout.get("global_hourly_coverage_mae")
                ),
                conditional_hourly_coverage_mae=self._float(
                    probabilistic_holdout.get("conditional_hourly_coverage_mae")
                ),
                rolling_calibrated_coverage=self._float(
                    probabilistic_holdout.get("rolling_calibrated_coverage")
                ),
                global_weekly_coverage_mae=self._float(
                    probabilistic_holdout.get("global_weekly_coverage_mae")
                ),
                rolling_weekly_coverage_mae=self._float(
                    probabilistic_holdout.get("rolling_weekly_coverage_mae")
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
                p10_rolling_calibrated_mw=float(row[Col.P10_ROLLING_CALIBRATED]),
                p90_rolling_calibrated_mw=float(row[Col.P90_ROLLING_CALIBRATED]),
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

    async def decisions(self) -> DecisionResponse:
        """Return optional synthetic scheduling-cost sensitivity results.

        Returns
        -------
        DecisionResponse
            Point-model and quantile scheduling results.
        """
        if not POINT_DECISIONS_PATH.exists() or not QUANTILE_DECISIONS_PATH.exists():
            raise ForecastNotFoundError(
                "decision artifacts not found; run `make benchmark probabilistic`"
            )
        point_data = pd.read_csv(POINT_DECISIONS_PATH)
        quantile_data = pd.read_csv(QUANTILE_DECISIONS_PATH)
        return DecisionResponse(
            point_models=[
                PointDecisionResult.model_validate(row)
                for row in point_data.to_dict(orient="records")
            ],
            quantile_schedules=[
                QuantileDecisionResult.model_validate(row)
                for row in quantile_data.to_dict(orient="records")
            ],
        )

    async def comparisons(self) -> StatisticalComparisonResponse:
        """Return optional dependence-aware paired model comparisons."""
        if not COMPARISON_SUMMARY_PATH.exists():
            raise ForecastNotFoundError(
                "comparison artifacts not found; run `make comparison`"
            )
        try:
            summary = StatisticalComparisonSummary.model_validate(
                self._read_json(COMPARISON_SUMMARY_PATH)
            )
            return StatisticalComparisonResponse(
                orientation=summary.orientation,
                bootstrap_method=summary.bootstrap_method,
                bootstrap_replicates=summary.config.bootstrap_replicates,
                block_length_folds=summary.config.block_length_folds,
                confidence_level=summary.config.confidence_level,
                familywise_confidence_level=summary.familywise_confidence_level,
                adjusted_per_comparison_confidence_level=(
                    summary.adjusted_per_comparison_confidence_level
                ),
                provenance_warnings=summary.provenance_warnings,
                comparisons=summary.comparisons,
            )
        except (OSError, ValueError) as error:
            raise InvalidArtifactError(
                f"invalid comparison artifact: {error}"
            ) from error

    async def foundation(self) -> FoundationResponse:
        """Return optional zero-shot time-series foundation-model results."""
        return self._foundation_response(
            FOUNDATION_SUMMARY_PATH, "make timesfm", TIMESFM_2P5
        )

    async def foundation3(self) -> FoundationResponse:
        """Return optional non-commercial TimesFM 3 benchmark results."""
        return self._foundation_response(
            FOUNDATION3_SUMMARY_PATH, "make timesfm3", TIMESFM_3
        )

    def _foundation_response(
        self,
        summary_path: Path,
        command: str,
        expected: FoundationModelIdentity,
    ) -> FoundationResponse:
        if not summary_path.exists():
            raise ForecastNotFoundError(
                f"foundation artifacts not found; run `{command}`"
            )
        try:
            summary = FoundationSummary.model_validate(self._read_json(summary_path))
            validate_foundation_identity(summary.config, expected)
            if summary.environment.timesfm != expected.package_version:
                raise ValueError("foundation package version does not match endpoint")
            if (
                expected.checkpoint_sha256 is not None
                and summary.environment.checkpoint_sha256 != expected.checkpoint_sha256
            ):
                raise ValueError("foundation checkpoint digest does not match endpoint")
            if (
                expected.config_sha256 is not None
                and summary.environment.checkpoint_config_sha256
                != expected.config_sha256
            ):
                raise ValueError(
                    "foundation configuration digest does not match endpoint"
                )
            dependency_lock = (
                summary.environment.dependency_lock_sha256
                or summary.environment.timesfm_lock_sha256
            )
            if dependency_lock is None:
                raise ValueError("foundation dependency lock digest is required")
            return FoundationResponse(
                model_name=summary.config.model_name,
                model_id=summary.config.model_id,
                model_revision=summary.config.model_revision,
                weights_license=summary.config.weights_license,
                context_length=summary.config.context_length,
                horizon=summary.config.horizon,
                holdout_folds=summary.config.holdout_folds,
                model_parameters=summary.config.model_parameters,
                timing=summary.timing,
                metrics=summary.metrics,
                environment=FoundationRuntime(
                    python=summary.environment.python,
                    platform=summary.environment.platform,
                    torch=summary.environment.torch,
                    timesfm=summary.environment.timesfm,
                    device=summary.environment.device,
                    checkpoint_sha256=summary.environment.checkpoint_sha256,
                    checkpoint_config_sha256=(
                        summary.environment.checkpoint_config_sha256
                    ),
                    dependency_lock_sha256=dependency_lock,
                    timesfm_lock_sha256=dependency_lock,
                ),
            )
        except (OSError, ValueError) as error:
            raise InvalidArtifactError(
                f"invalid foundation artifact: {error}"
            ) from error

    def _benchmark_summary(self) -> dict[str, object]:
        summary_path = self.data.probabilistic_summary.get("benchmark_summary")
        if isinstance(summary_path, dict):
            return cast(dict[str, object], summary_path)
        config = self._object_dict(self.data.probabilistic_summary.get("config"))
        holdout = self.data.benchmark_forecasts.loc[
            self.data.benchmark_forecasts[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)
        ]
        validation = self.data.benchmark_forecasts.loc[
            self.data.benchmark_forecasts[Col.SPLIT].eq("validation")
        ]
        return {
            "config": config,
            "validation_start": validation[Col.TIMESTAMP].min().isoformat(),
            "holdout_start": holdout[Col.TIMESTAMP].min().isoformat(),
            "holdout_end": holdout[Col.TIMESTAMP].max().isoformat(),
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

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"{path} must contain a JSON object"
            raise ValueError(msg)
        return cast(dict[str, object], payload)
