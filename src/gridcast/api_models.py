from datetime import datetime
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gridcast.foundation_models import TIMESFM_2P5
from gridcast.model_comparison import COMPARISONS


class APIModel(BaseModel):
    """Base API model with strict response serialization."""

    model_config = ConfigDict(extra="forbid")


class StrictArtifactModel(APIModel):
    """Base model for generated artifacts without implicit coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class HealthResponse(APIModel):
    """API process health response.

    Parameters
    ----------
    status : str
        Current service status.
    service : str
        Service identifier.
    version : str
        Public API version.
    """

    status: str
    service: str
    version: str


class DatasetMetadata(APIModel):
    """Public PJME dataset metadata.

    Parameters
    ----------
    observations : int
        Number of normalized hourly observations.
    start : datetime
        First observation timestamp.
    end : datetime
        Last observation timestamp.
    years : float
        Approximate coverage in years.
    mean_load_mw : float
        Mean hourly electricity load.
    maximum_load_mw : float
        Maximum observed electricity load.
    """

    observations: int = Field(ge=1)
    start: datetime
    end: datetime
    years: float = Field(gt=0.0)
    mean_load_mw: float = Field(gt=0.0)
    maximum_load_mw: float = Field(gt=0.0)


class ExperimentMetadata(APIModel):
    """Chronological experiment boundaries.

    Parameters
    ----------
    horizon_hours : int
        Forecast horizon in hours.
    validation_folds : int
        Number of validation folds.
    holdout_folds : int
        Number of historical holdout folds.
    validation_start : datetime
        First validation timestamp.
    holdout_start : datetime
        First historical holdout timestamp.
    holdout_end : datetime
        Last historical holdout timestamp.
    """

    horizon_hours: int = Field(ge=1)
    validation_folds: int = Field(ge=1)
    holdout_folds: int = Field(ge=1)
    validation_start: datetime
    holdout_start: datetime
    holdout_end: datetime


class ProbabilisticMetadata(APIModel):
    """Probabilistic calibration metadata.

    Parameters
    ----------
    quantiles : list of float
        Quantile levels produced by the model.
    target_coverage : float
        Nominal interval coverage.
    conformal_correction_mw : float
        Symmetric correction applied to each interval bound.
    raw_coverage : float
        Historical holdout coverage before calibration.
    calibrated_coverage : float
        Historical holdout coverage after calibration.
    hourly_calibrated_coverage : float
        Frozen-test coverage after hour-conditional calibration.
    global_hourly_coverage_mae : float
        Mean absolute hourly deviation from nominal coverage for global calibration.
    conditional_hourly_coverage_mae : float
        Mean absolute hourly deviation for hour-conditional calibration.
    rolling_calibrated_coverage : float
        Prequential holdout coverage with causal rolling calibration.
    global_weekly_coverage_mae : float
        Mean absolute weekly coverage deviation for static global calibration.
    rolling_weekly_coverage_mae : float
        Mean absolute weekly coverage deviation for rolling calibration.
    """

    quantiles: list[float]
    target_coverage: float = Field(ge=0.0, le=1.0)
    conformal_correction_mw: float = Field(ge=0.0)
    raw_coverage: float = Field(ge=0.0, le=1.0)
    calibrated_coverage: float = Field(ge=0.0, le=1.0)
    hourly_calibrated_coverage: float = Field(ge=0.0, le=1.0)
    global_hourly_coverage_mae: float = Field(ge=0.0, le=1.0)
    conditional_hourly_coverage_mae: float = Field(ge=0.0, le=1.0)
    rolling_calibrated_coverage: float = Field(ge=0.0, le=1.0)
    global_weekly_coverage_mae: float = Field(ge=0.0, le=1.0)
    rolling_weekly_coverage_mae: float = Field(ge=0.0, le=1.0)


class MetadataResponse(APIModel):
    """Combined dataset and experiment metadata response."""

    dataset: DatasetMetadata
    experiment: ExperimentMetadata
    probabilistic: ProbabilisticMetadata


class LeaderboardEntry(APIModel):
    """Aggregate point-forecast model result."""

    split: str
    model: str
    label: str
    folds: int = Field(ge=1)
    observations: int = Field(ge=1)
    mae_mw: float = Field(ge=0.0)
    rmse_mw: float = Field(ge=0.0)
    mase: float = Field(ge=0.0)
    improvement_vs_weekly_percent: float


class PointForecast(APIModel):
    """One timestamped point forecast."""

    timestamp: datetime
    actual_mw: float
    prediction_mw: float
    model: str
    model_label: str
    split: str
    fold: int = Field(ge=1)


class PointForecastResponse(APIModel):
    """Point forecasts selected for one chronological fold."""

    split: str
    fold: int
    models: list[str]
    forecasts: list[PointForecast]


class ProbabilisticForecast(APIModel):
    """One timestamped calibrated probabilistic forecast."""

    timestamp: datetime
    actual_mw: float
    p10_mw: float
    p50_mw: float
    p90_mw: float
    p10_calibrated_mw: float
    p90_calibrated_mw: float
    p10_hourly_calibrated_mw: float
    p90_hourly_calibrated_mw: float
    p10_rolling_calibrated_mw: float
    p90_rolling_calibrated_mw: float
    split: str
    fold: int = Field(ge=1)


class ProbabilisticForecastResponse(APIModel):
    """Probabilistic forecasts selected for one chronological fold."""

    split: str
    fold: int
    target_coverage: float
    forecasts: list[ProbabilisticForecast]


class PerformanceMeasurement(APIModel):
    """One environment-qualified local model performance measurement."""

    model: str
    fit_time_ms: float = Field(ge=0.0)
    prediction_median_ms: float = Field(gt=0.0)
    prediction_p95_ms: float = Field(gt=0.0)
    throughput_rows_per_second: float = Field(gt=0.0)
    serialized_model_kib: float = Field(gt=0.0)
    fit_rss_delta_mib: float = Field(ge=0.0)
    horizon_rows: int = Field(ge=1)
    repetitions: int = Field(ge=1)


class PerformanceResponse(APIModel):
    """Local benchmark environment and per-model measurements."""

    environment: dict[str, str | int]
    measurements: list[PerformanceMeasurement]


class PointDecisionResult(APIModel):
    """Point-model result under one synthetic decision scenario."""

    scenario: str
    model: str
    shortage_cost: float = Field(ge=0.0)
    surplus_cost: float = Field(ge=0.0)
    optimal_quantile: float = Field(ge=0.0, le=1.0)
    folds: int = Field(ge=1)
    observations: int = Field(ge=1)
    mean_cost: float = Field(ge=0.0)
    regret_vs_perfect: float = Field(ge=0.0)
    cost_increase_vs_best_pct: float = Field(ge=0.0)


class QuantileDecisionResult(APIModel):
    """Cost-aware quantile result under one synthetic decision scenario."""

    scenario: str
    shortage_cost: float = Field(ge=0.0)
    surplus_cost: float = Field(ge=0.0)
    selected_quantile: float = Field(ge=0.0, le=1.0)
    p50_cost: float = Field(ge=0.0)
    cost_aware_quantile_cost: float = Field(ge=0.0)
    cost_savings_pct: float


class DecisionResponse(APIModel):
    """Point-model and probabilistic scheduling sensitivity results."""

    point_models: list[PointDecisionResult]
    quantile_schedules: list[QuantileDecisionResult]


class StatisticalComparison(StrictArtifactModel):
    """One dependence-aware paired weekly MAE comparison."""

    comparison_order: int = Field(ge=1)
    comparison_id: str = Field(min_length=1)
    candidate_model: str = Field(min_length=1)
    reference_model: str = Field(min_length=1)
    folds: int = Field(ge=1)
    observations_per_fold: int = Field(ge=1)
    candidate_mae_mw: float = Field(gt=0.0)
    reference_mae_mw: float = Field(gt=0.0)
    mean_mae_improvement_mw: float
    relative_improvement_pct: float
    wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    losses: int = Field(ge=0)
    weekly_win_rate: float = Field(ge=0.0, le=1.0)
    bootstrap_method: Literal["circular"]
    block_length_folds: int = Field(ge=1)
    bootstrap_replicates: int = Field(ge=1)
    bootstrap_seed: int
    confidence_level: float = Field(gt=0.0, lt=1.0)
    ci_low_mw: float
    ci_high_mw: float
    relative_ci_low_pct: float
    relative_ci_high_pct: float
    weekly_win_rate_ci_low: float = Field(ge=0.0, le=1.0)
    weekly_win_rate_ci_high: float = Field(ge=0.0, le=1.0)
    bootstrap_directional_support: float = Field(ge=0.0, le=1.0)
    first_half_improvement_mw: float
    second_half_improvement_mw: float
    adjusted_ci_low_mw: float
    adjusted_ci_high_mw: float
    adjusted_relative_ci_low_pct: float
    adjusted_relative_ci_high_pct: float
    simultaneous_superiority_supported: bool
    family_size: int = Field(ge=1)
    multiplicity_method: Literal["Bonferroni"]
    familywise_confidence_level: float = Field(gt=0.0, lt=1.0)
    adjusted_per_comparison_confidence_level: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        """Validate paired counts, intervals, orientation, and support flag."""
        if self.candidate_model == self.reference_model:
            raise ValueError("comparison models must be distinct")
        if self.wins + self.ties + self.losses != self.folds:
            raise ValueError("weekly outcomes must sum to fold count")
        if self.ci_low_mw > self.ci_high_mw:
            raise ValueError("marginal confidence interval is reversed")
        if self.relative_ci_low_pct > self.relative_ci_high_pct:
            raise ValueError("relative confidence interval is reversed")
        if self.weekly_win_rate_ci_low > self.weekly_win_rate_ci_high:
            raise ValueError("weekly win-rate confidence interval is reversed")
        if self.adjusted_ci_low_mw > self.adjusted_ci_high_mw:
            raise ValueError("adjusted confidence interval is reversed")
        if self.adjusted_relative_ci_low_pct > self.adjusted_relative_ci_high_pct:
            raise ValueError("adjusted relative confidence interval is reversed")
        if (
            self.adjusted_ci_low_mw > self.ci_low_mw
            or self.adjusted_ci_high_mw < self.ci_high_mw
            or self.adjusted_relative_ci_low_pct > self.relative_ci_low_pct
            or self.adjusted_relative_ci_high_pct < self.relative_ci_high_pct
        ):
            raise ValueError("adjusted intervals must contain marginal intervals")
        if self.simultaneous_superiority_supported != (self.adjusted_ci_low_mw > 0.0):
            raise ValueError("simultaneous support flag does not match interval")
        if not np.isclose(
            self.mean_mae_improvement_mw,
            self.reference_mae_mw - self.candidate_mae_mw,
        ):
            raise ValueError("absolute MAE improvement is inconsistent")
        expected_relative = 100.0 * self.mean_mae_improvement_mw / self.reference_mae_mw
        if not np.isclose(self.relative_improvement_pct, expected_relative):
            raise ValueError("relative MAE improvement is inconsistent")
        expected_win_rate = (self.wins + 0.5 * self.ties) / self.folds
        if not np.isclose(self.weekly_win_rate, expected_win_rate):
            raise ValueError("weekly win rate is inconsistent")
        first_folds = self.folds // 2
        second_folds = self.folds - first_folds
        recombined = (
            self.first_half_improvement_mw * first_folds
            + self.second_half_improvement_mw * second_folds
        ) / self.folds
        if not np.isclose(self.mean_mae_improvement_mw, recombined):
            raise ValueError("subperiod effects do not recombine to the full effect")
        return self


class StatisticalComparisonConfig(StrictArtifactModel):
    """Fixed paired block-bootstrap configuration."""

    bootstrap_replicates: int = Field(ge=1)
    block_length_folds: int = Field(ge=1)
    seed: int
    confidence_level: float = Field(gt=0.0, lt=1.0)
    sensitivity_block_lengths: list[int] = Field(min_length=1)
    expected_folds: int = Field(ge=1)
    observations_per_fold: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_blocks(self) -> Self:
        """Validate block lengths against the paired sample size."""
        if self.block_length_folds > self.expected_folds:
            raise ValueError("primary block length cannot exceed fold count")
        if max(self.sensitivity_block_lengths) > self.expected_folds:
            raise ValueError("sensitivity block length cannot exceed fold count")
        if self.block_length_folds not in self.sensitivity_block_lengths:
            raise ValueError("primary block length must be included in sensitivity")
        if len(set(self.sensitivity_block_lengths)) != len(
            self.sensitivity_block_lengths
        ):
            raise ValueError("sensitivity block lengths must be unique")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        return self


class StatisticalComparisonSummary(StrictArtifactModel):
    """Validated model-comparison summary artifact."""

    schema_version: Literal[1]
    config: StatisticalComparisonConfig
    orientation: str = Field(min_length=1)
    bootstrap_method: Literal["paired circular block bootstrap"]
    rng: Literal["numpy.random.PCG64"]
    quantile_method: Literal["linear"]
    family_size: int = Field(ge=1)
    multiplicity_method: Literal["Bonferroni"]
    familywise_confidence_level: float = Field(gt=0.0, lt=1.0)
    adjusted_per_comparison_confidence_level: float = Field(gt=0.0, lt=1.0)
    resample_indices_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifacts: dict[str, dict[str, object]]
    provenance_warnings: list[str]
    comparisons: list[StatisticalComparison] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_family(self) -> Self:
        """Validate family dimensions and the shared specified protocol."""
        expected_family = [
            (comparison_id, candidate, reference)
            for comparison_id, candidate, reference in COMPARISONS
        ]
        actual_family = [
            (item.comparison_id, item.candidate_model, item.reference_model)
            for item in self.comparisons
        ]
        if self.family_size != len(self.comparisons):
            raise ValueError("comparison family size does not match rows")
        if actual_family != expected_family:
            raise ValueError("comparison family does not match specified protocol")
        if [item.comparison_order for item in self.comparisons] != list(
            range(1, self.family_size + 1)
        ):
            raise ValueError("comparison order does not match specified protocol")
        if any(item.family_size != self.family_size for item in self.comparisons):
            raise ValueError("comparison row family sizes do not match")
        if any(item.folds != self.config.expected_folds for item in self.comparisons):
            raise ValueError("comparison fold count does not match config")
        expected_adjusted = (
            1.0 - (1.0 - self.config.confidence_level) / self.family_size
        )
        if not np.isclose(
            self.familywise_confidence_level, self.config.confidence_level
        ) or not np.isclose(
            self.adjusted_per_comparison_confidence_level, expected_adjusted
        ):
            raise ValueError("comparison confidence levels are inconsistent")
        for item in self.comparisons:
            if (
                item.observations_per_fold != self.config.observations_per_fold
                or item.bootstrap_replicates != self.config.bootstrap_replicates
                or item.block_length_folds != self.config.block_length_folds
                or item.bootstrap_seed != self.config.seed
                or not np.isclose(item.confidence_level, self.config.confidence_level)
                or not np.isclose(
                    item.familywise_confidence_level,
                    self.familywise_confidence_level,
                )
                or not np.isclose(
                    item.adjusted_per_comparison_confidence_level,
                    self.adjusted_per_comparison_confidence_level,
                )
            ):
                raise ValueError("comparison row protocol does not match config")
        model_mae: dict[str, float] = {}
        for item in self.comparisons:
            for model, mae in (
                (item.candidate_model, item.candidate_mae_mw),
                (item.reference_model, item.reference_mae_mw),
            ):
                if model in model_mae and not np.isclose(model_mae[model], mae):
                    raise ValueError("model MAE is inconsistent across comparisons")
                model_mae[model] = mae
        return self


class StatisticalComparisonResponse(APIModel):
    """Public paired model-comparison protocol and results."""

    orientation: str
    bootstrap_method: str
    bootstrap_replicates: int
    block_length_folds: int
    confidence_level: float
    familywise_confidence_level: float
    adjusted_per_comparison_confidence_level: float
    provenance_warnings: list[str]
    comparisons: list[StatisticalComparison]


class FoundationMetrics(StrictArtifactModel):
    """Zero-shot foundation-model historical holdout metrics."""

    model: str
    observations: int = Field(ge=1)
    folds: int = Field(ge=1)
    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    mase: float = Field(ge=0.0)
    p10_pinball_loss: float = Field(ge=0.0)
    p50_pinball_loss: float = Field(ge=0.0)
    p90_pinball_loss: float = Field(ge=0.0)
    raw_80_coverage: float = Field(ge=0.0, le=1.0)
    raw_80_mean_width_mw: float = Field(ge=0.0)


class FoundationTiming(StrictArtifactModel):
    """First-call and warm timings for the foundation-model benchmark."""

    first_call_seconds: float = Field(ge=0.0)
    warm_call_seconds: float = Field(ge=0.0)


class FoundationConfigMetadata(StrictArtifactModel):
    """Fully resolved foundation-model benchmark configuration."""

    model_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    weights_license: str = Field(default="unspecified", min_length=1)
    context_length: int = Field(ge=1)
    horizon: int = Field(ge=1)
    holdout_folds: int = Field(ge=1)
    model_parameters: dict[str, bool | int | float | str]

    @model_validator(mode="after")
    def resolve_legacy_license(self) -> Self:
        """Resolve the known TimesFM 2.5 license in legacy summaries."""
        if self.weights_license != "unspecified":
            return self
        if (
            self.model_name,
            self.model_id,
            self.model_revision,
        ) == (
            TIMESFM_2P5.model_name,
            TIMESFM_2P5.model_id,
            TIMESFM_2P5.model_revision,
        ):
            self.weights_license = TIMESFM_2P5.weights_license
            return self
        raise ValueError("foundation weights license is required")


class FoundationEnvironment(StrictArtifactModel):
    """Reproducible runtime metadata for a foundation-model benchmark."""

    python: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    timesfm: str = Field(min_length=1)
    device: Literal["cpu"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    dependency_lock_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    timesfm_lock_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    installed_packages: dict[str, str]

    @model_validator(mode="after")
    def validate_dependency_lock(self) -> Self:
        """Require a canonical or legacy dependency-lock digest."""
        if self.dependency_lock_sha256 is None and self.timesfm_lock_sha256 is None:
            raise ValueError("foundation dependency lock digest is required")
        if (
            self.dependency_lock_sha256 is not None
            and self.timesfm_lock_sha256 is not None
            and self.dependency_lock_sha256 != self.timesfm_lock_sha256
        ):
            raise ValueError("foundation dependency lock digests do not match")
        return self


class FoundationRuntime(StrictArtifactModel):
    """Stable public subset of foundation-model runtime metadata."""

    python: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    timesfm: str = Field(min_length=1)
    device: Literal["cpu"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timesfm_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FoundationSummary(StrictArtifactModel):
    """Validated on-disk foundation-model benchmark summary."""

    config: FoundationConfigMetadata
    metrics: FoundationMetrics
    timing: FoundationTiming
    environment: FoundationEnvironment

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Ensure aggregate metrics match the configured benchmark dimensions."""
        if self.metrics.model != self.config.model_name:
            raise ValueError("foundation model names do not match")
        if self.metrics.folds != self.config.holdout_folds:
            raise ValueError("foundation fold counts do not match")
        expected_observations = self.config.holdout_folds * self.config.horizon
        if self.metrics.observations != expected_observations:
            raise ValueError("foundation observation count does not match dimensions")
        return self


class FoundationResponse(APIModel):
    """Optional foundation-model configuration, metrics, and runtime."""

    model_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    weights_license: str = Field(min_length=1)
    context_length: int = Field(ge=1)
    horizon: int = Field(ge=1)
    holdout_folds: int = Field(ge=1)
    model_parameters: dict[str, bool | int | float | str]
    timing: FoundationTiming
    metrics: FoundationMetrics
    environment: FoundationRuntime
