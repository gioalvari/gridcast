import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, Col
from gridcast.foundation_models import TIMESFM_2P5, TIMESFM_3
from gridcast.provenance import build_experiment_manifest, file_sha256, write_manifest

DAILY_NAIVE = "seasonal_naive_24h"
LIGHTGBM = "lightgbm_exogenous"

COMPARISONS = (
    ("lightgbm_vs_daily_naive", LIGHTGBM, DAILY_NAIVE),
    ("timesfm_2_5_vs_daily_naive", TIMESFM_2P5.model_name, DAILY_NAIVE),
    ("timesfm_3_vs_daily_naive", TIMESFM_3.model_name, DAILY_NAIVE),
    ("timesfm_2_5_vs_lightgbm", TIMESFM_2P5.model_name, LIGHTGBM),
    ("timesfm_3_vs_lightgbm", TIMESFM_3.model_name, LIGHTGBM),
    ("timesfm_3_vs_timesfm_2_5", TIMESFM_3.model_name, TIMESFM_2P5.model_name),
)

REQUIRED_FORECAST_COLUMNS = (
    Col.TIMESTAMP,
    Col.TARGET,
    Col.PREDICTION,
    Col.MODEL,
    Col.SPLIT,
    Col.FOLD,
    Col.CUTOFF,
)


@dataclass(frozen=True)
class ComparisonConfig:
    """Fixed dependence-aware paired-comparison protocol.

    Parameters
    ----------
    bootstrap_replicates : int, default=100000
        Resamples used for each bootstrap specification.
    block_length_folds : int, default=4
        Primary circular block length in weekly folds.
    seed : int, default=20260903
        PCG64 seed shared by all primary contrasts.
    confidence_level : float, default=0.95
        Marginal percentile confidence level.
    sensitivity_block_lengths : tuple, default=(2, 4, 6, 8, 13, 26)
        Circular block lengths used for sensitivity analysis.
    expected_folds : int, default=52
        Required number of paired weekly folds.
    observations_per_fold : int, default=168
        Required hourly observations in every weekly fold.
    """

    bootstrap_replicates: int = 100_000
    block_length_folds: int = 4
    seed: int = 20_260_903
    confidence_level: float = 0.95
    sensitivity_block_lengths: tuple[int, ...] = (2, 4, 6, 8, 13, 26)
    expected_folds: int = 52
    observations_per_fold: int = 168

    def __post_init__(self) -> None:
        """Validate the specified bootstrap protocol."""
        if (
            min(
                self.bootstrap_replicates,
                self.block_length_folds,
                self.expected_folds,
                self.observations_per_fold,
            )
            < 1
        ):
            raise ValueError("bootstrap dimensions must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if (
            not self.sensitivity_block_lengths
            or min(self.sensitivity_block_lengths) < 1
        ):
            raise ValueError("sensitivity block lengths must be positive")
        if self.block_length_folds > self.expected_folds:
            raise ValueError("primary block length cannot exceed fold count")
        if max(self.sensitivity_block_lengths) > self.expected_folds:
            raise ValueError("sensitivity block lengths cannot exceed fold count")
        if self.block_length_folds not in self.sensitivity_block_lengths:
            raise ValueError("primary block length must be included in sensitivity")
        if len(set(self.sensitivity_block_lengths)) != len(
            self.sensitivity_block_lengths
        ):
            raise ValueError("sensitivity block lengths must be unique")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class ComparisonResult:
    """Weekly losses, paired comparisons, sensitivity, and audit metadata."""

    fold_losses: pd.DataFrame
    comparisons: pd.DataFrame
    sensitivity: pd.DataFrame
    resample_indices_sha256: str


def _validate_forecast_frame(
    data: pd.DataFrame,
    model: str,
    expected_folds: int,
    observations_per_fold: int,
) -> pd.DataFrame:
    missing = set(REQUIRED_FORECAST_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"{model} forecasts missing columns: {sorted(missing)}")
    selected = data.loc[
        data[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT) & data[Col.MODEL].eq(model),
        list(REQUIRED_FORECAST_COLUMNS),
    ].copy()
    if selected.empty:
        raise ValueError(f"historical holdout forecasts not found for {model}")
    if selected.duplicated([Col.FOLD, Col.TIMESTAMP]).any():
        raise ValueError(f"duplicate fold timestamps found for {model}")
    if selected[[Col.TIMESTAMP, Col.CUTOFF]].isna().any().any():
        raise ValueError(f"missing timestamps or cutoffs found for {model}")
    numeric = selected[[Col.TARGET, Col.PREDICTION]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"non-finite target or prediction found for {model}")
    counts = selected.groupby(Col.FOLD, sort=True).size()
    if list(counts.index) != list(range(1, expected_folds + 1)):
        raise ValueError(f"{model} must contain folds 1 through {expected_folds}")
    if not counts.eq(observations_per_fold).all():
        raise ValueError(
            f"{model} must contain {observations_per_fold} observations per fold"
        )
    ordered = selected.sort_values([Col.FOLD, Col.TIMESTAMP]).reset_index(drop=True)
    expected_step = pd.Timedelta(hours=1)
    fold_starts: list[pd.Timestamp] = []
    for _, weekly in ordered.groupby(Col.FOLD, sort=True):
        if weekly[Col.CUTOFF].nunique() != 1:
            raise ValueError(f"each {model} fold must contain exactly one cutoff")
        timestamps = pd.DatetimeIndex(weekly[Col.TIMESTAMP])
        cutoff = pd.Timestamp(weekly[Col.CUTOFF].iloc[0])
        if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
            raise ValueError(f"{model} fold timestamps must be strictly increasing")
        if len(timestamps) > 1 and not (timestamps[1:] - timestamps[:-1]).equals(
            pd.TimedeltaIndex([expected_step] * (len(timestamps) - 1))
        ):
            raise ValueError(f"{model} fold timestamps must be contiguous hourly data")
        if timestamps[0] != cutoff + expected_step:
            raise ValueError(f"{model} forecast must start one hour after its cutoff")
        start = pd.Timestamp(weekly[Col.TIMESTAMP].iloc[0])
        if pd.isna(start):
            raise ValueError(f"{model} fold start cannot be missing")
        fold_starts.append(start)
    starts = pd.DatetimeIndex(fold_starts)
    expected_fold_step = pd.Timedelta(hours=observations_per_fold)
    if len(starts) > 1 and not (starts[1:] - starts[:-1]).equals(
        pd.TimedeltaIndex([expected_fold_step] * (len(starts) - 1))
    ):
        raise ValueError(f"{model} folds must be contiguous and non-overlapping")
    return ordered


def build_fold_losses(
    forecasts: dict[str, pd.DataFrame],
    *,
    expected_folds: int = 52,
    observations_per_fold: int = 168,
) -> pd.DataFrame:
    """Validate paired forecast rows and compute one MAE per model and week."""
    models = tuple(dict.fromkeys(model for pair in COMPARISONS for model in pair[1:]))
    missing_models = set(models).difference(forecasts)
    if missing_models:
        raise ValueError(f"forecast inputs missing models: {sorted(missing_models)}")

    validated = {
        model: _validate_forecast_frame(
            forecasts[model], model, expected_folds, observations_per_fold
        )
        for model in models
    }
    key = [Col.SPLIT, Col.FOLD, Col.CUTOFF, Col.TIMESTAMP]
    canonical = validated[DAILY_NAIVE]
    rows: list[dict[str, object]] = []
    for model, data in validated.items():
        if not canonical[key].equals(data[key]):
            raise ValueError(f"forecast keys do not match for {model}")
        if not np.array_equal(
            canonical[Col.TARGET].to_numpy(), data[Col.TARGET].to_numpy()
        ):
            raise ValueError(f"actual values do not match for {model}")
        for fold, weekly in data.groupby(Col.FOLD, sort=True):
            absolute_error = np.abs(
                weekly[Col.TARGET].to_numpy(dtype=np.float64)
                - weekly[Col.PREDICTION].to_numpy(dtype=np.float64)
            )
            rows.append(
                {
                    Col.SPLIT: HISTORICAL_HOLDOUT_SPLIT,
                    Col.FOLD: int(str(fold)),
                    Col.CUTOFF: weekly[Col.CUTOFF].iloc[0],
                    "forecast_start": weekly[Col.TIMESTAMP].min(),
                    "forecast_end": weekly[Col.TIMESTAMP].max(),
                    "observations": len(weekly),
                    Col.MODEL: model,
                    "mae_mw": float(absolute_error.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values([Col.FOLD, Col.MODEL]).reset_index(drop=True)


def circular_block_indices(
    folds: int,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Generate fixed-length circular block-bootstrap indices."""
    if min(folds, block_length, replicates) < 1:
        raise ValueError("folds, block length, and replicates must be positive")
    block_count = int(np.ceil(folds / block_length))
    starts = rng.integers(0, folds, size=(replicates, block_count))
    indices = (starts[:, :, None] + np.arange(block_length)) % folds
    return np.asarray(indices.reshape(replicates, -1)[:, :folds], dtype=np.int64)


def moving_block_indices(
    folds: int,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Generate moving-block indices without wraparound for sensitivity checks."""
    if min(folds, block_length, replicates) < 1 or block_length > folds:
        raise ValueError("moving block length must be between one and fold count")
    block_count = int(np.ceil(folds / block_length))
    starts = rng.integers(0, folds - block_length + 1, size=(replicates, block_count))
    indices = starts[:, :, None] + np.arange(block_length)
    return np.asarray(indices.reshape(replicates, -1)[:, :folds], dtype=np.int64)


def _loss_matrix(fold_losses: pd.DataFrame) -> tuple[list[str], NDArray[np.float64]]:
    pivot = fold_losses.pivot(index=Col.FOLD, columns=Col.MODEL, values="mae_mw")
    models = list(pivot.columns.astype(str))
    return models, np.asarray(pivot.to_numpy(), dtype=np.float64)


def _comparison_rows(
    models: list[str],
    losses: NDArray[np.float64],
    indices: NDArray[np.int64],
    config: ComparisonConfig,
    *,
    family_size: int,
) -> list[dict[str, object]]:
    model_index = {model: index for index, model in enumerate(models)}
    alpha = 1.0 - config.confidence_level
    family_alpha = alpha / family_size
    adjusted_confidence_level = 1.0 - family_alpha
    rows: list[dict[str, object]] = []
    for order, (comparison_id, candidate, reference) in enumerate(COMPARISONS, 1):
        candidate_loss = losses[:, model_index[candidate]]
        reference_loss = losses[:, model_index[reference]]
        difference = reference_loss - candidate_loss
        candidate_boot = candidate_loss[indices].mean(axis=1)
        reference_boot = reference_loss[indices].mean(axis=1)
        if reference_loss.mean() <= 0.0 or np.any(reference_boot <= 0.0):
            raise ValueError(f"reference MAE must be positive for {comparison_id}")
        difference_boot = reference_boot - candidate_boot
        relative_boot = 100.0 * difference_boot / reference_boot
        win_boot = (difference[indices] > 0).mean(axis=1) + 0.5 * (
            difference[indices] == 0
        ).mean(axis=1)
        ci = np.quantile(
            difference_boot, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
        )
        relative_ci = np.quantile(
            relative_boot, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
        )
        win_ci = np.quantile(
            win_boot, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
        )
        family_ci = np.quantile(
            difference_boot,
            [family_alpha / 2.0, 1.0 - family_alpha / 2.0],
            method="linear",
        )
        family_relative_ci = np.quantile(
            relative_boot,
            [family_alpha / 2.0, 1.0 - family_alpha / 2.0],
            method="linear",
        )
        wins = int((difference > 0).sum())
        ties = int((difference == 0).sum())
        midpoint = len(difference) // 2
        rows.append(
            {
                "comparison_order": order,
                "comparison_id": comparison_id,
                "candidate_model": candidate,
                "reference_model": reference,
                "folds": len(difference),
                "observations_per_fold": config.observations_per_fold,
                "candidate_mae_mw": float(candidate_loss.mean()),
                "reference_mae_mw": float(reference_loss.mean()),
                "mean_mae_improvement_mw": float(difference.mean()),
                "relative_improvement_pct": float(
                    100.0 * difference.mean() / reference_loss.mean()
                ),
                "wins": wins,
                "ties": ties,
                "losses": int(len(difference) - wins - ties),
                "weekly_win_rate": float((wins + 0.5 * ties) / len(difference)),
                "bootstrap_method": "circular",
                "block_length_folds": config.block_length_folds,
                "bootstrap_replicates": config.bootstrap_replicates,
                "bootstrap_seed": config.seed,
                "confidence_level": config.confidence_level,
                "ci_low_mw": float(ci[0]),
                "ci_high_mw": float(ci[1]),
                "relative_ci_low_pct": float(relative_ci[0]),
                "relative_ci_high_pct": float(relative_ci[1]),
                "weekly_win_rate_ci_low": float(win_ci[0]),
                "weekly_win_rate_ci_high": float(win_ci[1]),
                "bootstrap_directional_support": float(
                    ((difference_boot > 0).sum() + 0.5 * (difference_boot == 0).sum())
                    / len(difference_boot)
                ),
                "first_half_improvement_mw": float(difference[:midpoint].mean()),
                "second_half_improvement_mw": float(difference[midpoint:].mean()),
                "adjusted_ci_low_mw": float(family_ci[0]),
                "adjusted_ci_high_mw": float(family_ci[1]),
                "adjusted_relative_ci_low_pct": float(family_relative_ci[0]),
                "adjusted_relative_ci_high_pct": float(family_relative_ci[1]),
                "simultaneous_superiority_supported": bool(family_ci[0] > 0.0),
                "family_size": family_size,
                "multiplicity_method": "Bonferroni",
                "familywise_confidence_level": config.confidence_level,
                "adjusted_per_comparison_confidence_level": (adjusted_confidence_level),
            }
        )
    return rows


def run_model_comparison(
    forecasts: dict[str, pd.DataFrame],
    config: ComparisonConfig,
) -> ComparisonResult:
    """Run the fixed family of dependence-aware paired MAE comparisons."""
    fold_losses = build_fold_losses(
        forecasts,
        expected_folds=config.expected_folds,
        observations_per_fold=config.observations_per_fold,
    )
    models, losses = _loss_matrix(fold_losses)
    primary_rng = np.random.Generator(np.random.PCG64(config.seed))
    primary_indices = circular_block_indices(
        config.expected_folds,
        config.block_length_folds,
        config.bootstrap_replicates,
        primary_rng,
    )
    comparisons = pd.DataFrame(
        _comparison_rows(
            models,
            losses,
            primary_indices,
            config,
            family_size=len(COMPARISONS),
        )
    )
    sensitivity_rows: list[dict[str, object]] = []
    specs = [
        ("circular", block_length) for block_length in config.sensitivity_block_lengths
    ] + [("moving", config.block_length_folds)]
    for spec_index, (method, block_length) in enumerate(specs, 1):
        if method == "circular" and block_length == config.block_length_folds:
            indices = primary_indices
        elif method == "circular":
            rng = np.random.Generator(np.random.PCG64(config.seed + spec_index))
            indices = circular_block_indices(
                config.expected_folds,
                block_length,
                config.bootstrap_replicates,
                rng,
            )
        else:
            rng = np.random.Generator(np.random.PCG64(config.seed + spec_index))
            indices = moving_block_indices(
                config.expected_folds,
                block_length,
                config.bootstrap_replicates,
                rng,
            )
        rows = _comparison_rows(
            models,
            losses,
            indices,
            config,
            family_size=len(COMPARISONS),
        )
        for row in rows:
            sensitivity_rows.append(
                {
                    "comparison_id": row["comparison_id"],
                    "bootstrap_method": method,
                    "block_length_folds": block_length,
                    "bootstrap_seed": (
                        config.seed
                        if method == "circular"
                        and block_length == config.block_length_folds
                        else config.seed + spec_index
                    ),
                    "resample_indices_sha256": hashlib.sha256(
                        indices.tobytes()
                    ).hexdigest(),
                    "primary": method == "circular"
                    and block_length == config.block_length_folds,
                    "ci_low_mw": row["ci_low_mw"],
                    "ci_high_mw": row["ci_high_mw"],
                    "relative_ci_low_pct": row["relative_ci_low_pct"],
                    "relative_ci_high_pct": row["relative_ci_high_pct"],
                    "weekly_win_rate_ci_low": row["weekly_win_rate_ci_low"],
                    "weekly_win_rate_ci_high": row["weekly_win_rate_ci_high"],
                    "bootstrap_directional_support": row[
                        "bootstrap_directional_support"
                    ],
                    "adjusted_ci_low_mw": row["adjusted_ci_low_mw"],
                    "adjusted_ci_high_mw": row["adjusted_ci_high_mw"],
                    "adjusted_relative_ci_low_pct": row["adjusted_relative_ci_low_pct"],
                    "adjusted_relative_ci_high_pct": row[
                        "adjusted_relative_ci_high_pct"
                    ],
                    "simultaneous_superiority_supported": row[
                        "simultaneous_superiority_supported"
                    ],
                }
            )
    digest = hashlib.sha256(primary_indices.tobytes()).hexdigest()
    return ComparisonResult(
        fold_losses=fold_losses,
        comparisons=comparisons,
        sensitivity=pd.DataFrame(sensitivity_rows),
        resample_indices_sha256=digest,
    )


def load_comparison_inputs(
    benchmark_path: Path,
    timesfm25_path: Path,
    timesfm3_path: Path,
) -> dict[str, pd.DataFrame]:
    """Load the four specified models from three forecast artifacts."""
    paths = (benchmark_path, timesfm25_path, timesfm3_path)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"comparison forecast artifacts not found: {missing}")
    benchmark = pd.read_parquet(benchmark_path)
    return {
        DAILY_NAIVE: benchmark,
        LIGHTGBM: benchmark,
        TIMESFM_2P5.model_name: pd.read_parquet(timesfm25_path),
        TIMESFM_3.model_name: pd.read_parquet(timesfm3_path),
    }


def write_comparison_artifacts(
    result: ComparisonResult,
    config: ComparisonConfig,
    output_dir: Path,
    source_paths: dict[str, Path],
) -> None:
    """Persist paired-comparison tables, summary, and source provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.fold_losses.to_parquet(output_dir / "fold_losses.parquet", index=False)
    result.comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    result.sensitivity.to_csv(output_dir / "bootstrap_sensitivity.csv", index=False)
    source_artifacts: dict[str, dict[str, object]] = {}
    provenance_warnings: list[str] = []
    for name, path in source_paths.items():
        item: dict[str, object] = {"path": str(path), "sha256": file_sha256(path)}
        manifest_path = path.parent / "experiment_manifest.json"
        if manifest_path.exists():
            upstream = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(upstream, dict):
                raise ValueError(
                    f"upstream manifest must be an object: {manifest_path}"
                )
            item.update(
                manifest_path=str(manifest_path),
                manifest_sha256=file_sha256(manifest_path),
                experiment=upstream.get("experiment"),
                git_commit=upstream.get("git_commit"),
                git_dirty=upstream.get("git_dirty"),
                git_worktree_sha256=upstream.get("git_worktree_sha256"),
            )
            if upstream.get("git_dirty") is not False:
                provenance_warnings.append(
                    f"{name} upstream manifest is dirty or predates "
                    "dirty-state tracking"
                )
        else:
            item["manifest_path"] = None
            item["manifest_sha256"] = None
            provenance_warnings.append(f"{name} upstream manifest is missing")
        source_artifacts[name] = item
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "orientation": (
            "reference weekly MAE minus candidate weekly MAE; positive favors "
            "the candidate"
        ),
        "bootstrap_method": "paired circular block bootstrap",
        "rng": "numpy.random.PCG64",
        "quantile_method": "linear",
        "family_size": len(COMPARISONS),
        "multiplicity_method": "Bonferroni",
        "familywise_confidence_level": config.confidence_level,
        "adjusted_per_comparison_confidence_level": 1.0
        - (1.0 - config.confidence_level) / len(COMPARISONS),
        "resample_indices_sha256": result.resample_indices_sha256,
        "source_artifacts": source_artifacts,
        "provenance_warnings": provenance_warnings,
        "comparisons": result.comparisons.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_experiment_manifest(
        "pjme-dependence-aware-model-comparison",
        asdict(config),
        {"fold_losses": result.fold_losses},
        features=["weekly_mae"],
        boundaries={
            "holdout_start": result.fold_losses["forecast_start"].min().isoformat(),
            "holdout_end": result.fold_losses["forecast_end"].max().isoformat(),
        },
        environment={
            "source_artifacts": source_artifacts,
            "resample_indices_sha256": result.resample_indices_sha256,
        },
    )
    write_manifest(manifest, output_dir / "experiment_manifest.json")


def load_comparison_summary(path: Path) -> dict[str, object]:
    """Load a generated comparison summary JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("comparison summary must contain a JSON object")
    return payload
