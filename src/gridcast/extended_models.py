import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gridcast.benchmark import (
    AUTOML_CANDIDATES,
    AUTOML_EXOGENOUS_MODEL,
    EXTENDED_POINT_MODELS,
    LIGHTGBM_EXOGENOUS_MODEL,
)
from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, VALIDATION_SPLIT, Col
from gridcast.provenance import file_sha256


@dataclass(frozen=True)
class ExtendedBenchmarkBundle:
    """Validated extended benchmark artifact bundle."""

    leaderboard: pd.DataFrame
    selection: dict[str, object]
    comparisons: pd.DataFrame
    sensitivity: pd.DataFrame
    summary: dict[str, object]


def load_extended_benchmark_bundle(directory: Path) -> ExtendedBenchmarkBundle:
    """Load and cross-validate one complete extended benchmark run."""
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = _read_object(summary_path)
    artifact_hashes = summary.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("extended summary does not contain artifact hashes")
    required_files = {
        "leaderboard.csv",
        "model_selection.json",
        "extended_comparisons.csv",
        "extended_comparison_sensitivity.csv",
    }
    if set(artifact_hashes) != required_files:
        raise ValueError("extended artifact hash set is incomplete")
    for filename, expected in artifact_hashes.items():
        path = directory / str(filename)
        if file_sha256(path) != expected:
            raise ValueError(f"extended artifact digest does not match: {filename}")

    leaderboard = pd.read_csv(directory / "leaderboard.csv")
    selection = _read_object(directory / "model_selection.json")
    comparisons = pd.read_csv(directory / "extended_comparisons.csv")
    sensitivity = pd.read_csv(directory / "extended_comparison_sensitivity.csv")
    _validate_selection(leaderboard, selection)
    _validate_comparisons(leaderboard, comparisons, sensitivity, summary)
    return ExtendedBenchmarkBundle(
        leaderboard=leaderboard,
        selection=selection,
        comparisons=comparisons,
        sensitivity=sensitivity,
        summary=summary,
    )


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"extended artifact must contain an object: {path}")
    return payload


def _validate_selection(
    leaderboard: pd.DataFrame,
    selection: dict[str, object],
) -> None:
    required = {
        Col.SPLIT,
        Col.MODEL,
        "folds",
        "observations",
        "mae",
        "rmse",
        "mase",
        "mae_improvement_vs_weekly_pct",
    }
    missing = required.difference(leaderboard.columns)
    if missing:
        raise ValueError(f"extended leaderboard missing columns: {sorted(missing)}")
    numeric = leaderboard[["mae", "rmse", "mase"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("extended leaderboard metrics must be finite")
    if selection.get("selection_split") != VALIDATION_SPLIT:
        raise ValueError("extended selection split must be validation")
    if selection.get("holdout_target_used_for_selection") is not False:
        raise ValueError("extended selection must not use holdout targets")
    selected_model = str(selection.get("selected_model", ""))
    scores = selection.get("candidate_validation_mae")
    if not isinstance(scores, dict) or set(scores) != set(AUTOML_CANDIDATES):
        raise ValueError("extended validation candidate set is incomplete")
    finite_scores = {str(model): float(value) for model, value in scores.items()}
    if not np.isfinite(list(finite_scores.values())).all():
        raise ValueError("extended validation scores must be finite")
    validation = leaderboard.loc[
        leaderboard[Col.SPLIT].eq(VALIDATION_SPLIT)
        & leaderboard[Col.MODEL].isin(AUTOML_CANDIDATES)
    ]
    observed = validation.set_index(Col.MODEL)["mae"].to_dict()
    if set(observed) != set(AUTOML_CANDIDATES) or any(
        not np.isclose(float(observed[model]), score)
        for model, score in finite_scores.items()
    ):
        raise ValueError("extended validation scores do not match leaderboard")
    expected = min(finite_scores, key=lambda model: (finite_scores[model], model))
    if selected_model != expected:
        raise ValueError("extended selected model is not validation winner")
    holdout = leaderboard.loc[leaderboard[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)]
    automl = holdout.loc[holdout[Col.MODEL].eq(AUTOML_EXOGENOUS_MODEL)]
    selected = holdout.loc[holdout[Col.MODEL].eq(selected_model)]
    if len(automl) != 1 or len(selected) != 1:
        raise ValueError("extended holdout is missing policy or selected model")
    if not np.allclose(
        automl.iloc[0][["mae", "rmse", "mase"]].to_numpy(dtype=float),
        selected.iloc[0][["mae", "rmse", "mase"]].to_numpy(dtype=float),
    ):
        raise ValueError("extended policy metrics do not match selected model")


def _validate_comparisons(
    leaderboard: pd.DataFrame,
    comparisons: pd.DataFrame,
    sensitivity: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    comparison_columns = {
        "candidate_model",
        "reference_model",
        "mean_mae_improvement_mw",
        "adjusted_ci_low_mw",
        "adjusted_ci_high_mw",
    }
    sensitivity_columns = {
        "candidate_model",
        "block_length_folds",
        "bootstrap_seed",
        "adjusted_ci_low_mw",
        "adjusted_ci_high_mw",
        "simultaneous_superiority_supported",
    }
    if comparison_columns.difference(comparisons.columns):
        raise ValueError("extended comparison schema is incomplete")
    if sensitivity_columns.difference(sensitivity.columns):
        raise ValueError("extended sensitivity schema is incomplete")
    expected = set(EXTENDED_POINT_MODELS)
    if set(comparisons["candidate_model"]) != expected or len(comparisons) != 3:
        raise ValueError("extended comparison family is incomplete")
    if comparisons["reference_model"].ne(LIGHTGBM_EXOGENOUS_MODEL).any():
        raise ValueError("extended comparison reference is invalid")
    holdout = leaderboard.loc[
        leaderboard[Col.SPLIT].eq(HISTORICAL_HOLDOUT_SPLIT)
    ].set_index(Col.MODEL)
    for row in comparisons.to_dict(orient="records"):
        candidate = str(row["candidate_model"])
        reference_mae = float(
            holdout.loc[[LIGHTGBM_EXOGENOUS_MODEL], "mae"].to_numpy(dtype=float)[0]
        )
        candidate_mae = float(holdout.loc[[candidate], "mae"].to_numpy(dtype=float)[0])
        expected_effect = reference_mae - candidate_mae
        if not np.isclose(float(row["mean_mae_improvement_mw"]), expected_effect):
            raise ValueError("extended comparison effect does not match leaderboard")
    protocol = summary.get("extended_comparison_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("extended comparison protocol is missing")
    declared_blocks = protocol.get("sensitivity_block_lengths")
    if not isinstance(declared_blocks, list) or not declared_blocks:
        raise ValueError("extended sensitivity block lengths are missing")
    expected_specs = {int(value) for value in declared_blocks}
    if sensitivity.duplicated(["candidate_model", "block_length_folds"]).any():
        raise ValueError("extended sensitivity contains duplicate specifications")
    numeric = sensitivity[["adjusted_ci_low_mw", "adjusted_ci_high_mw"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        raise ValueError("extended sensitivity intervals must be finite")
    if (numeric[:, 0] > numeric[:, 1]).any():
        raise ValueError("extended sensitivity interval is reversed")
    expected_support = numeric[:, 0] > 0.0
    observed_support = sensitivity["simultaneous_superiority_supported"].astype(bool)
    if not np.array_equal(expected_support, observed_support.to_numpy()):
        raise ValueError("extended sensitivity support flag is inconsistent")
    expected_seeds = {
        block: (
            int(protocol["bootstrap_seed"])
            if block == int(protocol["block_length_folds"])
            else int(protocol["bootstrap_seed"])
            + sorted(expected_specs).index(block)
            + 1
        )
        for block in expected_specs
    }
    if any(
        int(row["bootstrap_seed"]) != expected_seeds[int(row["block_length_folds"])]
        for row in sensitivity.to_dict(orient="records")
    ):
        raise ValueError("extended sensitivity seed is inconsistent")
    specs = sensitivity.groupby("candidate_model")["block_length_folds"].apply(set)
    if set(specs.index) != expected or any(value != expected_specs for value in specs):
        raise ValueError("extended sensitivity grid is incomplete")
    primary = sensitivity.loc[
        sensitivity["block_length_folds"].eq(int(protocol["block_length_folds"]))
    ].set_index("candidate_model")
    for row in comparisons.to_dict(orient="records"):
        candidate = str(row["candidate_model"])
        if not np.allclose(
            primary.loc[
                [candidate], ["adjusted_ci_low_mw", "adjusted_ci_high_mw"]
            ].to_numpy(dtype=float)[0],
            np.asarray(
                [row["adjusted_ci_low_mw"], row["adjusted_ci_high_mw"]],
                dtype=float,
            ),
        ):
            raise ValueError("extended primary and sensitivity intervals do not match")


def package_versions(selection: dict[str, object]) -> dict[str, str]:
    """Return strict package-version metadata from a validated selection."""
    versions = selection.get("package_versions")
    if not isinstance(versions, dict):
        raise ValueError("extended package versions are missing")
    required = {"lightgbm", "scikit-learn", "catboost", "xgboost"}
    if set(versions) != required or any(not str(value) for value in versions.values()):
        raise ValueError("extended package version set is incomplete")
    return cast(dict[str, str], versions)
