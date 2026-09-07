from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from gridcast.benchmark import (
    AUTOML_EXOGENOUS_MODEL,
    LIGHTGBM_EXOGENOUS_MODEL,
)
from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, VALIDATION_SPLIT, Col
from gridcast.extended_models import (
    _read_object,
    _validate_comparisons,
    _validate_selection,
    load_extended_benchmark_bundle,
    package_versions,
)


def _bundle_tables() -> tuple[
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    validation_scores = {
        LIGHTGBM_EXOGENOUS_MODEL: 3900.0,
        "hist_gradient_boosting_exogenous": 3950.0,
        "catboost_exogenous": 3800.0,
        "xgboost_exogenous": 4100.0,
    }
    holdout_scores = {
        LIGHTGBM_EXOGENOUS_MODEL: 2900.0,
        "hist_gradient_boosting_exogenous": 2895.0,
        "catboost_exogenous": 2840.0,
        "xgboost_exogenous": 2915.0,
        AUTOML_EXOGENOUS_MODEL: 2840.0,
    }
    rows: list[dict[str, object]] = []
    for model, mae in validation_scores.items():
        rows.append(
            {
                Col.SPLIT: VALIDATION_SPLIT,
                Col.MODEL: model,
                "folds": 12,
                "observations": 2016,
                "mae": mae,
                "rmse": mae + 100.0,
                "mase": 1.1,
                "mae_improvement_vs_weekly_pct": 10.0,
            }
        )
    for model, mae in holdout_scores.items():
        rows.append(
            {
                Col.SPLIT: HISTORICAL_HOLDOUT_SPLIT,
                Col.MODEL: model,
                "folds": 52,
                "observations": 8736,
                "mae": mae,
                "rmse": 3900.0
                if "catboost" in model or model == AUTOML_EXOGENOUS_MODEL
                else 4000.0,
                "mase": 0.94
                if "catboost" in model or model == AUTOML_EXOGENOUS_MODEL
                else 0.96,
                "mae_improvement_vs_weekly_pct": 20.0,
            }
        )
    leaderboard = pd.DataFrame(rows)
    selection: dict[str, object] = {
        "selection_split": VALIDATION_SPLIT,
        "holdout_target_used_for_selection": False,
        "selected_model": "catboost_exogenous",
        "candidate_validation_mae": validation_scores,
        "package_versions": {
            "lightgbm": "4.7.0",
            "scikit-learn": "1.9.0",
            "catboost": "1.2.10",
            "xgboost": "3.4.1",
        },
    }
    gains = {
        "hist_gradient_boosting_exogenous": 5.0,
        "catboost_exogenous": 60.0,
        "xgboost_exogenous": -15.0,
    }
    comparisons = pd.DataFrame(
        [
            {
                "candidate_model": model,
                "reference_model": LIGHTGBM_EXOGENOUS_MODEL,
                "mean_mae_improvement_mw": gain,
                "adjusted_ci_low_mw": gain - 90.0,
                "adjusted_ci_high_mw": gain + 90.0,
            }
            for model, gain in gains.items()
        ]
    )
    blocks = [2, 4, 6, 8, 13, 26]
    sensitivity = pd.DataFrame(
        [
            {
                "candidate_model": model,
                "block_length_folds": block,
                "bootstrap_seed": (
                    20_260_906 if block == 4 else 20_260_906 + blocks.index(block) + 1
                ),
                "adjusted_ci_low_mw": gain - (90.0 if block == 4 else 100.0),
                "adjusted_ci_high_mw": gain + (90.0 if block == 4 else 100.0),
                "simultaneous_superiority_supported": False,
            }
            for model, gain in gains.items()
            for block in blocks
        ]
    )
    summary: dict[str, object] = {
        "extended_comparison_protocol": {
            "block_length_folds": 4,
            "bootstrap_seed": 20_260_906,
            "sensitivity_block_lengths": blocks,
        }
    }
    return leaderboard, selection, comparisons, sensitivity, summary


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_column", "missing columns"),
        ("nonfinite_metric", "metrics must be finite"),
        ("selection_split", "split must be validation"),
        ("holdout_used", "must not use holdout"),
        ("candidate_set", "candidate set is incomplete"),
        ("nonnumeric_score", "scores must be numeric"),
        ("nonfinite_score", "scores must be finite"),
        ("score_mismatch", "scores do not match"),
        ("wrong_winner", "not validation winner"),
        ("missing_policy", "missing policy"),
        ("policy_mismatch", "do not match selected"),
    ],
)
def test_selection_validator_rejects_invalid_contract(
    mutation: str,
    message: str,
) -> None:
    leaderboard, selection, _, _, _ = _bundle_tables()
    selection = deepcopy(selection)
    if mutation == "missing_column":
        leaderboard = leaderboard.drop(columns="rmse")
    elif mutation == "nonfinite_metric":
        leaderboard.loc[0, "mae"] = np.inf
    elif mutation == "selection_split":
        selection["selection_split"] = HISTORICAL_HOLDOUT_SPLIT
    elif mutation == "holdout_used":
        selection["holdout_target_used_for_selection"] = True
    elif mutation == "candidate_set":
        selection["candidate_validation_mae"] = {LIGHTGBM_EXOGENOUS_MODEL: 1.0}
    elif mutation == "nonnumeric_score":
        scores = cast_scores(selection)
        scores["catboost_exogenous"] = None  # type: ignore[assignment]
    elif mutation == "nonfinite_score":
        scores = cast_scores(selection)
        scores["catboost_exogenous"] = np.nan
    elif mutation == "score_mismatch":
        scores = cast_scores(selection)
        scores["catboost_exogenous"] += 1.0
    elif mutation == "wrong_winner":
        selection["selected_model"] = LIGHTGBM_EXOGENOUS_MODEL
    elif mutation == "missing_policy":
        leaderboard = leaderboard.loc[
            ~leaderboard[Col.MODEL].eq(AUTOML_EXOGENOUS_MODEL)
        ]
    else:
        mask = leaderboard[Col.MODEL].eq(AUTOML_EXOGENOUS_MODEL)
        leaderboard.loc[mask, "mae"] += 1.0

    with pytest.raises(ValueError, match=message):
        _validate_selection(leaderboard, selection)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("comparison_schema", "comparison schema"),
        ("sensitivity_schema", "sensitivity schema"),
        ("family", "family is incomplete"),
        ("reference", "reference is invalid"),
        ("effect", "effect does not match"),
        ("protocol", "protocol is missing"),
        ("blocks", "block lengths are missing"),
        ("duplicate", "duplicate specifications"),
        ("nonfinite", "intervals must be finite"),
        ("reversed", "interval is reversed"),
        ("support", "support flag is inconsistent"),
        ("seed", "seed is inconsistent"),
        ("grid", "grid is incomplete"),
        ("primary", "primary and sensitivity"),
    ],
)
def test_comparison_validator_rejects_invalid_contract(
    mutation: str,
    message: str,
) -> None:
    leaderboard, _, comparisons, sensitivity, summary = _bundle_tables()
    comparisons = comparisons.copy()
    sensitivity = sensitivity.copy()
    summary = deepcopy(summary)
    if mutation == "comparison_schema":
        comparisons = comparisons.drop(columns="reference_model")
    elif mutation == "sensitivity_schema":
        sensitivity = sensitivity.drop(columns="bootstrap_seed")
    elif mutation == "family":
        comparisons = comparisons.iloc[:-1]
    elif mutation == "reference":
        comparisons.loc[0, "reference_model"] = "other"
    elif mutation == "effect":
        comparisons.loc[0, "mean_mae_improvement_mw"] += 1.0
    elif mutation == "protocol":
        summary = {}
    elif mutation == "blocks":
        summary["extended_comparison_protocol"] = {}
    elif mutation == "duplicate":
        sensitivity = pd.concat([sensitivity, sensitivity.iloc[[0]]])
    elif mutation == "nonfinite":
        sensitivity.loc[0, "adjusted_ci_low_mw"] = np.nan
    elif mutation == "reversed":
        sensitivity.loc[0, "adjusted_ci_low_mw"] = 1000.0
    elif mutation == "support":
        sensitivity.loc[0, "simultaneous_superiority_supported"] = True
    elif mutation == "seed":
        sensitivity.loc[0, "bootstrap_seed"] = 1
    elif mutation == "grid":
        sensitivity = sensitivity.loc[sensitivity["block_length_folds"].ne(26)]
    else:
        sensitivity.loc[
            sensitivity["block_length_folds"].eq(4), "adjusted_ci_low_mw"
        ] += 1.0

    with pytest.raises(ValueError, match=message):
        _validate_comparisons(leaderboard, comparisons, sensitivity, summary)


def test_read_object_and_package_versions_validate_types(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "value.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        _read_object(path)
    with pytest.raises(ValueError, match="versions are missing"):
        package_versions({})
    with pytest.raises(ValueError, match="version set is incomplete"):
        package_versions({"package_versions": {"catboost": "1"}})


@pytest.mark.parametrize("summary", [None, {}, {"artifact_sha256": {}}])
def test_bundle_loader_requires_complete_summary_marker(
    tmp_path: object,
    summary: dict[str, object] | None,
) -> None:
    from pathlib import Path

    directory = Path(str(tmp_path))
    if summary is not None:
        (directory / "summary.json").write_text(
            __import__("json").dumps(summary), encoding="utf-8"
        )

    with pytest.raises((FileNotFoundError, ValueError)):
        load_extended_benchmark_bundle(directory)


def cast_scores(selection: dict[str, object]) -> dict[str, float]:
    """Return the mutable score dictionary used by a test fixture."""
    scores = selection["candidate_validation_mae"]
    assert isinstance(scores, dict)
    return scores
