import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.columns import HISTORICAL_HOLDOUT_SPLIT, Col
from gridcast.foundation_models import TIMESFM_2P5, TIMESFM_3
from gridcast.model_comparison import (
    COMPARISONS,
    DAILY_NAIVE,
    LIGHTGBM,
    ComparisonConfig,
    build_fold_losses,
    circular_block_indices,
    moving_block_indices,
    run_model_comparison,
    write_comparison_artifacts,
)


def _forecasts(*, folds: int = 4, observations: int = 2) -> dict[str, pd.DataFrame]:
    errors = {
        DAILY_NAIVE: 4.0,
        LIGHTGBM: 3.0,
        TIMESFM_2P5.model_name: 2.0,
        TIMESFM_3.model_name: 1.0,
    }
    result: dict[str, pd.DataFrame] = {}
    for model, error in errors.items():
        rows: list[dict[str, object]] = []
        for fold in range(1, folds + 1):
            cutoff = pd.Timestamp("2024-01-01") + pd.Timedelta(
                hours=observations * (fold - 1)
            )
            for step in range(observations):
                rows.append(
                    {
                        Col.TIMESTAMP: cutoff + pd.Timedelta(hours=step + 1),
                        Col.TARGET: 100.0,
                        Col.PREDICTION: 100.0 - error,
                        Col.MODEL: model,
                        Col.SPLIT: HISTORICAL_HOLDOUT_SPLIT,
                        Col.FOLD: fold,
                        Col.CUTOFF: cutoff,
                    }
                )
        result[model] = pd.DataFrame(rows)
    return result


def _config() -> ComparisonConfig:
    return ComparisonConfig(
        bootstrap_replicates=100,
        block_length_folds=2,
        seed=42,
        sensitivity_block_lengths=(1, 2, 4),
        expected_folds=4,
        observations_per_fold=2,
    )


def test_fold_losses_are_strictly_paired_and_ordered() -> None:
    losses = build_fold_losses(_forecasts(), expected_folds=4, observations_per_fold=2)

    assert len(losses) == 16
    assert losses.groupby(Col.MODEL)["mae_mw"].first().to_dict() == {
        DAILY_NAIVE: 4.0,
        LIGHTGBM: 3.0,
        TIMESFM_2P5.model_name: 2.0,
        TIMESFM_3.model_name: 1.0,
    }


@pytest.mark.parametrize(
    "invalid",
    ["duplicate", "mismatch", "nonfinite", "gap", "cutoff", "overlap"],
)
def test_fold_losses_reject_invalid_pairing(invalid: str) -> None:
    forecasts = _forecasts()
    frame = forecasts[TIMESFM_3.model_name]
    if invalid == "duplicate":
        forecasts[TIMESFM_3.model_name] = pd.concat([frame, frame.iloc[[0]]])
    elif invalid == "mismatch":
        forecasts[TIMESFM_3.model_name] = frame.assign(
            **{Col.TIMESTAMP: frame[Col.TIMESTAMP] + pd.Timedelta(hours=1)}
        )
    elif invalid == "nonfinite":
        forecasts[TIMESFM_3.model_name].loc[0, Col.PREDICTION] = np.nan
    elif invalid == "gap":
        forecasts[TIMESFM_3.model_name].loc[1, Col.TIMESTAMP] += pd.Timedelta(hours=1)
    elif invalid == "cutoff":
        forecasts[TIMESFM_3.model_name].loc[1, Col.CUTOFF] += pd.Timedelta(hours=1)
    else:
        later_folds = forecasts[TIMESFM_3.model_name][Col.FOLD].gt(1)
        forecasts[TIMESFM_3.model_name].loc[later_folds, Col.TIMESTAMP] -= pd.Timedelta(
            hours=1
        )
        forecasts[TIMESFM_3.model_name].loc[later_folds, Col.CUTOFF] -= pd.Timedelta(
            hours=1
        )

    message = "folds must be contiguous" if invalid == "overlap" else None
    with pytest.raises(ValueError, match=message):
        build_fold_losses(forecasts, expected_folds=4, observations_per_fold=2)


def test_block_indices_are_deterministic_and_preserve_blocks() -> None:
    first = circular_block_indices(4, 2, 3, np.random.default_rng(7))
    second = circular_block_indices(4, 2, 3, np.random.default_rng(7))
    moving = moving_block_indices(4, 2, 20, np.random.default_rng(7))

    assert np.array_equal(first, second)
    assert np.all((first[:, 1::2] - first[:, ::2]) % 4 == 1)
    assert np.all(moving[:, 1::2] - moving[:, ::2] == 1)
    assert moving.min() >= 0 and moving.max() < 4


def test_comparison_orientation_ties_and_intervals_are_deterministic() -> None:
    forecasts = _forecasts()
    forecasts[TIMESFM_3.model_name].loc[
        forecasts[TIMESFM_3.model_name][Col.FOLD].eq(1), Col.PREDICTION
    ] = 98.0

    first = run_model_comparison(forecasts, _config())
    second = run_model_comparison(forecasts, _config())
    lightgbm = first.comparisons.loc[
        first.comparisons["comparison_id"].eq("lightgbm_vs_daily_naive")
    ].iloc[0]
    version = first.comparisons.loc[
        first.comparisons["comparison_id"].eq("timesfm_3_vs_timesfm_2_5")
    ].iloc[0]

    assert list(first.comparisons["comparison_id"]) == [row[0] for row in COMPARISONS]
    assert len(first.comparisons) == 6
    assert lightgbm["mean_mae_improvement_mw"] == pytest.approx(1.0)
    assert lightgbm["ci_low_mw"] == pytest.approx(1.0)
    assert lightgbm["ci_high_mw"] == pytest.approx(1.0)
    assert lightgbm["simultaneous_superiority_supported"] is np.True_
    assert lightgbm["familywise_confidence_level"] == pytest.approx(0.95)
    assert lightgbm["adjusted_per_comparison_confidence_level"] == pytest.approx(
        1.0 - 0.05 / 6.0
    )
    assert version["wins"] == 3
    assert version["ties"] == 1
    assert version["weekly_win_rate"] == pytest.approx(0.875)
    assert first.resample_indices_sha256 == second.resample_indices_sha256
    pd.testing.assert_frame_equal(first.comparisons, second.comparisons)
    primary = first.sensitivity.loc[first.sensitivity["primary"]]
    specs = first.sensitivity[
        ["bootstrap_method", "block_length_folds", "bootstrap_seed"]
    ].drop_duplicates()
    assert len(first.sensitivity) == 24
    assert specs.to_records(index=False).tolist() == [
        ("circular", 1, 43),
        ("circular", 2, 42),
        ("circular", 4, 45),
        ("moving", 2, 46),
    ]
    assert (
        first.sensitivity.groupby(["bootstrap_method", "block_length_folds"])[
            "resample_indices_sha256"
        ]
        .nunique()
        .eq(1)
        .all()
    )
    pd.testing.assert_series_equal(
        primary.set_index("comparison_id")["ci_low_mw"],
        first.comparisons.set_index("comparison_id")["ci_low_mw"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        primary.set_index("comparison_id")["adjusted_ci_low_mw"],
        first.comparisons.set_index("comparison_id")["adjusted_ci_low_mw"],
        check_names=False,
    )


def test_config_rejects_invalid_block_dimensions() -> None:
    with pytest.raises(ValueError, match="primary block length"):
        ComparisonConfig(
            expected_folds=4,
            block_length_folds=5,
            sensitivity_block_lengths=(1, 2, 4),
        )
    with pytest.raises(ValueError, match="non-negative"):
        ComparisonConfig(seed=-1)
    with pytest.raises(ValueError, match="included in sensitivity"):
        ComparisonConfig(block_length_folds=4, sensitivity_block_lengths=(2, 6))
    with pytest.raises(ValueError, match="unique"):
        ComparisonConfig(sensitivity_block_lengths=(2, 4, 4))


def test_writer_records_source_hashes_and_upstream_warnings(tmp_path: Path) -> None:
    forecasts = _forecasts()
    result = run_model_comparison(forecasts, _config())
    source_paths: dict[str, Path] = {}
    for name, frame in forecasts.items():
        path = tmp_path / name / "forecasts.parquet"
        path.parent.mkdir()
        frame.to_parquet(path, index=False)
        source_paths[name] = path
    manifest_path = source_paths[DAILY_NAIVE].parent / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps({"experiment": "upstream", "git_commit": "abc", "git_dirty": True}),
        encoding="utf-8",
    )
    output = tmp_path / "comparison"

    write_comparison_artifacts(result, _config(), output, source_paths)

    assert {path.name for path in output.iterdir()} == {
        "fold_losses.parquet",
        "paired_comparisons.csv",
        "bootstrap_sensitivity.csv",
        "summary.json",
        "experiment_manifest.json",
    }
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["comparisons"]) == 6
    assert summary["familywise_confidence_level"] == pytest.approx(0.95)
    assert summary["adjusted_per_comparison_confidence_level"] == pytest.approx(
        1.0 - 0.05 / 6.0
    )
    assert summary["source_artifacts"][DAILY_NAIVE]["sha256"] is not None
    assert any("dirty" in warning for warning in summary["provenance_warnings"])
