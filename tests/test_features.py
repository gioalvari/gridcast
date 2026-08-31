import numpy as np
import pytest

from gridcast.columns import Col
from gridcast.data import generate_synthetic_load
from gridcast.features import Feature, build_exogenous_features, build_forecast_features


def test_target_features_are_delayed_by_at_least_one_week() -> None:
    data = generate_synthetic_load(periods=24 * 21)
    features = build_forecast_features(data)

    assert features.loc[24 * 7, Feature.LAG_168] == data.loc[0, Col.TARGET]
    assert features.loc[24 * 14, Feature.LAG_336] == data.loc[0, Col.TARGET]
    assert features.loc[24 * 7 + 23, Feature.MEAN_24_AT_168] == pytest.approx(
        np.mean(data.loc[:23, Col.TARGET])
    )


def test_future_target_changes_do_not_change_current_horizon_features() -> None:
    data = generate_synthetic_load(periods=24 * 21)
    origin = 24 * 14
    baseline = build_forecast_features(data).iloc[origin : origin + 24 * 7]
    changed = data.copy()
    changed.loc[origin:, Col.TARGET] *= 10.0

    modified = build_forecast_features(changed).iloc[origin : origin + 24 * 7]

    assert baseline.equals(modified)


def test_exogenous_features_are_known_at_weekly_origin() -> None:
    data = generate_synthetic_load(periods=24 * 800, start="2015-01-01")
    weather = data[[Col.TIMESTAMP]].assign(
        **{Col.TEMPERATURE: np.sin(np.arange(len(data)) / 100.0) * 15.0}
    )
    origin = 24 * 730
    baseline = build_exogenous_features(data, weather).iloc[origin : origin + 24 * 7]
    changed = weather.copy()
    changed.loc[origin:, Col.TEMPERATURE] = 100.0

    modified = build_exogenous_features(data, changed).iloc[origin : origin + 24 * 7]

    assert baseline.equals(modified)
    assert baseline[Feature.TEMPERATURE_CLIMATOLOGY].notna().all()
    assert baseline[Feature.IS_HOLIDAY].isin([0.0, 1.0]).all()


def test_exogenous_features_require_complete_weather() -> None:
    data = generate_synthetic_load(periods=24 * 14)
    weather = data[[Col.TIMESTAMP]].iloc[:-1].assign(**{Col.TEMPERATURE: 10.0})

    with pytest.raises(ValueError, match="cover every"):
        build_exogenous_features(data, weather)
