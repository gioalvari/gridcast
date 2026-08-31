import pandas as pd
import pytest

from gridcast.columns import Col
from gridcast.data import generate_synthetic_load


def test_synthetic_load_is_reproducible_and_hourly() -> None:
    first = generate_synthetic_load(periods=24 * 14, seed=7)
    second = generate_synthetic_load(periods=24 * 14, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [Col.TIMESTAMP, Col.TARGET]
    assert first[Col.TIMESTAMP].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert first[Col.TARGET].gt(0).all()


def test_synthetic_load_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="two weeks"):
        generate_synthetic_load(periods=100)
