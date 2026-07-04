"""Validation splits: random, group-preserving, temporal."""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm


@pytest.fixture()
def frame():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "group": rng.choice([f"g{i}" for i in range(40)], n),
            "exposure": rng.uniform(1.0, 100.0, n),
            "month": pd.date_range("2024-01-01", periods=n, freq="D"),
            "year": np.repeat([2023, 2024, 2025, 2026], 100),
            "x": rng.normal(size=n),
        }
    )


def test_random_split_sizes_and_disjoint(frame):
    train, test = rm.random_split(frame, test_fraction=0.25, random_state=1)
    assert len(train) + len(test) == len(frame)
    assert len(test) == round(0.25 * len(frame))
    assert set(train.index).isdisjoint(test.index)
    # original row order preserved on both sides
    assert list(train.index) == sorted(train.index)
    assert list(test.index) == sorted(test.index)


def test_random_split_reproducible(frame):
    _, t1 = rm.random_split(frame, random_state=42)
    _, t2 = rm.random_split(frame, random_state=42)
    pd.testing.assert_frame_equal(t1, t2)


def test_random_split_bad_fraction(frame):
    with pytest.raises(ValueError):
        rm.random_split(frame, test_fraction=0.0)
    with pytest.raises(ValueError):
        rm.random_split(frame, test_fraction=1.0)


def test_group_split_keeps_groups_whole(frame):
    train, test = rm.group_split(frame, group="group", test_fraction=0.3, random_state=2)
    assert set(train["group"]).isdisjoint(set(test["group"]))
    assert len(train) + len(test) == len(frame)
    assert len(set(train["group"])) >= 1 and len(set(test["group"])) >= 1


def test_group_split_weighted_share_reaches_target(frame):
    train, test = rm.group_split(
        frame, group="group", test_fraction=0.3, weights="exposure", random_state=3
    )
    share = test["exposure"].sum() / frame["exposure"].sum()
    assert share >= 0.3
    # overshoot bounded by one group's weight
    max_group = frame.groupby("group")["exposure"].sum().max()
    assert share <= 0.3 + max_group / frame["exposure"].sum() + 1e-12


def test_group_split_reproducible(frame):
    _, t1 = rm.group_split(frame, group="group", random_state=7)
    _, t2 = rm.group_split(frame, group="group", random_state=7)
    pd.testing.assert_frame_equal(t1, t2)


def test_group_split_needs_two_groups():
    df = pd.DataFrame({"group": ["only"] * 5, "x": range(5)})
    with pytest.raises(ValueError, match="2 distinct groups"):
        rm.group_split(df, group="group")


def test_temporal_split_datetime_cutoff(frame):
    train, test = rm.temporal_split(frame, date="month", cutoff="2024-07-01")
    assert (train["month"] < pd.Timestamp("2024-07-01")).all()
    assert (test["month"] >= pd.Timestamp("2024-07-01")).all()
    # the boundary date itself belongs to the test side
    assert pd.Timestamp("2024-07-01") in set(test["month"])
    assert len(train) + len(test) == len(frame)


def test_temporal_split_ordinal_column(frame):
    train, test = rm.temporal_split(frame, date="year", cutoff=2025)
    assert set(train["year"]) == {2023, 2024}
    assert set(test["year"]) == {2025, 2026}


def test_temporal_split_out_of_range_raises(frame):
    with pytest.raises(ValueError, match="one side"):
        rm.temporal_split(frame, date="year", cutoff=1990)
    with pytest.raises(ValueError, match="one side"):
        rm.temporal_split(frame, date="year", cutoff=2050)
