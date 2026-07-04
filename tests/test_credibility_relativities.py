"""Credibility-smoothed relativities and sparse-level collapsing."""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm


@pytest.fixture()
def frame():
    rng = np.random.default_rng(3)
    rows = []
    # thick, medium, thin levels with different true rates
    for lvl, n, rate in [("big", 200, 0.10), ("mid", 60, 0.14), ("thin", 6, 0.30)]:
        expo = rng.uniform(50, 150, n)
        claims = rng.poisson(rate * expo)
        for e, c in zip(expo, claims):
            rows.append((lvl, e, float(c)))
    return pd.DataFrame(rows, columns=["industry", "exposure", "claims"])


def test_limited_fluctuation_extremes(frame):
    # tiny standard -> full credibility -> relativity equals observed exactly
    full = rm.credibility_relativities(
        frame, "industry", "claims", exposure="exposure",
        method="limited_fluctuation", full_credibility=1e-9,
    )
    np.testing.assert_allclose(full["credibility"], 1.0)
    np.testing.assert_allclose(full["relativity"], full["observed"], rtol=1e-12)
    # huge standard -> no credibility -> relativity collapses to the prior
    none = rm.credibility_relativities(
        frame, "industry", "claims", exposure="exposure",
        method="limited_fluctuation", full_credibility=1e12, prior=1.0,
    )
    np.testing.assert_allclose(none["relativity"], 1.0, atol=1e-4)


def test_limited_fluctuation_z_is_square_root_rule(frame):
    fc = 5_000.0
    tab = rm.credibility_relativities(
        frame, "industry", "claims", exposure="exposure",
        method="limited_fluctuation", full_credibility=fc,
    )
    expected_z = np.minimum(1.0, np.sqrt(tab["response"] / fc))
    np.testing.assert_allclose(tab["credibility"], expected_z, rtol=1e-12)


def test_blend_identity(frame):
    tab = rm.credibility_relativities(
        frame, "industry", "claims", exposure="exposure",
        method="limited_fluctuation", full_credibility=5_000.0, prior=0.9,
    )
    manual = tab["credibility"] * tab["observed"] + (1 - tab["credibility"]) * 0.9
    np.testing.assert_allclose(tab["relativity"], manual, rtol=1e-12)


def test_buhlmann_matches_buhlmann_straub_adapter(frame):
    tab = rm.credibility_relativities(
        frame, "industry", "claims", exposure="exposure", method="buhlmann"
    )
    # reconstruct the internal long frame and run the public adapter
    work = frame.copy()
    work["value"] = work["claims"] / work["exposure"]
    work["period"] = work.groupby("industry").cumcount()
    bs = rm.buhlmann_straub(
        work, group="industry", period="period", value="value", exposure="exposure"
    )
    np.testing.assert_allclose(
        tab["relativity"],
        (bs.credibility_weighted / bs.overall_mean).reindex(tab.index),
        rtol=1e-10,
    )
    np.testing.assert_allclose(tab["credibility"], bs.credibility.reindex(tab.index), rtol=1e-10)


def test_thin_levels_shrink_more(frame):
    tab = rm.credibility_relativities(
        frame, "industry", "claims", exposure="exposure", method="buhlmann"
    )
    assert tab.loc["thin", "credibility"] < tab.loc["big", "credibility"]
    # shrink toward 1: thin level's relativity pulled below its observed value
    assert tab.loc["thin", "relativity"] < tab.loc["thin", "observed"]


def test_prior_mapping_and_missing_levels(frame):
    tab = rm.credibility_relativities(
        frame, "industry", "claims", exposure="exposure",
        method="limited_fluctuation", full_credibility=1e12,
        prior={"big": 0.8, "mid": 1.2},  # thin missing -> 1.0
    )
    assert tab.loc["big", "relativity"] == pytest.approx(0.8, abs=1e-4)
    assert tab.loc["mid", "relativity"] == pytest.approx(1.2, abs=1e-4)
    assert tab.loc["thin", "relativity"] == pytest.approx(1.0, abs=1e-4)


def test_base_level_rebase(frame):
    tab = rm.credibility_relativities(
        frame, "industry", "claims", exposure="exposure",
        method="limited_fluctuation", full_credibility=5_000.0, base_level="big",
    )
    assert tab.loc["big", "relativity"] == pytest.approx(1.0)
    assert tab.loc["big", "observed"] == pytest.approx(1.0)


def test_errors(frame):
    with pytest.raises(ValueError, match="full_credibility"):
        rm.credibility_relativities(
            frame, "industry", "claims", method="limited_fluctuation"
        )
    with pytest.raises(ValueError, match="unknown method"):
        rm.credibility_relativities(frame, "industry", "claims", method="ridge")
    with pytest.raises(ValueError, match="not found"):
        rm.credibility_relativities(frame, "segment", "claims")


def test_collapse_sparse_levels_exact():
    levels = pd.Series(["a", "a", "b", "c", "c", "c"], index=list("uvwxyz"))
    expo = pd.Series([10.0, 10.0, 1.0, 5.0, 5.0, 5.0], index=list("uvwxyz"))
    recoded, summary = rm.collapse_sparse_levels(
        levels, exposure=expo, min_exposure=10.0
    )
    assert isinstance(recoded, pd.Series)
    assert list(recoded.index) == list("uvwxyz")
    assert list(recoded) == ["a", "a", "Other", "c", "c", "c"]
    assert summary.loc["b", "collapsed"] and not summary.loc["a", "collapsed"]
    assert summary.loc["c", "exposure"] == 15.0


def test_collapse_min_n_and_array_input():
    levels = np.array(["x", "x", "x", "y"])
    recoded, summary = rm.collapse_sparse_levels(levels, min_n=2)
    assert isinstance(recoded, np.ndarray)
    assert list(recoded) == ["x", "x", "x", "Other"]
    assert summary.loc["y", "n"] == 1


def test_collapse_guards():
    levels = pd.Series(["a", "b"])
    with pytest.raises(ValueError, match="min_exposure"):
        rm.collapse_sparse_levels(levels)
    with pytest.raises(ValueError, match="every level"):
        rm.collapse_sparse_levels(levels, min_n=5)
    with pytest.raises(ValueError, match="already a kept level"):
        rm.collapse_sparse_levels(
            pd.Series(["keep"] * 5 + ["thin"]), min_n=2, other_label="keep"
        )
