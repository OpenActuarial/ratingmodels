"""Rate dislocation banding and constraint impact, on hand-checked numbers."""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm


def test_dislocation_hand_example():
    current = np.array([100.0, 100.0, 100.0, 100.0])
    proposed = np.array([88.0, 97.0, 103.0, 120.0])
    exposure = np.array([10.0, 20.0, 30.0, 40.0])
    tab = rm.rate_dislocation(current, proposed, exposure=exposure)

    assert tab.loc["below -10%", "n"] == 1
    assert tab.loc["below -10%", "current_premium"] == pytest.approx(1000.0)
    assert tab.loc["below -10%", "avg_change"] == pytest.approx(-0.12)
    assert tab.loc["-5% to +0%", "n"] == 1
    assert tab.loc["+0% to +5%", "n"] == 1
    assert tab.loc["above +10%", "n"] == 1
    assert tab.loc["above +10%", "proposed_premium"] == pytest.approx(4800.0)
    # empty bands are kept, at zero
    assert tab.loc["-10% to -5%", "n"] == 0
    assert tab.loc["-10% to -5%", "exposure"] == 0.0

    total = tab.loc["All"]
    assert total["n"] == 4
    assert total["current_premium"] == pytest.approx(10_000.0)
    assert total["proposed_premium"] == pytest.approx(10_710.0)
    assert total["avg_change"] == pytest.approx(0.071)
    assert total["exposure_share"] == pytest.approx(1.0)
    assert tab["exposure_share"].iloc[:-1].sum() == pytest.approx(1.0)


def test_dislocation_boundary_falls_in_lower_band():
    tab = rm.rate_dislocation([100.0, 100.0], [95.0, 105.0], include_total=False)
    assert tab.loc["-10% to -5%", "n"] == 1   # exactly -5% -> (low, high]
    assert tab.loc["+0% to +5%", "n"] == 1    # exactly +5%


def test_dislocation_without_exposure_uses_rates_as_premium():
    tab = rm.rate_dislocation([100.0, 200.0], [110.0, 220.0])
    assert tab.loc["All", "current_premium"] == pytest.approx(300.0)
    assert tab.loc["All", "exposure"] == 2.0


def test_dislocation_input_guards():
    with pytest.raises(ValueError, match="positive"):
        rm.rate_dislocation([0.0, 100.0], [90.0, 100.0])
    with pytest.raises(ValueError, match="distinct"):
        rm.rate_dislocation([100.0], [105.0], bands=[0.05, 0.05])


def test_constraint_impact_hand_example():
    ind = np.array([110.0, 100.0, 90.0])
    prop = np.array([105.0, 100.0, 95.0])
    expo = np.array([1.0, 2.0, 3.0])
    cur = np.array([100.0, 100.0, 100.0])
    s = rm.constraint_impact(ind, prop, exposure=expo, current_rate=cur)
    assert s["n"] == 3
    assert s["n_below"] == 1 and s["n_above"] == 1
    assert s["premium_shortfall"] == pytest.approx(5.0)
    assert s["premium_excess"] == pytest.approx(15.0)
    assert s["indicated_premium"] == pytest.approx(580.0)
    assert s["proposed_premium"] == pytest.approx(590.0)
    assert s["remaining_change"] == pytest.approx(580 / 590 - 1)
    assert s["indicated_change"] == pytest.approx(580 / 600 - 1)
    assert s["realized_change"] == pytest.approx(590 / 600 - 1)


def test_constraint_impact_grouped():
    ind = np.array([110.0, 100.0, 90.0, 120.0])
    prop = np.array([105.0, 100.0, 95.0, 110.0])
    g = np.array(["east", "east", "west", "west"])
    tab = rm.constraint_impact(ind, prop, by=g)
    assert isinstance(tab, pd.DataFrame)
    assert list(tab.index) == ["east", "west"]
    assert tab.loc["east", "premium_shortfall"] == pytest.approx(5.0)
    assert tab.loc["west", "premium_shortfall"] == pytest.approx(10.0)
    assert tab["n"].dtype.kind == "i"


def test_constraint_impact_no_gap_is_all_zero():
    r = np.array([100.0, 105.0])
    s = rm.constraint_impact(r, r)
    assert s["n_below"] == 0 and s["n_above"] == 0
    assert s["premium_shortfall"] == 0.0 and s["premium_excess"] == 0.0
    assert s["remaining_change"] == pytest.approx(0.0)
