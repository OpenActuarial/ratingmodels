"""ExperienceExhibit: assembly worksheet, identities, and composition."""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm


@pytest.fixture()
def exhibit():
    return rm.ExperienceExhibit(
        earned_premium=[1_000_000.0, 1_100_000.0],
        losses=[700_000.0, 650_000.0],
        on_level_factors=[1.08, 1.03],
        development_factors=[1.02, 1.10],
        trend_factors=[1.05, 1.02],
        period_labels=["CY2023", "CY2024"],
    )


def test_worksheet_hand_check(exhibit):
    ex = exhibit.exhibit()
    assert list(ex.index) == ["CY2023", "CY2024"]
    assert ex.loc["CY2023", "on_level_premium"] == pytest.approx(1_080_000.0)
    adj = 700_000.0 * 1.02 * 1.05
    assert ex.loc["CY2023", "adjusted_losses"] == pytest.approx(adj)
    assert ex.loc["CY2023", "loss_ratio"] == pytest.approx(adj / 1_080_000.0)


def test_default_weighting_is_the_aggregate_ratio(exhibit):
    assert exhibit.experience_loss_ratio == pytest.approx(
        exhibit.adjusted_losses / exhibit.on_level_premium, rel=1e-12
    )


def test_to_indication_wiring_is_exact(exhibit):
    ret = rm.RetentionLoad(variable_expense_ratio=0.12, profit_margin=0.03,
                           lae_ratio=0.04)
    exposure = 24_000.0
    ind = exhibit.to_indication(
        manual_loss_cost=70.0, credibility=1.0, current_rate=90.0,
        exposure=exposure, retention=ret,
    )
    # the indication's own loss ratio reproduces the exhibit's aggregate
    assert ind.experience_loss_ratio() == pytest.approx(
        exhibit.experience_loss_ratio, rel=1e-12
    )
    # at full credibility the indicated rate IS the gross-up of the
    # assembled loss cost -- one expense algebra, no drift
    assert ind.indicated_rate() == pytest.approx(
        ret.gross_rate(exhibit.adjusted_losses / exposure), rel=1e-12
    )
    assert ind.indicated_rate_change() == pytest.approx(
        ind.indicated_rate() / 90.0 - 1.0, rel=1e-12
    )


def test_composes_with_on_level_factors():
    olf = rm.on_level_factors(
        periods=[(0.0, 1.0), (1.0, 2.0)],
        rate_changes=[(0.5, 0.10)],
        policy_term=1.0,
    )
    ex = rm.ExperienceExhibit(
        earned_premium=[500_000.0, 520_000.0],
        losses=[380_000.0, 400_000.0],
        on_level_factors=olf["on_level_factor"].to_numpy(),
    )
    tab = ex.exhibit()
    np.testing.assert_allclose(
        tab["on_level_factor"], olf["on_level_factor"], rtol=1e-12
    )
    # the transition runs one policy term past the change (to t = 1.5),
    # so period two averages 1.0875 exactly, not 1.1
    assert tab["on_level_factor"].iloc[1] == pytest.approx(1.10 / 1.0875, rel=1e-12)
    assert tab["on_level_factor"].iloc[0] > tab["on_level_factor"].iloc[1] > 1.0


def test_composes_with_chain_ladder_development():
    from actuarialpy.reserving import ChainLadder

    tri = pd.DataFrame(
        {1: [100.0, 110.0, 120.0], 2: [150.0, 166.0, np.nan],
         3: [165.0, np.nan, np.nan]},
        index=pd.Index([2022, 2023, 2024], name="origin"),
    )
    cl = ChainLadder.fit(tri)
    proj = cl.project(tri)
    ex = rm.ExperienceExhibit(
        earned_premium=[200.0, 220.0, 240.0],
        losses=proj["latest"].to_numpy(),
        development_factors=proj["development_factor"].to_numpy(),
        period_labels=proj.index,
    )
    np.testing.assert_allclose(
        ex.exhibit()["adjusted_losses"], proj["ultimate"], rtol=1e-12
    )


def test_guards():
    with pytest.raises(ValueError, match="positive and finite"):
        rm.ExperienceExhibit(earned_premium=[0.0], losses=[1.0])
    with pytest.raises(ValueError, match="length-2"):
        rm.ExperienceExhibit(earned_premium=[1.0, 2.0], losses=[1.0, 2.0],
                             trend_factors=[1.0, 1.0, 1.0])
    ex = rm.ExperienceExhibit(earned_premium=[100.0], losses=[80.0])
    with pytest.raises(ValueError, match="exposure"):
        ex.to_indication(manual_loss_cost=1.0, credibility=0.5,
                         current_rate=1.0, exposure=0.0)


def test_custom_weights_change_diagnostic_not_indication(exhibit):
    ret = rm.RetentionLoad(variable_expense_ratio=0.10)
    weighted = rm.ExperienceExhibit(
        earned_premium=exhibit.earned_premium, losses=exhibit.losses,
        on_level_factors=exhibit.on_level_factors,
        development_factors=exhibit.development_factors,
        trend_factors=exhibit.trend_factors,
        weights=[1.0, 3.0],
    )
    assert weighted.experience_loss_ratio != pytest.approx(
        exhibit.experience_loss_ratio)
    a = exhibit.to_indication(70.0, 1.0, 90.0, 24_000.0, retention=ret)
    b = weighted.to_indication(70.0, 1.0, 90.0, 24_000.0, retention=ret)
    assert a.indicated_rate() == pytest.approx(b.indicated_rate(), rel=1e-12)


def test_decomposition_reconciles_from_exhibit(exhibit):
    ret = rm.RetentionLoad(variable_expense_ratio=0.10, lae_ratio=0.05)
    ind = exhibit.to_indication(
        manual_loss_cost=70.0, credibility=0.6, current_rate=90.0,
        exposure=24_000.0, retention=ret, trend_total_factor=1.08,
    )
    decomp = ind.rate_change_decomposition()
    total = float(ind.indicated_rate() / 90.0)
    # factors include the explicit residual, so their product IS the total
    assert float(np.prod(decomp.factors.to_numpy())) == pytest.approx(
        total, rel=1e-12)
    assert float(decomp.total_factor) == pytest.approx(total, rel=1e-12)
