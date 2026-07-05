"""Regression: the triangle-to-indication worked-example page numbers stay true."""
import numpy as np
import pandas as pd

from actuarialpy.reserving import ChainLadder

import ratingmodels as rm


def _inputs():
    triangle = pd.DataFrame(
        {12: [4_120_000.0, 4_390_000.0, 4_610_000.0],
         24: [5_230_000.0, 5_650_000.0, np.nan],
         36: [5_490_000.0, np.nan, np.nan]},
        index=pd.Index([2022, 2023, 2024], name="origin"),
    )
    olf = rm.on_level_factors(
        periods=[("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")],
        rate_changes=[("2023-07-01", 0.08), ("2024-04-01", 0.05)],
        policy_term=1.0,
    )
    retention = rm.RetentionLoad(variable_expense_ratio=0.11,
                                 profit_margin=0.03, lae_ratio=0.05)
    return triangle, olf, retention


def test_indication_page_numbers():
    triangle, olf, retention = _inputs()
    cl = ChainLadder.fit(triangle)
    assert round(cl.age_to_age[12], 4) == 1.2785
    assert round(cl.age_to_age[24], 4) == 1.0497

    mack = cl.mack_standard_errors(triangle)
    assert round(mack.loc[2023, "ultimate"], 0) == 5_930_880
    assert round(mack.loc[2024, "ultimate"], 0) == 6_186_869
    assert round(mack.loc["Total", "ibnr"], 0) == 1_857_748
    assert round(mack.loc["Total", "se"], 0) == 171_830

    assert round(olf["on_level_factor"].iloc[0], 4) == 1.1227
    assert round(olf["on_level_factor"].iloc[1], 4) == 1.0448

    proj = cl.project(triangle).loc[[2023, 2024]]
    ex = rm.ExperienceExhibit(
        earned_premium=[7_450_000.0, 7_980_000.0],
        losses=proj["latest"].to_numpy(),
        on_level_factors=olf["on_level_factor"].to_numpy(),
        development_factors=proj["development_factor"].to_numpy(),
        trend_factors=[1.045**2, 1.045],
        period_labels=["CY2023", "CY2024"],
    )
    tab = ex.exhibit()
    assert round(tab.loc["CY2023", "on_level_premium"], 0) == 8_364_027
    assert round(tab.loc["CY2024", "adjusted_losses"], 0) == 6_465_278
    assert round(ex.experience_loss_ratio, 4) == 0.7749

    ind = ex.to_indication(manual_loss_cost=395.0, credibility=0.7,
                           current_rate=455.0, exposure=33_600.0,
                           retention=retention)
    assert round(float(ind.indicated_rate()), 2) == 473.87
    assert round(float(ind.indicated_rate_change()), 4) == 0.0415


def test_indication_page_mack_sensitivity_band():
    triangle, olf, retention = _inputs()
    cl = ChainLadder.fit(triangle)
    mack = cl.mack_standard_errors(triangle)
    proj = cl.project(triangle).loc[[2023, 2024]]
    changes = {}
    for shift in (-1.0, 1.0):
        bumped = (proj["ultimate"].to_numpy()
                  + shift * mack.loc[[2023, 2024], "se"].to_numpy())
        ex = rm.ExperienceExhibit(
            earned_premium=[7_450_000.0, 7_980_000.0], losses=bumped,
            on_level_factors=olf["on_level_factor"].to_numpy(),
            trend_factors=[1.045**2, 1.045],
            period_labels=["CY2023", "CY2024"],
        )
        ind = ex.to_indication(395.0, 0.7, 455.0, 33_600.0,
                               retention=retention)
        changes[shift] = float(ind.indicated_rate_change())
    assert round(changes[-1.0], 4) == 0.0293
    assert round(changes[1.0], 4) == 0.0536
