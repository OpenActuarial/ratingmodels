"""Regression: the pricing-a-book worked-example page numbers stay true."""
import numpy as np
import pandas as pd

import ratingmodels as rm


def _book():
    book = pd.DataFrame(
        {
            "exposure": [9_600.0, 14_400.0, 6_000.0],
            "current": [545.0, 560.0, 530.0],
            "base": [420.0, 435.0, 410.0],
            "area": [1.05, 0.98, 1.12],
            "industry": [1.10, 1.00, 0.95],
            "n_claims": [820.0, 1_450.0, 260.0],
        },
        index=pd.Index(["G1", "G2", "G3"], name="group"),
    )
    large = pd.DataFrame({"group": ["G1", "G1", "G2", "G3", "G3"],
                          "amount": [390e3, 310e3, 420e3, 610e3, 260e3]})
    bulk = pd.Series([3.65e6, 5.88e6, 2.03e6], index=book.index)
    return book, large, bulk


def test_pricing_a_book_page_numbers():
    book, large, bulk = _book()

    _, excess = rm.pool_claims(large["amount"], 250_000, by=large["group"])
    incurred = bulk + large.groupby("group")["amount"].sum()
    assert excess.round(0).to_dict() == {"G1": 200_000, "G2": 170_000, "G3": 370_000}
    assert incurred.round(0).to_dict() == {"G1": 4_350_000, "G2": 6_300_000, "G3": 2_900_000}

    retention = rm.RetentionLoad(fixed_expense=12.0, variable_expense_ratio=0.11,
                                 profit_margin=0.03, lae_ratio=0.02)
    experience = rm.ExperienceRate(
        incurred_claims=incurred, exposure=book["exposure"],
        trend_annual=0.07, trend_years=1.5,
        pooled_excess=excess, pooling_charge=28.0, retention=retention,
    )
    manual = rm.ManualRate(book["base"],
                           {"area": book["area"], "industry": book["industry"]},
                           retention=retention)
    z = rm.limited_fluctuation_credibility(book["n_claims"], n_full=1_082)

    assert round(experience.trend_factor(), 4) == 1.1068
    assert experience.loss_cost().round(2).to_dict() == {"G1": 506.47, "G2": 499.17, "G3": 494.71}
    assert manual.loss_cost().round(2).to_dict() == {"G1": 485.10, "G2": 426.30, "G3": 436.24}
    assert z.round(3).to_dict() == {"G1": 0.871, "G2": 1.000, "G3": 0.490}

    indication = rm.RateIndication(
        experience_loss_cost=experience.loss_cost(),
        manual_loss_cost=manual.loss_cost(),
        credibility=z,
        current_rate=book["current"],
        trend_total_factor=experience.trend_factor(),
        retention=retention,
    )
    assert indication.blended_loss_cost().round(2).to_dict() == {"G1": 503.70, "G2": 499.17, "G3": 464.90}
    assert indication.indicated_rate().round(2).to_dict() == {"G1": 611.37, "G2": 605.99, "G3": 565.35}
    assert indication.indicated_rate_change().round(4).to_dict() == {"G1": 0.1218, "G2": 0.0821, "G3": 0.0667}

    d = indication.rate_change_decomposition()
    g1 = d.to_frame().round(4).loc["G1"]
    assert g1.loc["trend", "pct_point_contribution"] == 0.1075
    assert g1.loc["experience", "pct_point_contribution"] == 0.0399
    assert g1.loc["residual", "pct_point_contribution"] == -0.0257
    assert np.allclose(d.contributions.sum(axis=1), np.asarray(d.total_factor) - 1)

    action = rm.renew(book["current"], indication.indicated_rate(),
                      cap=pd.Series([0.10, 0.10, 0.12], index=book.index), floor=0.0)
    assert action.proposed_rate.round(2).to_dict() == {"G1": 599.50, "G2": 605.99, "G3": 565.35}
    assert action.capped.tolist() == [True, False, False]

    evaluation = rm.PricingEvaluation(
        loss_cost=indication.blended_loss_cost(),
        current_rate=book["current"],
        retention=retention,
        exposure=book["exposure"],
        persistency=pd.Series([0.90, 0.95, 0.80], index=book.index),
    )
    tidy = rm.scenario_frame(evaluation, {
        "formula": indication.indicated_rate_change(),
        "issued": action.proposed_change,
        "plan": 0.05,
    })
    pivot = tidy.pivot(index="case", columns="scenario", values="margin_ratio").round(4)
    assert (pivot["formula"] == 0.03).all()          # the algebra closes
    assert pivot.loc["G1", "issued"] == 0.0130
    assert pivot.loc["G1", "plan"] == -0.0288

    uplift = rm.uplift_for_target_margin(evaluation, action.proposed_change,
                                         target_margin=0.03)
    assert round(uplift * 100, 4) == 0.6332

    # vector renewal equals the scalar loop exactly
    caps = pd.Series([0.10, 0.10, 0.12], index=book.index)
    loop = [rm.renew(float(book["current"][g]), float(indication.indicated_rate()[g]),
                     cap=float(caps[g]), floor=0.0).proposed_rate for g in book.index]
    assert float(np.max(np.abs(action.proposed_rate.to_numpy() - np.asarray(loop)))) == 0.0
