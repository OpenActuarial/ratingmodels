"""Regression: the experience-to-renewal worked-example page numbers stay true."""
import numpy as np
import pandas as pd
import pytest

import actuarialpy as ap
import ratingmodels as rm
from experiencestudies import Experience


def _panel():
    rng = np.random.default_rng(42)
    months = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
    rows = []
    for seg, mm0, growth, f0, s0 in [("ppo", 5200, -0.010, 0.30, 950.0),
                                     ("hmo", 3100, +0.055, 0.34, 880.0)]:
        for i, m in enumerate(months):
            yrs = i / 12.0
            mm = mm0 * (1 + growth) ** yrs
            season = 1.0 + 0.06 * np.cos(2 * np.pi * (m.month - 1.5) / 12)
            freq = f0 * 1.02 ** yrs * season * (1 + rng.normal(0, 0.015))
            sev = s0 * 1.045 ** yrs * (1 + rng.normal(0, 0.01))
            cc = freq * mm
            rows.append((m, seg, mm, cc, cc * sev, 393.0 * mm))
    df = pd.DataFrame(rows, columns=["month", "segment", "member_months",
                                     "claim_count", "allowed", "premium"])
    df["year"] = df["month"].dt.year
    return df


def test_experience_renewal_page_numbers():
    df = _panel()
    exp = Experience(df, expense="allowed", revenue="premium",
                     exposure="member_months", date="month", count="claim_count")

    d = exp.decompose_trend(period_col="year", prior_period=2024,
                            current_period=2025, mix_by="segment").iloc[0]
    assert round(d["loss_per_exposure_current"], 4) == 339.5971
    assert round(d["frequency_trend"], 4) == 1.0156
    assert round(d["severity_trend"], 4) == 1.0465
    assert round(d["mix_trend"], 4) == 1.0007
    assert d["frequency_trend"] * d["severity_trend"] * d["mix_trend"] == pytest.approx(
        d["loss_per_exposure_trend"], rel=1e-12)

    dm = df.groupby("month", as_index=False)[["allowed", "member_months"]].sum()
    factors = ap.seasonality_factors(dm, date_col="month", value_col="allowed",
                                     exposure_col="member_months")
    assert round(factors[1], 3) == 1.049 and round(factors[7], 3) == 0.940
    dm2 = ap.deseasonalize(dm, factors, date_col="month", value_col="allowed")
    fit = ap.fit_trend(dm2, value_col="allowed_deseasonalized",
                       date_col="month", exposure_col="member_months")
    assert round(fit.annual_trend, 4) == 0.0666
    assert fit.r_squared > 0.95

    proj = float(d["loss_per_exposure_current"]) * ap.trend_factor(fit.annual_trend, months=18)
    assert proj == pytest.approx(374.10, abs=0.01)
    std = ap.full_credibility_claims(severity_cv=1.2)
    assert round(std) == 2641
    z = float(ap.limited_fluctuation_z(df.loc[df.year == 2025, "claim_count"].sum(), std))
    assert z == pytest.approx(1.0, abs=1e-12)
    assert float(ap.limited_fluctuation_z(1_200, std)) == pytest.approx(0.674, abs=5e-4)

    manual = rm.ManualRate(base_loss_cost=248.0, factors={"area": 1.06, "industry": 0.97})
    ind = rm.RateIndication(experience_loss_cost=proj, manual_loss_cost=manual.loss_cost(),
                            credibility=z, current_rate=393.0, target_loss_ratio=0.85)
    assert ind.indicated_rate() == pytest.approx(440.11, abs=0.01)
    assert ind.indicated_rate_change() == pytest.approx(0.1199, abs=5e-4)

    final_rate = rm.corridor(current_rate=393.0, indicated_rate=ind.indicated_rate(),
                             max_up=0.09, max_down=0.03)
    assert final_rate == pytest.approx(393.0 * 1.09, rel=1e-12)

    ret = rm.RetentionLoad(fixed_expense=22.0, variable_expense_ratio=0.10, profit_margin=0.02)
    pe = rm.PricingEvaluation(loss_cost=proj, current_rate=393.0, retention=ret)
    at_ind = pe.at(ind.indicated_rate() / 393.0 - 1)
    at_fin = pe.at(final_rate / 393.0 - 1)
    assert at_ind.margin_rate / at_ind.premium_rate == pytest.approx(0.0, abs=1e-4)
    assert at_fin.margin_rate / at_fin.premium_rate == pytest.approx(-0.0247, abs=5e-4)
