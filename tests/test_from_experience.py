"""ExperienceRate / ExperienceExhibit construction from the canonical Experience."""
import numpy as np
import pandas as pd
import pytest
from actuarialpy import Experience

import ratingmodels as rm


def _frame():
    months = pd.date_range("2025-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(7)
    rows = []
    for m in months:
        for member in range(1, 21):
            claims = float(rng.gamma(2.0, 400.0))
            rows.append((m, "A" if member <= 12 else "B", f"M{member:03d}",
                         claims, 900.0, 1.0))
    return pd.DataFrame(rows, columns=[
        "month", "segment", "member_id", "claims", "premium", "member_months"])


def _exp():
    return Experience(_frame(), expense="claims", revenue="premium",
                      exposure="member_months", date="month",
                      dimensions="segment")


def test_from_experience_matches_scalar_constructor():
    exp = _exp()
    df = exp.data
    totals = df.groupby("member_id")["claims"].sum()
    point = float(totals.quantile(0.90))
    excess = float((totals - totals.clip(upper=point)).sum())
    built = rm.ExperienceRate.from_experience(
        exp, pooling_point=point, claimant_col="member_id",
        trend_annual=0.075, trend_years=1.5, pooling_charge=4.0)
    manual = rm.ExperienceRate(
        incurred_claims=float(df["claims"].sum()),
        exposure=float(df["member_months"].sum()),
        trend_annual=0.075, trend_years=1.5,
        pooled_excess=excess, pooling_charge=4.0)
    assert built.pooled_excess == pytest.approx(manual.pooled_excess)
    assert built.loss_cost() == pytest.approx(manual.loss_cost())
    assert built.rate() == pytest.approx(manual.rate())


def test_from_experience_without_pooling_and_error_paths():
    exp = _exp()
    plain = rm.ExperienceRate.from_experience(exp)
    assert plain.pooled_excess == 0.0
    assert plain.incurred_claims == pytest.approx(float(exp.data["claims"].sum()))
    with pytest.raises(ValueError, match="claimant_col"):
        rm.ExperienceRate.from_experience(exp, pooling_point=100_000.0)
    no_exposure = Experience(_frame(), expense="claims", date="month")
    with pytest.raises(ValueError, match="exposure"):
        rm.ExperienceRate.from_experience(no_exposure)


def test_experience_rate_factory_by_segment():
    exp = _exp()
    book = rm.experience_rate(exp, by="segment", trend_annual=0.05)
    assert list(book["segment"]) == ["A", "B"]
    a = exp.filter(query="segment == 'A'")
    direct = rm.ExperienceRate.from_experience(a, trend_annual=0.05)
    row = book.loc[book["segment"] == "A"].iloc[0]
    assert row["rate"] == pytest.approx(direct.rate())
    assert row["exposure"] == pytest.approx(direct.exposure)
    single = rm.experience_rate(exp, trend_annual=0.05)
    assert isinstance(single, rm.ExperienceRate)


def test_exhibit_from_experience_annual_periods():
    exp = _exp()
    ex = rm.ExperienceExhibit.from_experience(
        exp, trend_factors=[1.045**2, 1.045], development_factors=1.02)
    sheet = ex.exhibit()
    per = (exp.data.set_index("month")[["premium", "claims"]]
           .resample("YE").sum())
    assert list(sheet.index) == [2025, 2026]
    assert sheet["earned_premium"].to_numpy() == pytest.approx(per["premium"].to_numpy())
    manual = rm.ExperienceExhibit(
        earned_premium=per["premium"].to_numpy(),
        losses=per["claims"].to_numpy(),
        trend_factors=[1.045**2, 1.045],
        development_factors=1.02,
        period_labels=[2025, 2026])
    assert sheet["loss_ratio"].to_numpy() == pytest.approx(
        manual.exhibit()["loss_ratio"].to_numpy())


def test_expense_selector_on_multi_expense_experience():
    from actuarialpy import Experience, Measures
    months = pd.date_range("2025-01-01", periods=3, freq="MS")
    membership = pd.DataFrame([{"member_id": m, "month": t, "member_months": 1.0}
                               for m in ("M1", "M2") for t in months])
    lines = pd.DataFrame([
        {"member_id": "M1", "incurred_date": months[0], "claim_type": ct, "paid": v}
        for ct, v in (("inpatient", 800.0), ("outpatient", 200.0))
    ])
    fees = pd.DataFrame([{"member_id": m, "month": t, "admin_fee": 10.0}
                         for m in ("M1", "M2") for t in months])
    exp = Experience.from_tables(
        membership, grain=["member_id", "month"], exposure="member_months",
        tables=[Measures(lines, expense="paid", wide_by="claim_type", date="incurred_date"),
                Measures(fees, expense="admin_fee")],
        date="month", period="M")
    claims_only = rm.ExperienceRate.from_experience(exp, expense=["inpatient", "outpatient"])
    assert claims_only.incurred_claims == pytest.approx(1_000.0)
    with pytest.raises(ValueError, match="bound expense roles"):
        rm.ExperienceRate.from_experience(exp, expense="dental")
    with pytest.raises(ValueError, match="expense"):
        rm.ExperienceRate.from_experience(exp)   # ambiguous: must select
