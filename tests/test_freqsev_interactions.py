"""Interaction passthrough in FrequencySeverityModel."""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm


def _data(n=8000, seed=13):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "area": rng.choice(["A", "B"], n),
        "ind": rng.choice(["x", "y"], n),
        "exposure": rng.uniform(80.0, 120.0, n),
    })
    f_rel = (np.where(df["area"] == "B", 1.3, 1.0)
             * np.where((df["area"] == "B") & (df["ind"] == "y"), 1.25, 1.0))
    counts = rng.poisson(0.08 * df["exposure"] * f_rel)
    sev = 900.0 * np.where(df["ind"] == "y", 1.1, 1.0)
    df["claims"] = counts
    df["amount"] = counts * sev * rng.gamma(60.0, 1 / 60.0, n)
    return df


@pytest.fixture(scope="module")
def fitted():
    df = _data()
    model = rm.FrequencySeverityModel().fit(
        df, claim_count="claims", claim_amount="amount", exposure="exposure",
        frequency_predictors=["area", "ind"],
        base_levels={"area": "A", "ind": "x"},
        frequency_interactions=[("area", "ind")],
    )
    return df, model


def test_severity_inherits_frequency_interactions(fitted):
    _, model = fitted
    assert model.frequency._design_info_["interactions"] == [("area", "ind")]
    assert model.severity._design_info_["interactions"] == [("area", "ind")]


def test_pure_premium_identity_with_interactions(fitted):
    df, model = fitted
    pp = model.pure_premium_prediction(df, exposure="exposure")
    np.testing.assert_allclose(
        pp,
        model.frequency_prediction(df, exposure="exposure")
        * model.severity_prediction(df),
        rtol=1e-12,
    )


def test_combined_relativities_interaction_cell(fitted):
    _, model = fitted
    tab = model.combined_relativities()["area:ind"]
    assert isinstance(tab.index, pd.MultiIndex)
    cell = tab.loc[("B", "y")]
    assert cell["combined"] == pytest.approx(
        cell["frequency"] * cell["severity"], rel=1e-12)
    # the planted frequency synergy is recovered in the frequency column
    assert cell["frequency"] == pytest.approx(1.25, rel=0.1)


def test_frequency_only_interaction_fills_severity_with_one():
    df = _data(seed=14)
    model = rm.FrequencySeverityModel().fit(
        df, claim_count="claims", claim_amount="amount", exposure="exposure",
        frequency_predictors=["area", "ind"],
        frequency_interactions=[("area", "ind")],
        severity_interactions=[],
    )
    tab = model.combined_relativities()["area:ind"]
    np.testing.assert_allclose(tab["severity"], 1.0)
    np.testing.assert_allclose(tab["combined"], tab["frequency"], rtol=1e-12)


def test_to_factor_tables_excludes_interactions_and_plan_warns(fitted):
    _, model = fitted
    assert set(model.to_factor_tables()) == {"area", "ind"}
    with pytest.warns(UserWarning, match="interaction terms"):
        rm.RatingPlan.from_model(model)
