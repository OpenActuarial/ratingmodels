"""GLM diagnostics: residual identities, relativity CIs, deviance explained.

The residual tests are exact identities, not approximations: squared Pearson
residuals must sum to ``pearson_chi2_`` and squared deviance residuals to
``deviance_`` on the training data, because the fit computes those statistics
from the same per-row quantities.
"""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm
from ratingmodels.datasets import sample_rating_data


@pytest.fixture(scope="module")
def fitted():
    df = sample_rating_data(n=3000, seed=7)
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "industry"],
        exposure="exposure", base_levels={"area": "A"},
    )
    return df, model


def test_pearson_residuals_sum_to_chi2(fitted):
    df, model = fitted
    r = model.residuals(df, kind="pearson")
    assert isinstance(r, pd.Series)
    assert (r.index == df.index).all()
    assert float((r**2).sum()) == pytest.approx(model.pearson_chi2_, rel=1e-10)


def test_deviance_residuals_sum_to_deviance(fitted):
    df, model = fitted
    r = model.residuals(df, kind="deviance")
    assert float((r**2).sum()) == pytest.approx(model.deviance_, rel=1e-10)


def test_response_residuals_are_raw_gaps(fitted):
    df, model = fitted
    r = model.residuals(df, kind="response")
    mu = model.predict(df, exposure="exposure")
    np.testing.assert_allclose(r.to_numpy(), df["claims"].to_numpy() - mu)


def test_standardized_residuals_scale_up_pearson(fitted):
    # standardized = pearson / sqrt(dispersion * (1 - h)) with h in [0, 1),
    # so |standardized| >= |pearson| / sqrt(dispersion) elementwise, exactly.
    df, model = fitted
    pearson = model.residuals(df, kind="pearson").to_numpy()
    std = model.residuals(df, kind="standardized").to_numpy()
    assert np.all(np.isfinite(std))
    floor = np.abs(pearson) / np.sqrt(model.dispersion_)
    assert np.all(np.abs(std) >= floor - 1e-10)
    # and on correctly-specified simulated data they are roughly unit scale
    assert 0.7 < std.std() < 1.3


def test_residuals_unknown_kind_raises(fitted):
    df, model = fitted
    with pytest.raises(ValueError, match="unknown residual kind"):
        model.residuals(df, kind="studentized")


def test_residuals_work_on_new_data_with_new_levels(fitted):
    df, model = fitted
    new = df.head(50).copy()
    new.loc[new.index[0], "area"] = "Z"  # unseen level -> base
    r = model.residuals(new, kind="deviance")
    assert np.all(np.isfinite(r))
    assert len(r) == 50


def test_relativity_table_matches_closed_form(fitted):
    _, model = fitted
    tab = model.relativity_table(confidence_level=0.95)
    z = 1.959963984540054
    for (var, lvl), row in tab.iterrows():
        if row["is_base"]:
            assert row["relativity"] == 1.0
            assert np.isnan(row["ci_low"]) and np.isnan(row["ci_high"])
            continue
        name = f"{var}::{lvl}"
        coef = model.coefficients_[name]
        se = model.se_[name]
        assert row["relativity"] == pytest.approx(np.exp(coef), rel=1e-12)
        assert row["ci_low"] == pytest.approx(np.exp(coef - z * se), rel=1e-9)
        assert row["ci_high"] == pytest.approx(np.exp(coef + z * se), rel=1e-9)
        assert row["ci_low"] < row["relativity"] < row["ci_high"]


def test_relativity_table_covers_true_relativities():
    # the 95% CI should cover the simulation truth for (nearly) every level
    df = sample_rating_data(n=6000, seed=11)
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "industry", "tier"],
        exposure="exposure",
        base_levels={"area": "A", "industry": "retail", "tier": "bronze"},
    )
    from ratingmodels.datasets import TRUE_RELATIVITIES

    tab = model.relativity_table()
    hits = misses = 0
    for (var, lvl), row in tab.iterrows():
        if row["is_base"]:
            continue
        truth = TRUE_RELATIVITIES[var][lvl]
        if row["ci_low"] <= truth <= row["ci_high"]:
            hits += 1
        else:
            misses += 1
    assert hits >= 5  # 6 non-base levels; allow at most one unlucky miss
    assert misses <= 1


def test_relativity_table_includes_continuous_terms():
    df = sample_rating_data(n=2000, seed=3)
    rng = np.random.default_rng(0)
    df["age"] = rng.uniform(20, 60, len(df))
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area"],
        continuous=["age"], exposure="exposure",
    )
    tab = model.relativity_table()
    assert ("age", "(per +1)") in tab.index
    row = tab.loc[("age", "(per +1)")]
    assert row["relativity"] == pytest.approx(np.exp(model.coefficients_["age"]), rel=1e-12)


def test_deviance_explained_between_zero_and_one(fitted):
    _, model = fitted
    de = model.deviance_explained_
    assert 0.0 < de < 1.0
    assert de == pytest.approx(1 - model.deviance_ / model.null_deviance_)


def test_unfit_model_raises():
    model = rm.GLMRelativities()
    with pytest.raises(RuntimeError):
        model.relativity_table()
    with pytest.raises(RuntimeError):
        model.residuals(pd.DataFrame({"x": [1.0]}))


def test_to_factor_tables_reproduces_predict(fitted):
    # base_value_ x product of factor lookups x exposure == predict, exactly
    df, model = fitted
    tables = model.to_factor_tables()
    assert set(tables) == {"area", "industry"}
    combined = np.ones(len(df))
    for var, tab in tables.items():
        assert tab.lookup("__unseen__") == 1.0
        combined *= tab.apply(df[var]).to_numpy()
    manual = model.base_value_ * combined * df["exposure"].to_numpy()
    np.testing.assert_allclose(manual, model.predict(df, exposure="exposure"), rtol=1e-10)


def test_to_factor_tables_unfit_raises():
    with pytest.raises(RuntimeError):
        rm.GLMRelativities().to_factor_tables()
