"""Adapter contract: GLMRelativities against an independent statsmodels fit.

statsmodels is the fitting engine, so these are no longer cross-checks of a
rival solver -- they are contract tests for the *adapter*. Each test builds
the statsmodels model independently (its own family objects, its own offset
construction from exposure, its own weights) and asserts the wrapper's
attributes, conventions, and in-package evaluation math agree:

* marshaling -- exposure -> log offset, ``weights`` -> ``var_weights``,
  ``scale="X2"`` (Pearson dispersion) for every family, named parameters
  aligned to the design columns;
* in-package math that must stay independent because it runs on arbitrary
  frames -- ``residuals`` (vs ``resid_pearson``/``resid_deviance``),
  ``relativity_table`` intervals (vs ``conf_int``), and the family deviance
  used by ``compare_models``;
* wiring -- ``null_deviance_``, ``dispersion_``, ``pearson_chi2_``,
  predictions.

Design alignment: the independent fit consumes the exact design matrix our
model built (``_design_matrix_from_info``), so parameters correspond
column-for-column and any disagreement is in the marshaling, not the coding.
"""
import numpy as np
import pandas as pd

import ratingmodels as rm
from ratingmodels.datasets import sample_frequency_severity_data, sample_rating_data

import statsmodels.api as sm


def _sm_design(model, df):
    """The exact design matrix the ratingmodels fit used, as a DataFrame."""
    X = model._design_matrix_from_info(df)
    return pd.DataFrame(X, columns=model._design_info_["columns"], index=df.index)


def _assert_matches(model, res, rtol_coef=1e-6, rtol_se=1e-5):
    ours = model.coefficients_
    theirs = pd.Series(res.params.to_numpy(), index=ours.index)
    np.testing.assert_allclose(ours, theirs, rtol=rtol_coef, atol=1e-10)
    np.testing.assert_allclose(
        model.se_, pd.Series(res.bse.to_numpy(), index=ours.index),
        rtol=rtol_se, atol=1e-12,
    )
    np.testing.assert_allclose(model.deviance_, res.deviance, rtol=1e-8)


def test_poisson_with_exposure_matches_statsmodels():
    df = sample_rating_data(n=3000, seed=21)
    rng = np.random.default_rng(0)
    df["age"] = rng.uniform(20.0, 60.0, len(df))

    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "industry", "tier"],
        continuous=["age"], exposure="exposure",
    )
    res = sm.GLM(
        df["claims"], _sm_design(model, df),
        family=sm.families.Poisson(sm.families.links.Log()),
        offset=np.log(df["exposure"].to_numpy()),
    ).fit(scale="X2")  # quasi-Poisson: Pearson dispersion, as our SEs use

    _assert_matches(model, res)
    np.testing.assert_allclose(model.dispersion_, res.scale, rtol=1e-8)
    np.testing.assert_allclose(model.pearson_chi2_, res.pearson_chi2, rtol=1e-8)
    np.testing.assert_allclose(model.null_deviance_, res.null_deviance, rtol=1e-7)
    np.testing.assert_allclose(
        model.predict(df, exposure="exposure"), res.fittedvalues, rtol=1e-6
    )
    # relativity CIs: exp of the coefficient interval at matching dispersion
    tab = model.relativity_table(confidence_level=0.95)
    ci = np.exp(res.conf_int(alpha=0.05))
    for (var, lvl), row in tab.iterrows():
        if row["is_base"] or lvl == "(per +1)":
            continue
        name = f"{var}::{lvl}"
        np.testing.assert_allclose(row["ci_low"], ci.loc[name, 0], rtol=1e-5)
        np.testing.assert_allclose(row["ci_high"], ci.loc[name, 1], rtol=1e-5)


def test_gamma_with_variance_weights_matches_statsmodels():
    # severity-style fit: positive response, claim-count variance weights
    df = sample_frequency_severity_data(n=3000, seed=22)
    df = df[(df["claim_count"] > 0) & (df["claim_amount"] > 0)].copy()
    df["severity"] = df["claim_amount"] / df["claim_count"]

    model = rm.GLMRelativities(family="gamma").fit(
        df, response="severity", predictors=["industry", "tier"],
        weights="claim_count",
    )
    res = sm.GLM(
        df["severity"], _sm_design(model, df),
        family=sm.families.Gamma(sm.families.links.Log()),
        var_weights=df["claim_count"].to_numpy(),
    ).fit()  # Gamma default scale is Pearson chi2 / dof, matching ours

    _assert_matches(model, res)
    np.testing.assert_allclose(model.dispersion_, res.scale, rtol=1e-8)


def test_gamma_null_deviance_with_offset_matches_statsmodels():
    # null_deviance_ must be the *fitted* intercept-only model. With a
    # non-Poisson family and varying offsets a naive weighted-mean rate is
    # not that model (the 0.5.x behavior); verify against an independently
    # constructed fit.
    rng = np.random.default_rng(23)
    n = 800
    off = rng.uniform(0.0, 2.0, n)
    y = rng.gamma(2.0, 50.0, n) * np.exp(off)
    df = pd.DataFrame({"y": y, "off": off, "grp": rng.choice(["a", "b"], n)})

    model = rm.GLMRelativities(family="gamma").fit(
        df, response="y", predictors=["grp"], offset="off"
    )
    res = sm.GLM(
        df["y"], _sm_design(model, df),
        family=sm.families.Gamma(sm.families.links.Log()),
        offset=off,
    ).fit()
    np.testing.assert_allclose(model.null_deviance_, res.null_deviance, rtol=1e-7)
    np.testing.assert_allclose(model.deviance_, res.deviance, rtol=1e-8)


def test_tweedie_matches_statsmodels():
    df = sample_frequency_severity_data(n=3000, seed=24)
    model = rm.GLMRelativities(family="tweedie", var_power=1.5).fit(
        df, response="claim_amount", predictors=["area", "industry"],
        exposure="exposure",
    )
    res = sm.GLM(
        df["claim_amount"], _sm_design(model, df),
        family=sm.families.Tweedie(
            link=sm.families.links.Log(), var_power=1.5, eql=True
        ),
        offset=np.log(df["exposure"].to_numpy()),
    ).fit()  # Tweedie default scale is Pearson-based, matching ours

    _assert_matches(model, res)
    np.testing.assert_allclose(model.dispersion_, res.scale, rtol=1e-8)
    np.testing.assert_allclose(model.null_deviance_, res.null_deviance, rtol=1e-7)


def test_pearson_residuals_match_statsmodels():
    df = sample_rating_data(n=1500, seed=25)
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "tier"], exposure="exposure"
    )
    res = sm.GLM(
        df["claims"], _sm_design(model, df),
        family=sm.families.Poisson(sm.families.links.Log()),
        offset=np.log(df["exposure"].to_numpy()),
    ).fit()
    # residuals amplify fitting tolerance: r = (y-mu)/sqrt(mu), so a ~1e-6
    # relative wiggle in mu is a large *relative* change wherever r is near
    # zero -- compare with an absolute floor rather than tightening rtol
    np.testing.assert_allclose(
        model.residuals(df, kind="pearson"), res.resid_pearson,
        rtol=1e-4, atol=1e-8,
    )
    np.testing.assert_allclose(
        model.residuals(df, kind="deviance"), res.resid_deviance,
        rtol=1e-4, atol=1e-8,
    )
