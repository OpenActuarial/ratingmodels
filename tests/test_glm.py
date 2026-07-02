"""Tests for GLMRelativities: closed-form correctness, convergence, inference.

The one-way Poisson test is exact: with a single categorical predictor and an
exposure offset, the GLM MLE relativities equal the observed rate ratios. This
is the test that catches the historical one-iteration convergence bug, which
produced coefficients one IRLS step from the crude start.
"""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm


def _poisson_frame(seed=4, n=4000):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "area": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
            "tier": rng.choice(["low", "mid", "high"], n),
            "age": rng.uniform(20.0, 60.0, n),
            "exposure": rng.uniform(0.5, 2.0, n),
        }
    )
    rel = {"A": 1.0, "B": 1.4, "C": 0.8, "low": 1.0, "mid": 1.2, "high": 1.6}
    mu = (
        0.12
        * df.exposure
        * np.array([rel[a] for a in df.area])
        * np.array([rel[t] for t in df.tier])
        * np.exp(0.01 * (df.age - 40.0))
    )
    df["claims"] = rng.poisson(mu)
    return df


def test_poisson_oneway_matches_rate_ratios_exactly():
    df = _poisson_frame()
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area"],
        exposure="exposure", base_levels={"area": "A"},
    )
    rates = df.groupby("area").claims.sum() / df.groupby("area").exposure.sum()
    assert model.base_value_ == pytest.approx(rates["A"], rel=1e-8)
    for lvl in ("B", "C"):
        assert model.relativities_["area"][lvl] == pytest.approx(
            rates[lvl] / rates["A"], rel=1e-8
        )


def test_irls_iterates_to_convergence():
    df = _poisson_frame()
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area"], exposure="exposure"
    )
    assert model.n_iter_ > 1        # one iteration means the historical bug
    assert model.converged_ is True


def test_intercept_only_with_exposure_is_overall_rate():
    df = _poisson_frame()
    model = rm.GLMRelativities(family="poisson").fit(
        df.assign(one="all"), response="claims", predictors=["one"], exposure="exposure"
    )
    assert model.base_value_ == pytest.approx(
        df.claims.sum() / df.exposure.sum(), rel=1e-8
    )


def test_gamma_weighted_mean_closed_form():
    rng = np.random.default_rng(7)
    n = 3000
    df = pd.DataFrame(
        {
            "area": rng.choice(["A", "B"], n),
            "w": rng.uniform(1.0, 3.0, n),
        }
    )
    df["sev"] = rng.gamma(2.0, np.where(df.area == "A", 500.0, 800.0))
    model = rm.GLMRelativities(family="gamma").fit(
        df, response="sev", predictors=["area"], weights="w",
        base_levels={"area": "A"},
    )
    wmean = df.groupby("area").apply(
        lambda g: np.average(g.sev, weights=g.w), include_groups=False
    )
    assert model.base_value_ == pytest.approx(wmean["A"], rel=1e-7)
    assert model.base_value_ * model.relativities_["area"]["B"] == pytest.approx(
        wmean["B"], rel=1e-7
    )


def test_continuous_covariate_slope_recovery():
    df = _poisson_frame(seed=8, n=12_000)
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "tier"],
        continuous=["age"], exposure="exposure",
    )
    assert model.coefficients_["age"] == pytest.approx(0.01, abs=0.004)


def test_predict_matches_fitted_and_handles_unknown_level():
    df = _poisson_frame()
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area"],
        exposure="exposure", base_levels={"area": "A"},
    )
    rates = df.groupby("area").claims.sum() / df.groupby("area").exposure.sum()
    pred = model.predict(df, exposure="exposure")
    expected = df.exposure.to_numpy() * rates.loc[df.area].to_numpy()
    assert np.allclose(pred, expected, rtol=1e-8)

    new = pd.DataFrame({"area": ["Z"], "exposure": [1.0]})  # unseen level -> base
    assert model.predict(new, exposure="exposure")[0] == pytest.approx(
        model.base_value_, rel=1e-10
    )


def test_summary_and_inference_fields():
    df = _poisson_frame()
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "tier"], exposure="exposure"
    )
    summ = model.summary()
    assert list(summ.columns) == ["coef", "se", "z", "relativity"]
    assert np.all(np.isfinite(summ["se"])) and np.all(summ["se"] > 0)
    assert np.isfinite(model.dispersion_) and model.dispersion_ > 0
    assert model.null_deviance_ > model.deviance_


def test_tweedie_converges_and_recovers_direction():
    rng = np.random.default_rng(9)
    n = 8000
    df = pd.DataFrame(
        {
            "area": rng.choice(["A", "B"], n),
            "exposure": rng.uniform(0.5, 2.0, n),
        }
    )
    rel = np.where(df.area == "B", 1.5, 1.0)
    freq = rng.poisson(0.4 * df.exposure * rel)
    sev = rng.gamma(2.0, 400.0, n)
    df["pp"] = freq * sev
    model = rm.GLMRelativities(family="tweedie", var_power=1.5).fit(
        df, response="pp", predictors=["area"],
        exposure="exposure", base_levels={"area": "A"},
    )
    assert model.converged_
    assert model.relativities_["area"]["B"] == pytest.approx(1.5, rel=0.15)
