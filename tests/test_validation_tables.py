"""Calibration, actual/expected, and model-comparison tables."""
import numpy as np
import pytest

import ratingmodels as rm
from ratingmodels.datasets import sample_rating_data


def test_calibration_perfect_predictions_have_unit_ae():
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 100.0, 500)
    w = rng.uniform(0.5, 2.0, 500)
    tab = rm.calibration_table(y, y, exposure=w, n_bands=8)
    assert list(tab.index) == list(range(1, 9))
    np.testing.assert_allclose(tab["ae_ratio"].to_numpy(), 1.0, rtol=1e-12)


def test_calibration_bands_match_lift_bands():
    # same equal-exposure banding as lift_table: band composition, exposure,
    # and actual means agree exactly. (predicted_mean deliberately differs:
    # calibration treats predictions as totals, symmetric with actuals.)
    rng = np.random.default_rng(1)
    y = rng.poisson(3.0, 800).astype(float)
    p = y * rng.uniform(0.8, 1.2, 800)
    w = rng.uniform(1.0, 5.0, 800)
    cal = rm.calibration_table(y, p, exposure=w, n_bands=10)
    lift = rm.lift_table(y, p, exposure=w, n_bands=10)
    np.testing.assert_allclose(cal["n"], lift["n"])
    np.testing.assert_allclose(cal["exposure"], lift["exposure"], rtol=1e-12)
    np.testing.assert_allclose(cal["actual_mean"], lift["actual_mean"], rtol=1e-12)


def test_calibration_grouped_multiindex():
    rng = np.random.default_rng(2)
    y = rng.gamma(2.0, 50.0, 400)
    g = rng.choice(["u", "v"], 400)
    tab = rm.calibration_table(y, y * 1.1, n_bands=4, by=g)
    assert tab.index.names == ["group", "band"]
    assert set(tab.index.get_level_values("group")) == {"u", "v"}
    # each prediction is 10% high everywhere -> ae 1/1.1 in every band
    np.testing.assert_allclose(tab["ae_ratio"], 1 / 1.1, rtol=1e-12)


def test_actual_expected_overall_row():
    y = np.array([10.0, 20.0, 30.0])
    e = np.array([12.0, 18.0, 30.0])
    w = np.array([1.0, 2.0, 3.0])
    tab = rm.actual_expected_table(y, e, exposure=w)
    assert list(tab.index) == ["All"]
    row = tab.loc["All"]
    assert row["n"] == 3
    assert row["actual"] == 60.0 and row["expected"] == 60.0
    assert row["actual_mean"] == pytest.approx(10.0)
    assert row["ae_ratio"] == pytest.approx(1.0)


def test_actual_expected_by_level_matches_groupby():
    df = sample_rating_data(n=1000, seed=5)
    pred = df["claims"].to_numpy() * 1.05 + 0.1
    tab = rm.actual_expected_table(
        df["claims"], pred, exposure=df["exposure"], by=df["area"]
    )
    for lvl in ["A", "B", "C"]:
        m = (df["area"] == lvl).to_numpy()
        assert tab.loc[lvl, "n"] == int(m.sum())
        assert tab.loc[lvl, "actual"] == pytest.approx(df.loc[m, "claims"].sum())
        assert tab.loc[lvl, "expected"] == pytest.approx(pred[m].sum())
        assert tab.loc[lvl, "ae_ratio"] == pytest.approx(
            df.loc[m, "claims"].sum() / pred[m].sum()
        )
    assert tab.loc["All", "n"] == 1000


def test_actual_expected_multi_variable_tidy():
    df = sample_rating_data(n=500, seed=6)
    pred = df["claims"].to_numpy() + 0.5
    tab = rm.actual_expected_table(
        df["claims"], pred, exposure=df["exposure"],
        by={"area": df["area"], "tier": df["tier"]},
    )
    assert tab.index.names == ["variable", "level"]
    assert ("area", "A") in tab.index and ("tier", "gold") in tab.index
    assert ("All", "") in tab.index
    # block totals reconcile to the overall row
    area_rows = tab.loc["area"]
    assert area_rows["actual"].sum() == pytest.approx(tab.loc[("All", ""), "actual"])
    assert tab.loc[("All", ""), "n"] == 500


def test_actual_expected_without_total():
    y = np.array([1.0, 2.0])
    tab = rm.actual_expected_table(y, y, by=np.array(["a", "b"]), include_total=False)
    assert list(tab.index) == ["a", "b"]


def test_compare_models_ranks_true_model_first():
    df = sample_rating_data(n=4000, seed=9)
    full = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "industry", "tier"], exposure="exposure"
    )
    weak = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["tier"], exposure="exposure"
    )
    tab = rm.compare_models(
        {"full": full, "weak": weak}, df,
        response="claims", exposure="exposure",
    )
    assert list(tab.index) == ["full", "weak"]
    assert tab.index.name == "model"
    assert tab.loc["full", "deviance"] < tab.loc["weak", "deviance"]
    assert tab.loc["full", "gini"] > tab.loc["weak", "gini"]
    assert tab.loc["full", "deviance_explained"] > tab.loc["weak", "deviance_explained"]
    # null deviance depends only on the data, not the model
    assert tab["null_deviance"].nunique() == 1
    # canonical-link Poisson MLE balances totals in-sample: A/E == 1
    assert tab.loc["full", "ae_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_compare_models_sequence_autonames():
    df = sample_rating_data(n=800, seed=10)
    m = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area"], exposure="exposure"
    )
    tab = rm.compare_models([m], df, response="claims", exposure="exposure")
    assert list(tab.index) == ["model_1"]
    assert tab.loc["model_1", "n_params"] == len(m.coefficients_)
