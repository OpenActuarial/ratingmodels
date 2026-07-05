"""Interaction terms: recovery, table plumbing, and design consistency."""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm


def _interaction_data(n=6000, seed=0, synergy=1.3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "area": rng.choice(["A", "B"], n),
            "ind": rng.choice(["x", "y"], n),
            "exposure": rng.uniform(50.0, 150.0, n),
        }
    )
    rel = (
        np.where(df["area"] == "B", 1.4, 1.0)
        * np.where(df["ind"] == "y", 0.8, 1.0)
        * np.where((df["area"] == "B") & (df["ind"] == "y"), synergy, 1.0)
    )
    df["claims"] = rng.poisson(0.10 * df["exposure"] * rel)
    return df


@pytest.fixture(scope="module")
def fitted_ix():
    df = _interaction_data()
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "ind"], exposure="exposure",
        base_levels={"area": "A", "ind": "x"},
        interactions=[("area", "ind")],
    )
    return df, model


def test_recovers_mains_and_synergy(fitted_ix):
    _, model = fitted_ix
    assert model.relativities_["area"]["B"] == pytest.approx(1.4, rel=0.05)
    assert model.relativities_["ind"]["y"] == pytest.approx(0.8, rel=0.05)
    ix = model.relativities_["area:ind"]
    assert ix[("B", "y")] == pytest.approx(1.3, rel=0.08)
    assert ix.index.names == ["area", "ind"]


def test_interaction_beats_mains_only(fitted_ix):
    df, model = fitted_ix
    mains = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "ind"], exposure="exposure"
    )
    assert model.deviance_ < mains.deviance_
    assert len(model.coefficients_) == len(mains.coefficients_) + 1


def test_relativity_table_has_interaction_row(fitted_ix):
    _, model = fitted_ix
    tab = model.relativity_table()
    assert ("area:ind", "B | y") in tab.index
    row = tab.loc[("area:ind", "B | y")]
    assert row["ci_low"] < row["relativity"] < row["ci_high"]
    assert row["ci_low"] <= 1.3 <= row["ci_high"]


def test_to_factor_tables_excludes_interactions(fitted_ix):
    _, model = fitted_ix
    assert set(model.to_factor_tables()) == {"area", "ind"}


def test_predict_multiplies_all_three(fitted_ix):
    df, model = fitted_ix
    cell = pd.DataFrame({"area": ["A", "B"], "ind": ["x", "y"], "exposure": [1.0, 1.0]})
    mu = model.predict(cell, exposure="exposure")
    expected_by = (
        model.base_value_
        * model.relativities_["area"]["B"]
        * model.relativities_["ind"]["y"]
        * model.relativities_["area:ind"][("B", "y")]
    )
    assert mu[1] == pytest.approx(expected_by, rel=1e-10)
    assert mu[0] == pytest.approx(model.base_value_, rel=1e-10)


def test_residuals_on_frame_with_unseen_combo(fitted_ix):
    df, model = fitted_ix
    new = df.head(20).copy()
    new.loc[new.index[0], "area"] = "Z"  # unseen level -> base everywhere
    r = model.residuals(new, kind="deviance")
    assert np.all(np.isfinite(r)) and len(r) == 20


def test_cat_by_continuous_interaction():
    rng = np.random.default_rng(3)
    n = 6000
    df = pd.DataFrame(
        {
            "area": rng.choice(["A", "B"], n),
            "age": rng.uniform(-10.0, 10.0, n),
            "exposure": np.full(n, 100.0),
        }
    )
    slope = np.where(df["area"] == "B", 0.05, 0.02)
    df["claims"] = rng.poisson(0.10 * df["exposure"] * np.exp(slope * df["age"]))
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area"], continuous=["age"],
        exposure="exposure", base_levels={"area": "A"},
        interactions=[("age", "area")],  # order-agnostic
    )
    assert model.coefficients_["age"] == pytest.approx(0.02, abs=0.005)
    assert model.coefficients_["area::B:age"] == pytest.approx(0.03, abs=0.007)
    assert ("area:age", "B (per +1)") in model.relativity_table().index


def test_invalid_interaction_raises():
    df = _interaction_data(n=200)
    with pytest.raises(ValueError, match="interaction"):
        rm.GLMRelativities(family="poisson").fit(
            df, response="claims", predictors=["area"], exposure="exposure",
            interactions=[("area", "missing_col")],
        )


def test_predict_interval_brackets_and_scales(fitted_ix):
    df, model = fitted_ix
    pi = model.predict_interval(df.head(50), exposure="exposure")
    assert list(pi.columns) == ["predicted", "ci_low", "ci_high"]
    assert (pi["ci_low"] < pi["predicted"]).all()
    assert (pi["predicted"] < pi["ci_high"]).all()
    np.testing.assert_allclose(
        pi["predicted"], model.predict(df.head(50), exposure="exposure"), rtol=1e-12
    )
    per_unit = model.predict_interval(df.head(50))
    np.testing.assert_allclose(
        pi["ci_high"], per_unit["ci_high"] * df.head(50)["exposure"], rtol=1e-12
    )


def test_two_interactions_simultaneously():
    rng = np.random.default_rng(5)
    n = 5000
    df = pd.DataFrame({
        "a": rng.choice(["p", "q"], n),
        "b": rng.choice(["u", "v"], n),
        "c": rng.choice(["x", "y"], n),
        "exposure": np.full(n, 100.0),
    })
    df["claims"] = rng.poisson(0.1 * df["exposure"])
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["a", "b", "c"], exposure="exposure",
        interactions=[("a", "b"), ("a", "c")],
    )
    spec_kinds = [t[0] for t in model._design_info_["spec"]]
    assert spec_kinds.count("ixcc") == 2  # one non-base pair per interaction
    assert {"a:b", "a:c"} <= set(model.relativities_)


def test_unobserved_pair_is_skipped_not_aliased():
    df = pd.DataFrame({
        "a": ["p", "p", "q", "q", "p", "q"] * 200,
        "b": ["u", "v", "u", "u", "u", "u"] * 200,   # (q, v) never occurs
        "exposure": [100.0] * 1200,
    })
    df["claims"] = np.random.default_rng(6).poisson(10.0, 1200)
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["a", "b"], exposure="exposure",
        base_levels={"a": "p", "b": "u"},
        interactions=[("a", "b")],
    )
    ixcc = [t for t in model._design_info_["spec"] if t[0] == "ixcc"]
    assert ixcc == []  # the only candidate cell (q, v) is unobserved
    # predicting the unseen combo falls back to the product of mains
    cell = pd.DataFrame({"a": ["q"], "b": ["v"], "exposure": [1.0]})
    expected = (model.base_value_ * model.relativities_["a"]["q"]
                * model.relativities_["b"]["v"])
    assert model.predict(cell, exposure="exposure")[0] == pytest.approx(
        expected, rel=1e-10)


def test_predict_interval_offset_and_guards(fitted_ix):
    df, model = fitted_ix
    head = df.head(30).copy()
    head["off"] = 0.3
    base = model.predict_interval(head)
    shifted = model.predict_interval(head, offset="off")
    for col in ("predicted", "ci_low", "ci_high"):
        np.testing.assert_allclose(shifted[col], base[col] * np.exp(0.3),
                                   rtol=1e-12)
    with pytest.raises(ValueError, match="confidence_level"):
        model.predict_interval(head, confidence_level=1.2)
