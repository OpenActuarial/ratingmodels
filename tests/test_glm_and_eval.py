"""GLM relativities against a hand-solvable case, and the evaluation metrics."""
import numpy as np
import pandas as pd
import pytest

from ratingmodels import GLMRelativities, gini_coefficient, lift_table


def _one_factor_data():
    # area A: 100 exposures, 20 claims (rate 0.20); area B: 50 exposures, 15 (0.30)
    return pd.DataFrame({
        "area": ["A", "A", "B", "B"],
        "exposure": [60.0, 40.0, 30.0, 20.0],
        "claims": [12.0, 8.0, 9.0, 6.0],
    })


def test_saturated_poisson_glm_recovers_the_rate_ratio():
    g = GLMRelativities(family="poisson").fit(
        _one_factor_data(), response="claims", predictors=["area"],
        exposure="exposure", base_levels={"area": "A"},
    )
    assert g.converged_
    assert g.relativities_["area"]["A"] == pytest.approx(1.0, rel=1e-9)
    assert g.relativities_["area"]["B"] == pytest.approx(1.5, rel=1e-6)
    assert g.base_value_ == pytest.approx(0.20, rel=1e-6)


def test_saturated_glm_prediction_matches_observed_counts():
    df = _one_factor_data()
    g = GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area"], exposure="exposure",
        base_levels={"area": "A"},
    )
    assert np.allclose(g.predict(df) * df["exposure"], df["claims"], rtol=1e-6)


def test_gini_endpoints_and_permutation_invariance():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert gini_coefficient(actual, actual) == pytest.approx(1.0, abs=1e-9)
    assert gini_coefficient(actual, -actual) == pytest.approx(-1.0, abs=1e-9)
    rng = np.random.default_rng(1)
    predicted = actual + rng.normal(0.0, 0.1, actual.size)  # no ties
    perm = rng.permutation(actual.size)
    assert gini_coefficient(actual, predicted) == pytest.approx(
        gini_coefficient(actual[perm], predicted[perm]), rel=1e-12)


def test_lift_table_structure_and_monotone_bands():
    rng = np.random.default_rng(0)
    predicted = np.linspace(1.0, 10.0, 200)
    actual = predicted + rng.normal(0.0, 1.0, 200)
    lt = lift_table(actual, predicted, n_bands=5)
    assert len(lt) == 5
    assert lt["n"].sum() == 200
    assert (lt["predicted_mean"].diff().dropna() >= 0).all()
