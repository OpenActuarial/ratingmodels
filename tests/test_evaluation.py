"""Tests for pricing-model evaluation: ordered-Lorenz Gini and lift tables."""
import numpy as np
import pandas as pd
import pytest

from ratingmodels import gini_coefficient, lift_table


def _book(seed=30, n=20_000):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, n)
    risk = rng.lognormal(-2.0, 0.6, n)          # true underlying rate
    actual = rng.poisson(risk * exposure)
    return actual.astype(float), risk, exposure


def test_gini_perfect_prediction_is_one():
    actual, _, exposure = _book()
    assert gini_coefficient(actual, actual, exposure) == pytest.approx(1.0)


def test_gini_constant_prediction_is_zero():
    actual, _, exposure = _book()
    g = gini_coefficient(actual, np.ones_like(actual), exposure)
    assert abs(g) < 0.05


def test_gini_informative_model_between_zero_and_one():
    actual, risk, exposure = _book()
    g = gini_coefficient(actual, risk, exposure)
    assert 0.10 < g < 1.0


def test_gini_normalization_relationship():
    actual, risk, exposure = _book()
    raw = gini_coefficient(actual, risk, exposure, normalize=False)
    norm = gini_coefficient(actual, risk, exposure, normalize=True)
    perfect_raw = gini_coefficient(actual, actual, exposure, normalize=False)
    assert norm == pytest.approx(raw / perfect_raw, rel=1e-12)


def test_gini_input_validation():
    with pytest.raises(ValueError):
        gini_coefficient([1.0, 2.0], [1.0], None)
    with pytest.raises(ValueError):
        gini_coefficient([1.0], [1.0], [-1.0])


def test_lift_table_structure_and_totals():
    actual, risk, exposure = _book()
    table = lift_table(actual, risk, exposure, n_bands=10)
    assert list(table.columns) == ["n", "exposure", "predicted_mean", "actual_mean", "lift"]
    assert len(table) == 10
    assert table["exposure"].sum() == pytest.approx(exposure.sum(), rel=1e-9)
    assert table["n"].sum() == actual.size
    # exposure-weighted actual means recombine to the overall mean (lift 1.0)
    overall = float(np.sum(table["actual_mean"] * table["exposure"]) / table["exposure"].sum())
    assert overall == pytest.approx(actual.sum() / exposure.sum(), rel=1e-9)


def test_lift_table_orders_risk():
    actual, risk, exposure = _book()
    table = lift_table(actual, risk, exposure, n_bands=10)
    assert table["predicted_mean"].is_monotonic_increasing
    assert table["actual_mean"].iloc[-1] > table["actual_mean"].iloc[0]
    assert table["lift"].iloc[-1] > 1.0 > table["lift"].iloc[0]


def test_lift_table_unweighted_and_validation():
    actual, risk, _ = _book(n=5_000)
    table = lift_table(actual, risk, n_bands=5)
    assert len(table) == 5
    assert isinstance(table, pd.DataFrame)
    with pytest.raises(ValueError):
        lift_table(actual, risk, n_bands=1)
