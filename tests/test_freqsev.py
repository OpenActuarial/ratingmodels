"""FrequencySeverityModel: recovery, the multiplicative identity, edge cases."""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm
from ratingmodels.datasets import (
    TRUE_RELATIVITIES,
    TRUE_SEVERITY_RELATIVITIES,
    sample_frequency_severity_data,
)


@pytest.fixture(scope="module")
def fitted():
    df = sample_frequency_severity_data(n=4000, seed=0)
    model = rm.FrequencySeverityModel().fit(
        df,
        claim_count="claim_count",
        claim_amount="claim_amount",
        exposure="exposure",
        frequency_predictors=["area", "industry", "tier"],
        base_levels={"area": "A", "industry": "retail", "tier": "bronze"},
    )
    return df, model


def test_recovers_frequency_structure(fitted):
    _, model = fitted
    for var in ("area", "industry", "tier"):
        for lvl, truth in TRUE_RELATIVITIES[var].items():
            assert model.frequency.relativities_[var][lvl] == pytest.approx(truth, rel=0.05)


def test_recovers_severity_structure(fitted):
    _, model = fitted
    for var in ("industry", "tier"):
        for lvl, truth in TRUE_SEVERITY_RELATIVITIES[var].items():
            assert model.severity.relativities_[var][lvl] == pytest.approx(truth, rel=0.05)
    # area does not drive severity in the generator: relativities near 1
    for lvl in ("B", "C"):
        assert model.severity.relativities_["area"][lvl] == pytest.approx(1.0, abs=0.06)


def test_pure_premium_is_exact_product(fitted):
    df, model = fitted
    f = model.frequency_prediction(df, exposure="exposure")
    s = model.severity_prediction(df)
    np.testing.assert_array_equal(model.pure_premium_prediction(df, exposure="exposure"), f * s)
    # per-unit form: no exposure anywhere
    f1 = model.frequency_prediction(df)
    np.testing.assert_array_equal(model.pure_premium_prediction(df), f1 * s)


def test_combined_relativities_are_products(fitted):
    _, model = fitted
    combined = model.combined_relativities()
    for var, tab in combined.items():
        np.testing.assert_allclose(
            tab["combined"], tab["frequency"] * tab["severity"], rtol=1e-12
        )
        assert (tab["combined"] > 0).all()


def test_base_value_is_product(fitted):
    _, model = fitted
    assert model.base_value_ == pytest.approx(
        model.frequency.base_value_ * model.severity.base_value_, rel=1e-12
    )


def test_severity_can_use_fewer_predictors():
    df = sample_frequency_severity_data(n=2500, seed=4)
    model = rm.FrequencySeverityModel().fit(
        df, claim_count="claim_count", claim_amount="claim_amount",
        exposure="exposure",
        frequency_predictors=["area", "industry", "tier"],
        severity_predictors=["industry", "tier"],
    )
    combined = model.combined_relativities()
    assert "area" in combined  # frequency-only variable still present
    np.testing.assert_allclose(combined["area"]["severity"], 1.0)
    np.testing.assert_allclose(
        combined["area"]["combined"], combined["area"]["frequency"], rtol=1e-12
    )


def test_orphan_amounts_raise():
    df = pd.DataFrame(
        {
            "area": ["A", "B", "A", "B"],
            "exposure": [10.0, 10.0, 10.0, 10.0],
            "claim_count": [2.0, 0.0, 1.0, 0.0],
            "claim_amount": [500.0, 250.0, 100.0, 0.0],  # row 2: amount w/o claims
        }
    )
    with pytest.raises(ValueError, match="positive claim_amount"):
        rm.FrequencySeverityModel().fit(
            df, claim_count="claim_count", claim_amount="claim_amount",
            exposure="exposure", frequency_predictors=["area"],
        )


def test_zero_amount_claims_warn_and_are_excluded():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame(
        {
            "area": rng.choice(["A", "B"], n),
            "exposure": np.ones(n) * 10,
            "claim_count": rng.poisson(2.0, n).astype(float),
        }
    )
    df["claim_amount"] = df["claim_count"] * rng.gamma(2.0, 50.0, n)
    zero_rows = df.index[df["claim_count"] > 0][:5]
    df.loc[zero_rows, "claim_amount"] = 0.0
    with pytest.warns(UserWarning, match="zero amount"):
        model = rm.FrequencySeverityModel().fit(
            df, claim_count="claim_count", claim_amount="claim_amount",
            exposure="exposure", frequency_predictors=["area"],
        )
    assert model._fit_info_["n_severity_rows"] == int(
        ((df["claim_count"] > 0) & (df["claim_amount"] > 0)).sum()
    )


def test_summary_stacks_both_models(fitted):
    _, model = fitted
    s = model.summary()
    assert s.index.names == ["model", "term"]
    assert {"frequency", "severity"} == set(s.index.get_level_values("model"))


def test_unfit_raises():
    model = rm.FrequencySeverityModel()
    with pytest.raises(RuntimeError):
        model.pure_premium_prediction(pd.DataFrame({"x": [1]}))


def test_to_factor_tables_combined(fitted):
    _, model = fitted
    tables = model.to_factor_tables()
    combined = model.combined_relativities()
    for var, tab in tables.items():
        for lvl, val in combined[var]["combined"].items():
            assert tab.lookup(lvl) == pytest.approx(val, rel=1e-12)
        assert tab.lookup("__unseen__") == 1.0
