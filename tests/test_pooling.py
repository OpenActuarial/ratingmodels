"""Pooling charge from the duck-typed severity tail protocol."""
import math

import numpy as np
import pytest

import ratingmodels as rm


class StubExponential:
    """Closed-form exponential: memoryless, so every identity is exact."""

    def __init__(self, mean):
        self.mu = float(mean)

    def sf(self, x):
        return math.exp(-x / self.mu)

    def mean_excess(self, d):
        return self.mu  # memoryless


def test_build_up_exact_for_exponential():
    sev = StubExponential(mean=40_000.0)
    out = rm.pooling_charge_from_severity(
        sev, pooling_point=100_000.0, expected_frequency=0.8,
        expense_ratio=0.10, risk_margin=0.05,
    )
    surv = math.exp(-100_000.0 / 40_000.0)
    assert out["exceedance_probability"] == pytest.approx(surv, rel=1e-12)
    assert out["mean_excess"] == pytest.approx(40_000.0)
    assert out["expected_excess_per_claim"] == pytest.approx(surv * 40_000.0, rel=1e-12)
    assert out["pure_excess_cost"] == pytest.approx(0.8 * surv * 40_000.0, rel=1e-12)
    assert out["pooling_charge"] == pytest.approx(
        0.8 * surv * 40_000.0 * 1.05 / 0.90, rel=1e-12
    )
    # no loadings -> charge equals the pure cost
    plain = rm.pooling_charge_from_severity(sev, 100_000.0, 0.8)
    assert plain["pooling_charge"] == pytest.approx(plain["pure_excess_cost"])


def test_zero_survival_gives_zero_charge():
    class Capped:
        def sf(self, x):
            return 0.0

        def mean_excess(self, d):  # pragma: no cover - must not be called
            raise AssertionError("mean_excess should not be evaluated at S(d)=0")

    out = rm.pooling_charge_from_severity(Capped(), 1e6, 1.0)
    assert out["pooling_charge"] == 0.0 and out["mean_excess"] == 0.0


def test_protocol_and_validation_errors():
    with pytest.raises(TypeError, match="sf"):
        rm.pooling_charge_from_severity(object(), 100.0, 1.0)

    class InfiniteTail(StubExponential):
        def mean_excess(self, d):
            return float("inf")

    with pytest.raises(ValueError, match="no finite pooling charge"):
        rm.pooling_charge_from_severity(InfiniteTail(1.0), 100.0, 1.0)
    with pytest.raises(ValueError, match="expense_ratio"):
        rm.pooling_charge_from_severity(StubExponential(1.0), 100.0, 1.0,
                                        expense_ratio=1.0)


def test_feeds_experience_rate():
    sev = StubExponential(mean=50_000.0)
    charge = rm.pooling_charge_from_severity(sev, 200_000.0, 0.5)["pooling_charge"]
    assert np.isfinite(charge) and charge > 0


def test_seam_lossmodels_distribution_and_layer():
    lm = pytest.importorskip("lossmodels")
    exp = lm.Exponential(rate=1 / 40_000.0)
    out = rm.pooling_charge_from_severity(exp, 100_000.0, 0.8)
    # closed form through the real package machinery: freq * e^{-d/mu} * mu
    assert out["pooling_charge"] == pytest.approx(
        0.8 * np.exp(-2.5) * 40_000.0, rel=1e-7)
    # a layer exhausted below the pooling point has nothing to pool
    from lossmodels.coverage import Layer

    capped = Layer(exp, d=0.0, u=50_000.0)
    zero = rm.pooling_charge_from_severity(capped, 60_000.0, 0.8)
    assert zero["pooling_charge"] == 0.0


def test_seam_extremeloss_gpd_closed_form():
    el = pytest.importorskip("extremeloss")
    fit = el.GPDFit(threshold=100_000.0, xi=0.25, beta=30_000.0,
                    exceedance_fraction=0.04, n_exceedances=400)
    d = 150_000.0
    out = rm.pooling_charge_from_severity(fit, d, expected_frequency=0.6)
    surv = 0.04 * (1 + 0.25 * (d - 100_000.0) / 30_000.0) ** (-1 / 0.25)
    me = (30_000.0 + 0.25 * (d - 100_000.0)) / (1 - 0.25)
    assert out["exceedance_probability"] == pytest.approx(surv, rel=1e-9)
    assert out["mean_excess"] == pytest.approx(me, rel=1e-12)
    assert out["pooling_charge"] == pytest.approx(0.6 * surv * me, rel=1e-9)


def test_seam_cross_validated_against_layer_mean():
    """freq x S(d) x e(d) must equal freq x E[(X-d)+] -- and lossmodels'
    Layer(d, inf).mean() computes E[(X-d)+] through entirely independent
    machinery. Two code paths, one number."""
    lm = pytest.importorskip("lossmodels")
    from lossmodels.coverage import Layer

    sev = lm.Lognormal(mu=9.0, sigma=1.1)
    d, freq = 120_000.0, 0.7
    out = rm.pooling_charge_from_severity(sev, d, freq)
    independent = Layer(sev, d=d, u=np.inf).mean()
    assert out["pure_excess_cost"] == pytest.approx(freq * independent,
                                                    rel=1e-6)


def test_seam_empirical_severity_is_exact_sample_arithmetic():
    """EmpiricalSeverity inherits the protocol, so the seam prices a layer
    straight from data with no parametric model -- and the charge must
    equal the plain sample mean of (x - d)+ times frequency, exactly."""
    pytest.importorskip("lossmodels")
    from lossmodels.empirical import EmpiricalSeverity

    rng = np.random.default_rng(4)
    x = rng.lognormal(9.0, 1.2, 3_000)
    emp = EmpiricalSeverity(x)
    d, freq = 80_000.0, 0.5
    out = rm.pooling_charge_from_severity(emp, d, freq)
    assert out["pure_excess_cost"] == pytest.approx(
        freq * np.mean(np.maximum(x - d, 0.0)), rel=1e-9)
