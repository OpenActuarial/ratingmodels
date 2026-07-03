"""Closed-form pricing algebra: inversions the scenario layer must satisfy."""
import pytest

from ratingmodels import PricingEvaluation, RetentionLoad, uplift_for_target_margin

RET = RetentionLoad(fixed_expense=22.0, variable_expense_ratio=0.09,
                    profit_margin=0.03, lae_ratio=0.05)
PE = PricingEvaluation(loss_cost=198.05, current_rate=255.0, retention=RET,
                       exposure=12_500.0, persistency=0.90)


@pytest.mark.parametrize("m", [0.0, 0.02, 0.05])
def test_premium_for_margin_inverts_through_at(m):
    premium = PE.premium_for_margin(m)
    outcome = PE.at(premium / 255.0 - 1.0)
    assert outcome.premium_rate == pytest.approx(premium, rel=1e-12)
    assert outcome.margin_rate / outcome.premium_rate == pytest.approx(m, abs=1e-12)


def test_zero_margin_rate_change_zeroes_the_margin():
    rc = PE.zero_margin_rate_change()
    assert PE.at(rc).margin_rate == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("m", [0.01, 0.03])
def test_rate_change_for_margin_consistency(m):
    assert PE.rate_change_for_margin(m) == pytest.approx(
        PE.premium_for_margin(m) / 255.0 - 1.0, rel=1e-12)


def test_infeasible_margin_raises():
    fat = RetentionLoad(variable_expense_ratio=0.5)
    pe = PricingEvaluation(loss_cost=100.0, current_rate=200.0, retention=fat)
    with pytest.raises(ValueError):
        pe.premium_for_margin(0.5)  # V + m == 1: no finite premium


def _cases():
    a = PricingEvaluation(loss_cost=198.05, current_rate=255.0, retention=RET,
                          exposure=12_500.0, persistency=0.90)
    b = PricingEvaluation(loss_cost=310.0, current_rate=380.0, retention=RET,
                          exposure=4_000.0, persistency=0.75)
    return {"a": a, "b": b}


@pytest.mark.parametrize("mode", ["multiplicative", "additive"])
def test_uplift_hits_the_aggregate_target_margin(mode):
    cases = _cases()
    base = {"a": 0.02, "b": -0.01}
    target = 0.05
    u = uplift_for_target_margin(cases, base, target, mode=mode)
    num = den = 0.0
    for key, pe in cases.items():
        rc = (1 + base[key]) * (1 + u) - 1 if mode == "multiplicative" else base[key] + u
        out = pe.at(rc)
        w = pe.exposure * pe.persistency
        num += w * out.margin_rate
        den += w * out.premium_rate
    assert num / den == pytest.approx(target, abs=1e-10)
