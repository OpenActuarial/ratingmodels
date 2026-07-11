"""Regression: Example 10's solve numbers stay true.

Pins the ratingmodels segment of docs page ``worked-example-contract.md``
(Example 10): the contract-pinned constructors turn the page's contract-year
loss costs and dollar-expense levels (produced and pinned at full precision
in projectionmodels' suite) into the charged rates and renewal actions —
and the gross pin's action is identically the claims trend.
"""

import pytest

from ratingmodels import RateIndication, RetentionLoad

# hand-offs from projectionmodels' suite (page values, full precision there)
C1, C2 = 497.348761, 531.545174        # contract-year claims per member-month
F1, F2 = 72.907305, 75.886646          # premium-independent expense PMPM
LR = 0.85
COMMISSION = {"commission": 0.03}


def _action(contract_cy1, contract_cy2):
    rate1 = contract_cy1.gross_rate(C1)
    change = RateIndication(
        experience_loss_cost=C2, manual_loss_cost=C2, credibility=1.0,
        current_rate=rate1, retention=contract_cy2,
    ).indicated_rate_change()
    return rate1, change


def test_gross_pin_rate_and_action():
    contract = RetentionLoad.from_gross_loss_ratio(LR, variable_items=COMMISSION)
    rate1, action = _action(contract, contract)
    assert round(float(rate1), 4) == 585.1162
    assert round(float(action), 6) == 0.068757
    # the constant-gross-LR identity: the action IS the claims trend
    assert float(action) == pytest.approx(C2 / C1 - 1, abs=1e-12)


def test_net_pin_rate_and_action():
    cy1 = RetentionLoad.from_net_loss_ratio(LR, fixed_expense=F1,
                                            variable_items=COMMISSION)
    cy2 = RetentionLoad.from_net_loss_ratio(LR, fixed_expense=F2,
                                            variable_items=COMMISSION)
    rate1, action = _action(cy1, cy2)
    assert round(float(rate1), 4) == 678.3747
    assert round(float(action), 6) == 0.065667
    # closed form and contract check
    assert float(rate1) == pytest.approx((C1 / LR + F1) / (1 - 0.03), abs=1e-9)
    assert float(cy1.implied_net_loss_ratio(C1)) == pytest.approx(LR, abs=1e-12)
    # expense trend (F slower than C) dilutes the action below claims trend
    assert float(action) < C2 / C1 - 1
