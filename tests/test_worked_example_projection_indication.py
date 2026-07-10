"""Regression: Example 7's indication and renewal-constraint numbers stay true.

Pins the ratingmodels segment of docs page ``worked-example-projection.md``
(Example 7): the CY2027 loss costs feed a ``RateIndication`` and ``renew``
applies the 10% corridor. The loss costs themselves are produced and pinned
at full precision in projectionmodels' suite; they cross over here as the
page's rounded values.
"""
import pandas as pd

import ratingmodels as rm


def test_projection_page_indication_numbers():
    lc = pd.Series({"A": 590.3863, "B": 604.9014})   # CY2027 claims per exposure
    current = pd.Series({"A": 585.0, "B": 612.0})
    retention = rm.RetentionLoad(fixed_expense=24.0,
                                 variable_expense_ratio=0.030,
                                 profit_margin=0.02)
    indication = rm.RateIndication(
        experience_loss_cost=lc * 1.012, manual_loss_cost=lc * 1.012,
        credibility=1.0, current_rate=current, retention=retention)
    change = indication.indicated_rate_change()
    assert round(float(change["A"]), 4) == 0.1183
    assert round(float(change["B"]), 4) == 0.0942

    action = rm.renew(current, indication.indicated_rate(), cap=0.10, floor=0.0)
    frame = action.to_frame()
    assert bool(frame.loc["A", "capped"]) and not bool(frame.loc["B", "capped"])
    assert round(float(frame.loc["A", "proposed_rate"]), 2) == 643.50
    assert round(float(frame.loc["B", "proposed_rate"]), 2) == 669.64
    # the issued action carried into projectionmodels' companion test
    assert round(float(action.proposed_change["B"]), 6) == 0.094183
