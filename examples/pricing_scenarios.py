"""Scenario pricing in one pass: build up a rate, evaluate changes, hit a
portfolio margin target, and read the renewal impact.

Run:  python examples/pricing_scenarios.py
"""
import pandas as pd

import ratingmodels as rm

# --- build up a net rate from a base loss cost --------------------------------
result = (rm.BuildUp()
          .start("base loss cost", 483.21)
          .multiply("trend (2 yrs @ 4%)", 1.04**2)
          .add("pooling charge", 18.50)
          .checkpoint("net claim cost")
          .multiply("lae", 1.05)
          .evaluate())
print(result.to_frame().to_string(index=False))
print(f"net claim cost subtotal : {result.subtotal('net claim cost'):8.2f}")
print(f"final per-unit value    : {result.value:8.2f}\n")

# --- evaluate rate changes against a margin target ----------------------------
retention = rm.RetentionLoad(fixed_expense=22.0, variable_expense_ratio=0.09,
                             profit_margin=0.03, lae_ratio=0.05)
book = {
    "renewal_a": rm.PricingEvaluation(loss_cost=198.05, current_rate=255.0,
                                      retention=retention, exposure=12_500.0,
                                      persistency=0.90),
    "renewal_b": rm.PricingEvaluation(loss_cost=310.00, current_rate=380.0,
                                      retention=retention, exposure=4_000.0,
                                      persistency=0.75),
}
for name, pe in book.items():
    out = pe.at(0.05)
    print(f"{name}: +5.0% -> premium {out.premium_rate:7.2f}, "
          f"margin {out.margin_rate / out.premium_rate:6.2%}")

uplift = rm.uplift_for_target_margin(book, base_changes={"renewal_a": 0.02, "renewal_b": -0.01},
                                     target_margin=0.05)
print(f"\nuniform uplift for a 5% portfolio margin: {uplift:+.4%}\n")

# --- renewal impact at the unit level -----------------------------------------
census = pd.DataFrame({"age_f": [1.10, 0.90, 1.00], "area_f": [1.00, 1.20, 0.95],
                       "count": [3, 5, 2]})
print(rm.unit_level_renewal(census, base_rate=100.0,
                            factor_cols=["age_f", "area_f"]).to_string(index=False))
