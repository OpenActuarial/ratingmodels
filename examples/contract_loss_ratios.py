"""Contract-pinned loss ratios: solving premium when the contract fixes the ratio.

Two standard pins, both parameterizations of the fundamental insurance equation:
gross  --  claims / premium = LR*                =>  P = C / LR*
net    --  claims / (premium - expenses) = LR*   =>  P = (C / LR* + F) / (1 - V)
"""

from __future__ import annotations

import pandas as pd

import ratingmodels as rm


def run_example() -> dict[str, object]:
    loss_cost = 450.0  # projected claims per member-month

    # --- gross pin: the contract fixes claims / premium at 0.85 ------------
    gross = rm.RetentionLoad.from_gross_loss_ratio(
        0.85, variable_items={"commission": 0.03, "premium_tax": 0.023}
    )
    gross_rate = gross.gross_rate(loss_cost)
    print("=== Gross-pinned contract (LR* = 0.85) ===")
    print(f"charged rate            : {gross_rate:.4f}")
    print(f"implied loss ratio      : {gross.implied_loss_ratio(loss_cost):.4f}")
    print(f"retention split         : variable {gross.variable_expense_ratio:.3f}, "
          f"profit remainder {gross.profit_margin:.3f}")

    # --- net pin: the contract fixes claims / (premium - expenses) ---------
    net = rm.RetentionLoad.from_net_loss_ratio(
        0.87, fixed_expense=25.0, variable_items={"commission": 0.03}
    )
    net_rate = net.gross_rate(loss_cost)
    expenses = 25.0 + 0.03 * net_rate
    print("\n=== Net-pinned contract (LR* = 0.87, $25 flat fee, 3% commission) ===")
    print(f"charged rate            : {net_rate:.4f}")
    print(f"contract check C/(P-E)  : {net.implied_net_loss_ratio(loss_cost):.6f}")
    print(f"margin (claims-prop.)   : {net_rate - expenses - loss_cost:.4f} "
          f"= C(1-LR)/LR = {loss_cost * (1 - 0.87) / 0.87:.4f}")

    # --- a book of pinned ratios, in columns --------------------------------
    book = pd.DataFrame(
        {"loss_cost": [400.0, 450.0, 500.0], "contract_lr": [0.80, 0.85, 0.90]},
        index=["g1", "g2", "g3"],
    )
    book["charged_rate"] = rm.RetentionLoad.from_gross_loss_ratio(
        book["contract_lr"]
    ).gross_rate(book["loss_cost"])
    print("\n=== Book of pinned-ratio groups ===")
    print(book.to_string())

    return {"gross": gross, "net": net, "book": book}


if __name__ == "__main__":
    run_example()
