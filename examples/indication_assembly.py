"""From raw experience periods to an indicated rate change.

    rate-change history -> on_level_factors        (premium restatement)
    claims triangle     -> ChainLadder (actuarialpy) -> development factors
    both + trend        -> ExperienceExhibit        (the worksheet)
    -> to_indication -> RateIndication -> indicated change

Every adjustment is a visible column; the gross-up is the same
RetentionLoad algebra as everywhere else in the package.

Run with:  python examples/indication_assembly.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from actuarialpy.reserving import ChainLadder

import ratingmodels as rm


def main() -> None:
    # ----- premium side: restate two calendar years at current rates ----- #
    olf = rm.on_level_factors(
        periods=[("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")],
        rate_changes=[("2023-07-01", 0.08), ("2024-04-01", 0.05)],
        policy_term=1.0,
    )
    print("=== On-level factors ===")
    print(olf.round(4).to_string(index=False))

    # ----- loss side: develop the two origin years to ultimate ----------- #
    triangle = pd.DataFrame(
        {12: [710_000.0, 748_000.0], 24: [845_000.0, np.nan]},
        index=pd.Index([2023, 2024], name="origin"),
    )
    # a 2-origin triangle only supports one factor; real books have more
    cl = ChainLadder.fit(
        pd.concat([triangle, pd.DataFrame(
            {12: [690_000.0], 24: [822_000.0]},
            index=pd.Index([2022], name="origin"))]).sort_index()
    )
    proj = cl.project(triangle)
    print("\n=== Development to ultimate ===")
    print(proj.round(0).to_string())

    # ----- the worksheet -------------------------------------------------- #
    ex = rm.ExperienceExhibit(
        earned_premium=[1_180_000.0, 1_240_000.0],
        losses=proj["latest"].to_numpy(),
        on_level_factors=olf["on_level_factor"].to_numpy(),
        development_factors=proj["development_factor"].to_numpy(),
        trend_factors=[1.045**2, 1.045],       # trend to the rating period
        period_labels=["CY2023", "CY2024"],
    )
    print("\n=== Experience exhibit ===")
    print(ex.exhibit().round(4).to_string())

    # ----- the indication ------------------------------------------------- #
    retention = rm.RetentionLoad(variable_expense_ratio=0.11,
                                 profit_margin=0.03, lae_ratio=0.05)
    ind = ex.to_indication(
        manual_loss_cost=68.0, credibility=0.7, current_rate=86.0,
        exposure=28_800.0, retention=retention,
    )
    print("\n=== Indication ===")
    print(f"experience loss ratio : {ind.experience_loss_ratio():.4f}")
    print(f"indicated rate        : {ind.indicated_rate():.2f}")
    print(f"indicated change      : {ind.indicated_rate_change():+.2%}")


if __name__ == "__main__":
    main()
