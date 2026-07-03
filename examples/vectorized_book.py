"""Vectorized rating: price a whole book with DataFrames in, DataFrames out.

Every numeric argument in ratingmodels follows one contract -- scalar in,
float out; column in, column out -- so the same code that rates one group
rates a book. This example runs the full workflow over a three-group book
without a single Python loop:

    claim file -> grouped pooling -> experience rates -> manual rates
    -> credibility -> indication -> per-case decomposition
    -> capped renewals -> pricing scenarios -> book-level uplift

Run with:  python examples/vectorized_book.py
"""
from __future__ import annotations

import pandas as pd

import ratingmodels as rm

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 20)


def main() -> None:
    # ----- the book: one row per group ------------------------------------ #
    book = pd.DataFrame(
        {
            "exposure": [9_600.0, 14_400.0, 6_000.0],       # member-months
            "current_rate": [545.0, 560.0, 530.0],          # charged today
            "base": [420.0, 435.0, 410.0],                  # manual base loss cost
            "area": [1.05, 0.98, 1.12],                     # manual relativities
            "industry": [1.10, 1.00, 0.95],
            "n_claims": [820.0, 1450.0, 260.0],             # credibility counts
            "persistency": [0.90, 0.95, 0.80],
        },
        index=pd.Index(["G1", "G2", "G3"], name="group"),
    )

    # ----- large-claim file -> pooled excess per group, one groupby pass -- #
    # the claim file lists individual large claimants; routine claims come in
    # as a bulk total per group.
    large_claims = pd.DataFrame(
        {
            "group": ["G1", "G1", "G2", "G3", "G3"],
            "amount": [390_000.0, 310_000.0, 420_000.0, 610_000.0, 260_000.0],
        }
    )
    bulk_claims = pd.Series([3.65e6, 5.88e6, 2.03e6], index=book.index)
    pooling_point = 250_000.0
    _, excess = rm.pool_claims(
        large_claims["amount"], pooling_point, by=large_claims["group"]
    )
    incurred = bulk_claims + large_claims.groupby("group")["amount"].sum()
    print("=== Incurred and pooled excess (per group) ===")
    print(pd.DataFrame({"incurred": incurred, "excess_over_250k": excess}), "\n")

    # ----- experience and manual rates, whole book at once ---------------- #
    retention = rm.RetentionLoad(
        fixed_expense=12.0, variable_expense_ratio=0.11,
        profit_margin=0.03, lae_ratio=0.02,
    )
    experience = rm.ExperienceRate(
        incurred_claims=incurred,
        exposure=book["exposure"],
        trend_annual=0.07, trend_years=1.5,
        pooled_excess=excess,
        pooling_charge=28.0,
        retention=retention,
    )
    manual = rm.ManualRate(
        book["base"],
        {"area": book["area"], "industry": book["industry"]},
        retention=retention,
    )
    z = rm.limited_fluctuation_credibility(book["n_claims"], n_full=1_082.0)

    print("=== Manual build-up, tidy long breakdown (first rows) ===")
    print(manual.breakdown().to_frame().head(6).round(4), "\n")

    # ----- indication + per-case driver decomposition --------------------- #
    indication = rm.RateIndication(
        experience_loss_cost=experience.loss_cost(),
        manual_loss_cost=manual.loss_cost(),
        credibility=z,
        current_rate=book["current_rate"],
        trend_total_factor=experience.trend_factor(),
        retention=retention,
    )
    summary = book.assign(
        experience_lc=experience.loss_cost().round(2),
        manual_lc=manual.loss_cost().round(2),
        Z=z.round(3),
        indicated=indication.indicated_rate().round(2),
        change=indication.indicated_rate_change().round(4),
    )
    print("=== Indication (one row per group) ===")
    print(summary[["exposure", "experience_lc", "manual_lc", "Z", "indicated", "change"]], "\n")

    print("=== Rate-change decomposition, (case, driver) long table ===")
    print(indication.rate_change_decomposition().to_frame().round(4).head(8), "\n")

    # ----- capped renewal actions ----------------------------------------- #
    action = rm.renew(
        book["current_rate"], indication.indicated_rate(),
        cap=pd.Series([0.10, 0.10, 0.12], index=book.index), floor=0.0,
    )
    print("=== Renewal actions ===")
    print(action.to_frame().round(4), "\n")

    # ----- pricing scenarios over the book --------------------------------- #
    evaluation = rm.PricingEvaluation(
        loss_cost=indication.blended_loss_cost(),
        current_rate=book["current_rate"],
        retention=retention,
        exposure=book["exposure"],
        persistency=book["persistency"],
    )
    scenarios = rm.scenario_frame(
        evaluation,
        {
            "formula": indication.indicated_rate_change(),
            "issued": action.proposed_change,
            "flat 5%": 0.05,
        },
    )
    print("=== Scenario economics (tidy long) ===")
    cols = ["case", "scenario", "rate_change", "margin_ratio", "expected_margin"]
    print(scenarios[cols].round(4).to_string(index=False), "\n")

    uplift = rm.uplift_for_target_margin(
        evaluation, base_changes=action.proposed_change, target_margin=0.03
    )
    print(f"Uniform uplift restoring a 3% book margin over issued actions: {uplift:+.4%}")


if __name__ == "__main__":
    main()
