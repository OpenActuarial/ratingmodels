"""End-to-end rating example.

Runs the full workflow on illustrative data and prints each step:

    book experience -> base rate (off-balance) -> retention / loading
    -> experience rate -> manual rate -> credibility -> indication
    -> decomposition -> capped renewal -> GLM relativities

Claims and expenses go in; the charged rate and the loss ratio come out.

Run with:  python examples/quickstart.py
"""
from __future__ import annotations

import pandas as pd

import ratingmodels as rm
from ratingmodels.datasets import sample_rating_data


def main() -> None:
    exposure = 96_000  # member-months for the group (~8,000 members)

    # ----- retention: what we load on top of claims ----------------------- #
    # percent-of-premium loads plus a flat admin dollar PMPM; profit margin.
    retention = rm.RetentionLoad.from_items(
        fixed_expense_pmpm=22.0,
        variable_items={"commission": 0.030, "premium_tax": 0.023, "aca_fees": 0.005},
        profit_margin=0.030,
    )
    print("=== Retention ===")
    print(f"fixed expense PMPM    : {retention.fixed_expense_pmpm:.2f}")
    print(f"variable + profit     : {retention.variable_and_profit:.3f} of premium")

    # ----- base rate from book experience (off-balance) ------------------- #
    # one row per rating cell: exposure, rating factors, trended/developed loss.
    book = pd.DataFrame([
        {"cell": "A", "exposure": 24_000, "area": 1.00, "tier": 1.00, "loss": 11_750_000},
        {"cell": "B", "exposure": 18_000, "area": 1.20, "tier": 1.10, "loss": 11_400_000},
        {"cell": "C", "exposure": 30_000, "area": 0.90, "tier": 0.95, "loss": 12_300_000},
        {"cell": "D", "exposure": 24_000, "area": 1.05, "tier": 1.25, "loss": 15_200_000},
    ])
    base = rm.base_rate_from_experience(
        book, exposure="exposure", loss="loss", factor_cols=["area", "tier"]
    )
    print("\n=== Base rate (book) ===")
    print(f"average loss cost PMPM: {base.average_loss_cost:.2f}")
    print(f"average relativity    : {base.average_relativity:.4f}")
    print(f"indicated base PMPM   : {base.base_loss_cost:.2f}  (loss cost / avg relativity)")

    # ----- this group's experience period --------------------------------- #
    incurred = 46_400_000.0                # total incurred claims (~483 PMPM)
    large_claims = [310_000, 420_000, 880_000, 265_000]  # the group's large claimants
    pooling_point = 250_000
    _, excess = rm.pool_claims(large_claims, pooling_point)

    exp = rm.ExperienceRate(
        incurred_claims=incurred, exposure=exposure,
        trend_annual=0.075, trend_years=1.5,
        pooled_excess=excess, pooling_charge_pmpm=4.00,
        benefit_factor=1.00, demographic_factor=1.01,
        retention=retention,
    )
    print("\n=== Experience ===")
    print(f"excess above {pooling_point:,}: {excess:,.0f}  ({excess / exposure:.2f} PMPM)")
    print(f"experience claims PMPM: {exp.claims_pmpm():.2f}")
    print(f"experience rate PMPM  : {exp.rate():.2f}")

    # ----- manual rate (derived base + this group's factors) --------------- #
    man = rm.ManualRate(
        base_pmpm=base.base_loss_cost,
        factors={"area": 1.05, "industry": 0.97, "tier": 1.10},
        retention=retention,
    )
    print("\n=== Manual ===")
    print(f"manual claims PMPM    : {man.claims_pmpm():.2f}")
    print(f"manual rate PMPM      : {man.rate():.2f}")

    # ----- credibility & indication --------------------------------------- #
    z = rm.limited_fluctuation_credibility(n=exposure, n_full=120_000)
    current_rate = 560.0
    ind = rm.RateIndication(
        experience_claims_pmpm=exp.claims_pmpm(),
        manual_claims_pmpm=man.claims_pmpm(),
        credibility=z, current_rate=current_rate,
        current_premium=current_rate * exposure, exposure=exposure,
        trend_total_factor=exp.trend_factor(),
        benefit_factor=1.00, demographic_factor=1.01,
        retention=retention,
    )
    blended_claims = ind.blended_claims_pmpm()
    print("\n=== Indication ===")
    print(f"credibility Z         : {z:.3f}")
    print(f"blended claims PMPM   : {blended_claims:.2f}")
    print(f"indicated rate PMPM   : {ind.indicated_rate():.2f}")
    print(f"indicated change      : {ind.indicated_rate_change():+.2%}")
    # the loss ratio is an OUTPUT of the retention stack, not an input
    print(f"implied loss ratio    : {retention.implied_loss_ratio(blended_claims):.3f}"
          "  (note: large-group ACA MLR floor is 0.85)")

    # ----- why did the rate move? ----------------------------------------- #
    print("\n=== Rate-change decomposition ===")
    decomp = ind.rate_change_decomposition()
    print(decomp.to_frame().round(4).to_string())
    print(f"(contributions sum to {decomp.contributions.sum():+.4f} "
          f"= total change {decomp.total_change:+.4f})")

    # ----- renewal with a cap --------------------------------------------- #
    action = rm.renew(current_rate=current_rate,
                      indicated_rate=ind.indicated_rate(), cap=0.08)
    print("\n=== Renewal action (8% cap) ===")
    for k, v in action.to_dict().items():
        print(f"{k:18}: {v}")

    # ----- manual build-up with an audit trail ---------------------------- #
    # Assemble a medical par claim cost step by step (numbers illustrative),
    # blend par/non-par by participation, then add the drug stream. Every
    # intermediate carries a reconciling breakdown like a rating worksheet.
    print("\n=== Manual build-up (medical par) ===")
    med_par = rm.evaluate([
        rm.start("Par Base Claim Cost", 941.63),
        rm.add("$30 specialist copay", -11.44),
        rm.add("$30 PCP copay", -13.40),
        rm.add("$1,000 IP copay/admission", -6.40),
        rm.add("$150 ER copay", -9.22),
        rm.add("$250 Amb Surg OP copay", -0.56),
        rm.multiply("Rating Region", 1.083),
        rm.multiply("Employer Contribution to Deductible", 1.000),
        rm.checkpoint("Medical Par Base Claim Cost"),
    ])
    print(med_par.breakdown.round(2).to_string(index=False))

    med = rm.participation_blend(med_par.value, 1478.56, participation_rate=0.90)
    total = rm.combine_streams({"Medical": med, "Drug": 323.67}, label="Med + Drug Claim Cost")
    print("\n=== Combine streams ===")
    print(total.breakdown.round(2).to_string(index=False))
    print(f"-> feed {total.value:.2f} PMPM into trend / credibility / retention")

    # ----- GLM relativities on a book ------------------------------------- #
    print("\n=== GLM relativities (Poisson frequency) ===")
    glm_book = sample_rating_data(n=20_000, seed=42)
    model = rm.GLMRelativities(family="poisson").fit(
        glm_book, response="claims", predictors=["area", "industry", "tier"],
        exposure="exposure",
        base_levels={"area": "A", "industry": "retail", "tier": "bronze"},
    )
    print(f"base frequency        : {model.base_value_:.4f}")
    for var in ["area", "industry", "tier"]:
        print(f"{var:10} relativities: "
              + ", ".join(f"{lvl}={rel:.3f}" for lvl, rel in model.relativities_[var].items()))


if __name__ == "__main__":
    main()
