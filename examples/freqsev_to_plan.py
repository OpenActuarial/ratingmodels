"""Frequency-severity modeling, interactions included, ending in a plan.

    claims frame -> FrequencySeverityModel (Poisson x Gamma) with a
    frequency-side interaction -> pure premium = frequency x severity
    (bit-exact identity) -> combined_relativities (incl. the interaction
    cell) -> to_factor_tables -> RatingPlan -> rate a census

The plan step warns, correctly: an interaction has no single-variable
factor-table form, so the plan carries the mains only.

Run with:  python examples/freqsev_to_plan.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import ratingmodels as rm


def _claims_frame(n=20_000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "area": rng.choice(["urban", "rural"], n, p=[0.6, 0.4]),
        "industry": rng.choice(["retail", "manufacturing"], n),
        "member_months": rng.uniform(60.0, 140.0, n),
    })
    f_rel = (np.where(df["area"] == "urban", 1.25, 1.0)
             * np.where(df["industry"] == "manufacturing", 1.1, 1.0)
             * np.where((df["area"] == "urban")
                        & (df["industry"] == "manufacturing"), 1.2, 1.0))
    counts = rng.poisson(0.06 * df["member_months"] * f_rel)
    sev = 850.0 * np.where(df["industry"] == "manufacturing", 1.15, 1.0)
    df["claims"] = counts
    df["allowed"] = counts * sev * rng.gamma(50.0, 1 / 50.0, n)
    return df


def main() -> None:
    df = _claims_frame()
    model = rm.FrequencySeverityModel().fit(
        df, claim_count="claims", claim_amount="allowed",
        exposure="member_months",
        frequency_predictors=["area", "industry"],
        base_levels={"area": "rural", "industry": "retail"},
        frequency_interactions=[("area", "industry")],
        severity_interactions=[],           # severity stays mains-only
    )

    # ----- the identity that defines the model ---------------------------- #
    pp = model.pure_premium_prediction(df.head(), exposure="member_months")
    parts = (model.frequency_prediction(df.head(), exposure="member_months")
             * model.severity_prediction(df.head()))
    print("=== Pure premium = frequency x severity (max |diff|) ===")
    print(f"{np.abs(pp - parts).max():.3e}\n")

    # ----- combined structure, interaction cell included ------------------ #
    print("=== Combined relativities: industry ===")
    print(model.combined_relativities()["industry"].round(4).to_string())
    print("\n=== Combined relativities: area:industry (frequency-side) ===")
    print(model.combined_relativities()["area:industry"].round(4).to_string())

    # ----- into a plan (with the honest warning) --------------------------- #
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = rm.RatingPlan.from_model(model)
    print(f"\nfrom_model warned: {caught[0].message}")
    rated = plan.rate(df.head(4), exposure="member_months")
    print("\n=== Plan applied to a census slice ===")
    print(rated.round(2).to_string())


if __name__ == "__main__":
    main()
