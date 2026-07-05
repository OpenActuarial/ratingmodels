"""Out-of-sample model validation, with an interaction on trial.

    temporal_split (leakage-safe) -> fit mains-only vs interaction model
    -> compare_models adjudicates on the training data
    -> lift, Gini, calibration, actual/expected on the HOLDOUT

The interaction is planted in the generator, so the exhibits should side
with the richer model -- and they get to say so out of sample, which is
the only place it counts.

Run with:  python examples/model_validation.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ratingmodels as rm


def _panel(n=24_000, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "month": rng.integers(0, 36, n),
        "area": rng.choice(["A", "B"], n),
        "tier": rng.choice(["bronze", "gold"], n),
        "exposure": rng.uniform(50.0, 150.0, n),
    })
    rel = (np.where(df["area"] == "B", 1.35, 1.0)
           * np.where(df["tier"] == "gold", 0.85, 1.0)
           * np.where((df["area"] == "B") & (df["tier"] == "gold"), 1.3, 1.0))
    df["claims"] = rng.poisson(0.08 * df["exposure"] * rel)
    return df


def main() -> None:
    df = _panel()
    train, test = rm.temporal_split(df, date="month", cutoff=27)
    print(f"train {len(train)} rows (months < 27), holdout {len(test)} rows\n")

    mains = rm.GLMRelativities(family="poisson").fit(
        train, response="claims", predictors=["area", "tier"],
        exposure="exposure")
    inter = rm.GLMRelativities(family="poisson").fit(
        train, response="claims", predictors=["area", "tier"],
        exposure="exposure", interactions=[("area", "tier")])

    print("=== compare_models (training data) ===")
    print(rm.compare_models({"mains": mains, "interaction": inter},
                            train, response="claims",
                            exposure="exposure").round(4).to_string())

    # ----- everything below is out of sample ------------------------------ #
    actual = test["claims"].to_numpy()
    for name, model in [("mains", mains), ("interaction", inter)]:
        pred = model.predict(test, exposure="exposure")
        g = rm.gini_coefficient(actual, pred, exposure=test["exposure"])
        print(f"\nholdout Gini ({name}): {g:.4f}")

    pred = inter.predict(test, exposure="exposure")
    print("\n=== Holdout lift (interaction model, deciles) ===")
    lift = rm.lift_table(actual, pred, exposure=test["exposure"], n_bands=10)
    print(lift.round(4).to_string())

    print("\n=== Holdout calibration ===")
    print(rm.calibration_table(actual, pred, exposure=test["exposure"],
                               n_bands=5).round(4).to_string())

    print("\n=== Holdout actual/expected by area ===")
    ae = rm.actual_expected_table(actual, pred, by=test["area"])
    print(ae.round(4).to_string())


if __name__ == "__main__":
    main()
