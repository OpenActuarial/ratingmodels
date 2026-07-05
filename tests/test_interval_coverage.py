"""Empirical coverage of predict_interval: simulate from a known Poisson
GLM, count how often the 95% interval contains the true cell mean."""
import numpy as np
import pandas as pd

import ratingmodels as rm

TRUE_BASE, REL_B, REL_GOLD = 0.10, 1.4, 0.8


def _simulate(rng, n=1500):
    df = pd.DataFrame({
        "area": rng.choice(["A", "B"], n),
        "tier": rng.choice(["bronze", "gold"], n),
        "exposure": rng.uniform(50.0, 150.0, n),
    })
    rel = (np.where(df["area"] == "B", REL_B, 1.0)
           * np.where(df["tier"] == "gold", REL_GOLD, 1.0))
    df["claims"] = rng.poisson(TRUE_BASE * df["exposure"] * rel)
    return df


def test_predict_interval_coverage():
    rng = np.random.default_rng(3)
    cell = pd.DataFrame({"area": ["B"], "tier": ["gold"]})
    truth = TRUE_BASE * REL_B * REL_GOLD
    reps, hits = 200, 0
    for _ in range(reps):
        df = _simulate(rng)
        model = rm.GLMRelativities(family="poisson").fit(
            df, response="claims", predictors=["area", "tier"],
            exposure="exposure", base_levels={"area": "A", "tier": "bronze"},
        )
        pi = model.predict_interval(cell, confidence_level=0.95)
        hits += pi["ci_low"].iloc[0] <= truth <= pi["ci_high"].iloc[0]
    coverage = hits / reps
    # log-linear Wald on a Poisson mean: near-nominal at this volume
    assert 0.90 <= coverage <= 0.99, f"coverage {coverage:.3f}"
