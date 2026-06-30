"""Small synthetic data generators for examples and tests.

These produce reproducible, obviously-synthetic data; they are not real
experience and carry no filed assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_claims(n: int = 500, seed: int = 0) -> np.ndarray:
    """A lognormal claim-size sample with a heavy upper tail."""
    rng = np.random.default_rng(seed)
    body = rng.lognormal(mean=8.0, sigma=1.1, size=n)
    # inject a few large claims to exercise pooling / tail handling
    n_large = max(1, n // 100)
    body[:n_large] = rng.lognormal(mean=12.5, sigma=0.6, size=n_large)
    return np.sort(body)[::-1]


def sample_rating_data(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """A frequency dataset with correlated rating variables and exposure.

    Returns columns: ``area``, ``industry``, ``tier``, ``exposure``
    (member-months) and ``claims`` (counts), generated from known relativities
    so a GLM should approximately recover them.
    """
    rng = np.random.default_rng(seed)
    areas = np.array(["A", "B", "C"])
    industries = np.array(["retail", "manufacturing", "tech"])
    tiers = np.array(["bronze", "silver", "gold"])

    area = rng.choice(areas, size=n, p=[0.5, 0.3, 0.2])
    # industry correlated with area (so one-way analysis would be biased)
    industry = np.where(
        area == "A",
        rng.choice(industries, size=n, p=[0.6, 0.3, 0.1]),
        np.where(
            area == "B",
            rng.choice(industries, size=n, p=[0.2, 0.6, 0.2]),
            rng.choice(industries, size=n, p=[0.1, 0.3, 0.6]),
        ),
    )
    tier = rng.choice(tiers, size=n, p=[0.3, 0.4, 0.3])

    base = 0.08  # base monthly claim frequency
    area_rel = {"A": 1.0, "B": 1.25, "C": 0.85}
    ind_rel = {"retail": 1.0, "manufacturing": 1.4, "tech": 0.75}
    tier_rel = {"bronze": 1.0, "silver": 1.15, "gold": 1.3}

    exposure = rng.integers(12, 240, size=n).astype(float)  # member-months
    mu = (
        base
        * np.array([area_rel[a] for a in area])
        * np.array([ind_rel[i] for i in industry])
        * np.array([tier_rel[t] for t in tier])
        * exposure
    )
    claims = rng.poisson(mu).astype(float)
    return pd.DataFrame(
        {
            "area": area,
            "industry": industry,
            "tier": tier,
            "exposure": exposure,
            "claims": claims,
        }
    )


# the true relativities used by sample_rating_data, for validation
TRUE_RELATIVITIES = {
    "base": 0.08,
    "area": {"A": 1.0, "B": 1.25, "C": 0.85},
    "industry": {"retail": 1.0, "manufacturing": 1.4, "tech": 0.75},
    "tier": {"bronze": 1.0, "silver": 1.15, "gold": 1.3},
}
