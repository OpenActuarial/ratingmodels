"""Thin cells: collapse what cannot be estimated, shrink what barely can.

    sparse rating variable -> collapse_sparse_levels (tiny levels -> Other)
    -> observed relativities vs credibility_relativities
    (Buhlmann-Straub and limited-fluctuation) -> side by side

The thin levels share one true effect, so their observed scatter is pure
noise around a common value. The two methods then disagree in an
instructive way: Buhlmann-Straub shrinks according to how much the
levels *actually* differ (and here the core levels differ a lot, so Z
stays high even for thin cells), while limited-fluctuation shrinks by
volume against a fixed standard, indifferent to the between-level
spread. Neither is wrong -- they answer different questions, and the
exhibit shows both answers.

Run with:  python examples/credibility_smoothing.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ratingmodels as rm


def main() -> None:
    rng = np.random.default_rng(11)
    # two real levels with genuine effects, six thin ones with none
    levels = (["core_a"] * 6000 + ["core_b"] * 4000
              + [f"thin_{i}" for i in range(6) for _ in range(15)])
    df = pd.DataFrame({"segment": levels})
    n = len(df)
    df["exposure"] = rng.uniform(40.0, 90.0, n)
    true_rel = df["segment"].map(
        {"core_a": 1.0, "core_b": 1.3}).fillna(1.0)
    df["claims"] = rng.poisson(0.07 * df["exposure"] * true_rel)

    # ----- collapse what cannot be estimated at all ------------------------ #
    recoded, summary = rm.collapse_sparse_levels(
        df["segment"], exposure=df["exposure"], min_n=150)
    df["segment_grouped"] = recoded
    print("=== Sparse-level collapse ===")
    print(summary.to_string())

    # ----- raw vs credibility-smoothed relativities ------------------------ #
    frames = {
        "buhlmann": rm.credibility_relativities(
            df, factor="segment", response="claims", exposure="exposure",
            method="buhlmann",
        ),
        # square-root rule against the classic 1,082-claim full-credibility
        # standard (90% probability of being within 5%)
        "limited_fluctuation": rm.credibility_relativities(
            df, factor="segment", response="claims", exposure="exposure",
            method="limited_fluctuation", full_credibility=1_082,
        ),
    }
    out = frames["buhlmann"][["n", "observed"]].assign(
        Z_buhlmann=frames["buhlmann"]["credibility"],
        buhlmann=frames["buhlmann"]["relativity"],
        limited_fluctuation=frames["limited_fluctuation"]["relativity"],
    )
    print("\n=== Observed vs credibility relativities ===")
    print(out.round(4).to_string())
    print("\nThe thin levels share one true effect; their observed scatter")
    print("is noise. Buhlmann keeps Z near 0.74 because the CORE levels genuinely")
    print("differ (between-level variance is real); limited-fluctuation")
    print("shrinks the thin cells hard because ~70 claims is nowhere")
    print("near the 1,082 standard. Different questions, different answers.")


if __name__ == "__main__":
    main()
