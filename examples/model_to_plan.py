"""A fitted GLM becomes a filed plan, and the change gets an exhibit.

    GLM -> relativity_table with CIs
    -> predict_interval -> RatingPlan.from_model -> proposed plan
    -> compare_rating_plans -> summary / dislocation / who-absorbs-it

Run with:  python examples/model_to_plan.py
"""
from __future__ import annotations

import warnings

import ratingmodels as rm
from ratingmodels.datasets import sample_rating_data


def main() -> None:
    df = sample_rating_data(n=12_000, seed=11)

    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "industry", "tier"],
        exposure="exposure",
    )
    print("=== Relativities with confidence intervals ===")
    print(model.relativity_table().round(4).head(8).to_string())

    print("\n=== Rate uncertainty for three example cells ===")
    print(model.predict_interval(df.head(3)).round(4).to_string())

    # ----- the model, restated as tables ---------------------------------- #
    current = rm.RatingPlan.from_model(model)
    print("\n=== Plan check against a census ===")
    print(f"unknown levels: {len(current.validate(df))}")
    print(current.average_relativity(df, exposure='exposure').round(4).to_string())

    # ----- a proposed revision, compared ---------------------------------- #
    proposed = rm.RatingPlan.from_dict(current.to_dict())
    proposed.factors["tier"].factors["gold"] *= 1.10
    comp = rm.compare_rating_plans(current, proposed, df, exposure="exposure")
    print("\n=== Plan comparison ===")
    print(comp.summary().round(4).to_string())
    print("\n=== Dislocation ===")
    print(comp.dislocation().round(4).to_string())


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*small sample.*")
    main()
