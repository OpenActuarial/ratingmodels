"""RatingPlan: rating, unknown-level policy, round-trip, comparison."""
import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm
from ratingmodels.datasets import sample_rating_data


@pytest.fixture()
def plan():
    return rm.RatingPlan(
        base_rate=500.0,
        factors={
            "area": rm.FactorTable(
                name="area", factors={"A": 1.0, "B": 1.25, "C": 0.8}, default=1.0
            ),
            "tier": rm.FactorTable(
                name="tier", factors={"bronze": 1.0, "gold": 1.5}, default=1.0
            ),
        },
    )


@pytest.fixture()
def census():
    return pd.DataFrame(
        {
            "area": ["A", "B", "C", "B"],
            "tier": ["bronze", "gold", "bronze", "bronze"],
            "members": [10.0, 20.0, 5.0, 15.0],
        }
    )


def test_rate_decomposition_exact(plan, census):
    rated = plan.rate(census, exposure="members")
    assert list(rated.columns) == [
        "base_rate", "area_factor", "tier_factor",
        "combined_relativity", "rate", "premium",
    ]
    # row 1: B x gold = 500 * 1.25 * 1.5
    assert rated.loc[1, "rate"] == pytest.approx(937.5)
    assert rated.loc[1, "premium"] == pytest.approx(18750.0)
    np.testing.assert_allclose(
        rated["rate"], rated["base_rate"] * rated["combined_relativity"], rtol=1e-12
    )


def test_unknown_level_policies(plan, census):
    census2 = census.copy()
    census2.loc[0, "area"] = "Z"
    bad = plan.validate(census2)
    assert ("area", "Z") in bad.index and bad.loc[("area", "Z"), "n"] == 1
    # default policy: table default (1.0) applied
    rated = plan.rate(census2)
    assert rated.loc[0, "area_factor"] == 1.0
    # error policy: hard stop naming the level
    with pytest.raises(ValueError, match="area='Z'"):
        plan.rate(census2, unknown="error")
    # clean census validates empty
    assert len(plan.validate(census)) == 0


def test_column_mapping(plan, census):
    renamed = census.rename(columns={"area": "area_code"})
    rated = plan.rate(renamed, columns={"area": "area_code"})
    np.testing.assert_allclose(rated["rate"], plan.rate(census)["rate"], rtol=1e-12)
    with pytest.raises(ValueError, match="not found"):
        plan.rate(renamed)


def test_average_relativity_weights(plan, census):
    avg = plan.average_relativity(census, exposure="members")
    w = census["members"].to_numpy()
    fac = census["area"].map(plan.factors["area"].factors).to_numpy()
    assert avg["area"] == pytest.approx(np.average(fac, weights=w))
    assert set(avg.index) == {"area", "tier", "combined"}


def test_dict_round_trip(plan, census):
    rebuilt = rm.RatingPlan.from_dict(plan.to_dict())
    np.testing.assert_allclose(
        rebuilt.rate(census)["rate"], plan.rate(census)["rate"], rtol=1e-12
    )
    with pytest.raises(ValueError, match="schema"):
        rm.RatingPlan.from_dict({"schema": 99, "base_rate": 1.0, "factors": {}})


def test_from_model_reproduces_predictions():
    df = sample_rating_data(n=3000, seed=17)
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "industry", "tier"],
        exposure="exposure",
    )
    plan = rm.RatingPlan.from_model(model)
    rated = plan.rate(df, exposure="exposure")
    np.testing.assert_allclose(
        rated["premium"], model.predict(df, exposure="exposure"), rtol=1e-10
    )


def test_compare_rating_plans(plan, census):
    proposed = rm.RatingPlan.from_dict(plan.to_dict())
    proposed.factors["area"].factors["B"] = 1.5   # +20% on B
    comp = rm.compare_rating_plans(plan, proposed, census, exposure="members")

    s = comp.summary()
    assert s["n"] == 4
    b_mask = census["area"] == "B"
    cur = plan.rate(census, exposure="members")["premium"]
    prop = proposed.rate(census, exposure="members")["premium"]
    assert s["current_premium"] == pytest.approx(cur.sum())
    assert s["proposed_premium"] == pytest.approx(prop.sum())
    assert s["avg_change"] == pytest.approx(prop.sum() / cur.sum() - 1)
    assert s["share_increasing"] == pytest.approx(
        census.loc[b_mask, "members"].sum() / census["members"].sum()
    )
    assert s["share_unchanged"] + s["share_increasing"] == pytest.approx(1.0)

    d = comp.dislocation()
    assert d.loc["All", "proposed_premium"] == pytest.approx(prop.sum())

    by = comp.by(census["area"])
    assert by.loc["B", "avg_change"] == pytest.approx(0.2)
    assert by.loc["A", "avg_change"] == pytest.approx(0.0)


def test_plan_guards():
    with pytest.raises(ValueError, match="positive"):
        rm.RatingPlan(base_rate=0.0, factors={})
    with pytest.raises(TypeError, match="FactorTable"):
        rm.RatingPlan(base_rate=1.0, factors={"x": {"a": 1.0}})


def test_from_model_warns_on_unrepresentable_terms():
    df = sample_rating_data(n=2000, seed=21)
    df["age"] = np.random.default_rng(21).uniform(-5, 5, len(df))
    cont = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area"], continuous=["age"],
        exposure="exposure",
    )
    with pytest.warns(UserWarning, match="continuous covariates"):
        rm.RatingPlan.from_model(cont)
    ix = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "tier"],
        exposure="exposure", interactions=[("area", "tier")],
    )
    with pytest.warns(UserWarning, match="interaction terms"):
        rm.RatingPlan.from_model(ix)


def test_from_model_pure_categorical_is_silent(recwarn):
    df = sample_rating_data(n=2000, seed=22)
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "tier"], exposure="exposure",
    )
    rm.RatingPlan.from_model(model)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_comparison_custom_bands_and_label_guard(plan, census):
    proposed = rm.RatingPlan.from_dict(plan.to_dict())
    proposed.factors["area"].factors["B"] = 1.5
    comp = rm.compare_rating_plans(plan, proposed, census, exposure="members")
    d = comp.dislocation(bands=(0.0, 0.15))
    assert "All" in d.index and len(d) == 4  # 3 bands + total
    with pytest.raises(ValueError, match="align"):
        comp.by(np.array(["x", "y"]))


def test_json_round_trip_with_string_levels(plan, census):
    import json

    rebuilt = rm.RatingPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    np.testing.assert_allclose(
        rebuilt.rate(census)["rate"], plan.rate(census)["rate"], rtol=1e-12
    )


def test_json_stringifies_integer_levels_documented_footgun():
    import json

    plan = rm.RatingPlan(
        base_rate=100.0,
        factors={"terr": rm.FactorTable(name="terr", factors={1: 1.2, 2: 0.9},
                                        default=1.0)},
    )
    direct = rm.RatingPlan.from_dict(plan.to_dict())
    assert direct.factors["terr"].lookup(1) == 1.2  # dict path preserves types
    via_json = rm.RatingPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    # JSON keys are strings: the typed lookup now misses to the default,
    # exactly as the to_dict docstring warns
    assert via_json.factors["terr"].lookup(1) == 1.0
    assert via_json.factors["terr"].lookup("1") == 1.2


def test_compare_plans_propagates_unknown_policy(plan, census):
    census2 = census.copy()
    census2.loc[0, "area"] = "Z"
    with pytest.raises(ValueError, match="area='Z'"):
        rm.compare_rating_plans(plan, plan, census2, unknown="error")
