import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm
from ratingmodels.datasets import (
    TRUE_RELATIVITIES,
    sample_claims,
    sample_rating_data,
)


# --------------------------------------------------------------------------- #
# credibility
# --------------------------------------------------------------------------- #
def test_full_credibility_standard_classic():
    # p=0.90, k=0.05 -> ~1082 claims
    assert round(rm.full_credibility_standard(0.90, 0.05)) == 1082


def test_full_credibility_severity_inflates():
    base = rm.full_credibility_standard(0.95, 0.05)
    with_sev = rm.full_credibility_standard(0.95, 0.05, cv_severity=1.0)
    assert with_sev == pytest.approx(2 * base)


def test_limited_fluctuation_sqrt_rule_and_cap():
    assert rm.limited_fluctuation_credibility(30000, 120000) == pytest.approx(0.5)
    assert rm.limited_fluctuation_credibility(200000, 120000) == 1.0


def test_buhlmann_basic():
    # k = EPV/VHM = 100/25 = 4 ; Z = 16/(16+4) = 0.8
    assert rm.buhlmann_credibility(16, epv=100, vhm=25) == pytest.approx(0.8)


def test_buhlmann_straub_recovers_structure():
    # build grouped data where group means clearly differ -> high credibility
    rng = np.random.default_rng(1)
    rows = []
    group_means = {"g1": 100.0, "g2": 200.0, "g3": 300.0}
    for g, gm in group_means.items():
        for period in range(8):
            expo = rng.integers(50, 150)
            val = rng.normal(gm, 10)  # small within-group noise
            rows.append({"group": g, "period": period, "value": val, "exposure": expo})
    df = pd.DataFrame(rows)
    res = rm.buhlmann_straub(df, "group", "period", "value", "exposure")
    assert res.epv > 0
    assert res.vhm > 0
    # strong between-group signal, low noise -> credibility near 1
    assert (res.credibility > 0.9).all()
    # with Z ~ 1 the blended estimate barely shrinks: it tracks each group's
    # own (sample) mean, not the overall mean
    for g in group_means:
        assert res.credibility_weighted[g] == pytest.approx(
            res.group_means[g], rel=0.02
        )
    # and the ordering of the true means is preserved
    assert (
        res.credibility_weighted["g1"]
        < res.credibility_weighted["g2"]
        < res.credibility_weighted["g3"]
    )


def test_buhlmann_straub_no_signal_truncates():
    # all groups identical in expectation -> vhm truncated to 0, Z = 0
    rng = np.random.default_rng(2)
    rows = []
    for g in ["a", "b", "c", "d"]:
        for period in range(10):
            rows.append(
                {
                    "group": g,
                    "period": period,
                    "value": rng.normal(100, 30),
                    "exposure": rng.integers(40, 60),
                }
            )
    df = pd.DataFrame(rows)
    res = rm.buhlmann_straub(df, "group", "period", "value", "exposure")
    assert res.vhm == 0.0
    assert (res.credibility == 0.0).all()


# --------------------------------------------------------------------------- #
# trend
# --------------------------------------------------------------------------- #
def test_trend_factor_compounds():
    assert rm.trend_factor(0.075, 2.0) == pytest.approx(1.075**2)


def test_trend_factor_between_midpoints():
    f = rm.trend_factor_between(
        0.10,
        experience_period=("2023-01-01", "2023-12-31"),
        rating_period=("2025-01-01", "2025-12-31"),
    )
    # midpoints ~2 years apart
    assert f == pytest.approx(1.10**2, rel=1e-2)


def test_combine_and_split_trend_roundtrip():
    total = rm.combine_trend(0.03, 0.045)
    assert rm.split_total_trend(total, 0.03) == pytest.approx(0.045)


def test_trend_requires_valid_rate():
    with pytest.raises(ValueError):
        rm.trend_factor(-1.5, 1.0)


# --------------------------------------------------------------------------- #
# pooling / experience / manual
# --------------------------------------------------------------------------- #
def test_pool_claims_removes_excess():
    claims = [100_000, 600_000, 50_000, 900_000]
    capped, excess = rm.pool_claims(claims, pooling_point=500_000)
    # excess = (600k-500k) + (900k-500k) = 500k
    assert excess == pytest.approx(500_000)
    assert capped == pytest.approx(sum(claims) - 500_000)


def test_experience_rate_pipeline():
    exp = rm.ExperienceRate(
        incurred_claims=4_000_000,
        exposure=100_000,
        trend_annual=0.08,
        trend_years=1.5,
        pooled_excess=400_000,
        pooling_charge_pmpm=3.0,
        target_loss_ratio=0.80,
    )
    pooled_pmpm = (4_000_000 - 400_000) / 100_000  # 36
    trended = pooled_pmpm * (1.08**1.5)
    expected_claims = trended + 3.0
    assert exp.claims_pmpm() == pytest.approx(expected_claims)
    assert exp.rate() == pytest.approx(expected_claims / 0.80)


def test_manual_rate_product():
    man = rm.ManualRate(
        base_pmpm=500,
        factors={"area": 1.10, "industry": 0.90, "tier": 1.05},
        target_loss_ratio=0.85,
    )
    claims = 500 * 1.10 * 0.90 * 1.05
    assert man.claims_pmpm() == pytest.approx(claims)
    assert man.rate() == pytest.approx(claims / 0.85)


def test_aggregate_demographic_factor():
    census = pd.DataFrame({"members": [100, 300], "age_sex": [0.9, 1.2]})
    assert rm.aggregate_demographic_factor(census, "age_sex") == pytest.approx(
        (100 * 0.9 + 300 * 1.2) / 400
    )


# --------------------------------------------------------------------------- #
# blend / indication
# --------------------------------------------------------------------------- #
def test_blend_convex():
    assert rm.blend(100, 200, 0.0) == 200
    assert rm.blend(100, 200, 1.0) == 100
    assert rm.blend(100, 200, 0.25) == pytest.approx(175)


def test_indication_buildup():
    ind = rm.RateIndication(
        experience_claims_pmpm=420,
        manual_claims_pmpm=460,
        credibility=0.5,
        current_rate=560,
        target_loss_ratio=0.85,
        trend_total_factor=1.10,
    )
    blended_claims = 0.5 * 420 + 0.5 * 460  # 440
    indicated_rate = blended_claims / 0.85
    assert ind.blended_claims_pmpm() == pytest.approx(blended_claims)
    assert ind.indicated_rate() == pytest.approx(indicated_rate)
    assert ind.indicated_rate_change() == pytest.approx(indicated_rate / 560 - 1)


def test_indication_loss_ratio_method():
    ind = rm.RateIndication(
        experience_claims_pmpm=400,
        manual_claims_pmpm=420,
        credibility=1.0,
        current_rate=500,
        target_loss_ratio=0.80,
        current_premium=480 * 100_000 / 100_000,  # set premium below
        exposure=100_000,
        trend_total_factor=1.08,
    )
    # with Z=1 the loss-ratio indication = exp_LR/target - 1
    exp_lr = (400 * 100_000) / ind.current_premium
    assert ind.loss_ratio_indication() == pytest.approx(exp_lr / 0.80 - 1)


def test_loss_ratio_method_requires_premium():
    ind = rm.RateIndication(
        experience_claims_pmpm=400,
        manual_claims_pmpm=420,
        credibility=0.5,
        current_rate=500,
    )
    with pytest.raises(ValueError):
        ind.loss_ratio_indication()


# --------------------------------------------------------------------------- #
# decomposition
# --------------------------------------------------------------------------- #
def test_decomposition_contributions_sum_to_total():
    d = rm.decompose_rate_change(
        {"trend": 1.075, "experience": 0.96, "benefit": 1.02, "demographic": 1.01}
    )
    assert d.contributions.sum() == pytest.approx(d.total_change)


def test_decomposition_residual_reconciles():
    d = rm.decompose_rate_change(
        {"trend": 1.075, "experience": 0.96},
        total_factor=1.10,
    )
    assert "residual" in d.factors.index
    # product of all factors equals the total
    assert float(np.prod(d.factors.to_numpy())) == pytest.approx(1.10)
    assert d.contributions.sum() == pytest.approx(0.10)


def test_indication_decomposition_reconciles():
    ind = rm.RateIndication(
        experience_claims_pmpm=420,
        manual_claims_pmpm=460,
        credibility=0.6,
        current_rate=540,
        target_loss_ratio=0.85,
        trend_total_factor=1.09,
        benefit_factor=1.02,
        demographic_factor=1.01,
    )
    d = ind.rate_change_decomposition()
    assert d.total_factor == pytest.approx(ind.indicated_rate() / ind.current_rate)
    assert d.contributions.sum() == pytest.approx(d.total_change)


# --------------------------------------------------------------------------- #
# constraints / renewal
# --------------------------------------------------------------------------- #
def test_cap_and_floor():
    assert rm.cap_change(0.30, cap=0.15) == 0.15
    assert rm.cap_change(-0.20, floor=-0.10) == -0.10
    assert rm.cap_change(0.05, cap=0.15, floor=-0.10) == 0.05


def test_renew_caps_change():
    action = rm.renew(current_rate=500, indicated_rate=650, cap=0.15)
    assert action.capped is True
    assert action.proposed_rate == pytest.approx(575.0)  # 500 * 1.15
    assert action.proposed_change == pytest.approx(0.15)


def test_renew_uncapped_passthrough():
    action = rm.renew(current_rate=500, indicated_rate=520, cap=0.15)
    assert action.capped is False
    assert action.proposed_rate == pytest.approx(520.0)


def test_band_deadband_and_step():
    assert rm.band(0.005, deadband=0.01) == 0.0
    assert rm.band(0.037, step=0.005) == pytest.approx(0.035)


def test_member_level_renewal_rolls_up():
    census = pd.DataFrame(
        {"members": [10, 20], "area": [1.1, 0.9], "tier": [1.0, 1.2]}
    )
    out = rm.member_level_renewal(census, base_rate=400, factor_cols=["area", "tier"])
    assert out["member_rate"].iloc[0] == pytest.approx(400 * 1.1 * 1.0)
    assert out["premium"].sum() == pytest.approx(
        400 * 1.1 * 1.0 * 10 + 400 * 0.9 * 1.2 * 20
    )


# --------------------------------------------------------------------------- #
# GLM relativities
# --------------------------------------------------------------------------- #
def test_glm_recovers_known_relativities():
    df = sample_rating_data(n=20000, seed=7)
    # reference levels matching the data-generating process (all 1.0 there)
    bases = {"area": "A", "industry": "retail", "tier": "bronze"}
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "industry", "tier"],
        exposure="exposure", base_levels=bases,
    )
    # base frequency recovered
    assert model.base_value_ == pytest.approx(TRUE_RELATIVITIES["base"], rel=0.15)
    for var in ["area", "industry", "tier"]:
        for level, true_rel in TRUE_RELATIVITIES[var].items():
            assert model.relativities_[var][level] == pytest.approx(true_rel, rel=0.12)


def test_glm_predict_shape_and_positivity():
    df = sample_rating_data(n=2000, seed=3)
    model = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "tier"], exposure="exposure"
    )
    pred = model.predict(df, exposure="exposure")
    assert pred.shape == (len(df),)
    assert np.all(pred > 0)


def test_glm_one_way_vs_glm_differ_under_correlation():
    df = sample_rating_data(n=20000, seed=11)
    one_way = rm.one_way_relativities(
        df, factor="industry", response="claims", exposure="exposure",
        base_level="retail",
    )
    glm = rm.GLMRelativities(family="poisson").fit(
        df, response="claims", predictors=["area", "industry", "tier"],
        exposure="exposure", base_levels={"industry": "retail"},
    ).relativities_["industry"]
    # GLM should be closer to truth than one-way for the correlated variable
    truth = TRUE_RELATIVITIES["industry"]
    glm_err = sum(abs(glm[k] - v) for k, v in truth.items())
    ow_err = sum(abs(one_way[k] - v) for k, v in truth.items())
    assert glm_err < ow_err


def test_glm_gamma_family_runs():
    rng = np.random.default_rng(5)
    n = 3000
    area = rng.choice(["A", "B"], size=n)
    sev = np.where(area == "A", 1000, 1500) * rng.gamma(shape=2.0, scale=0.5, size=n)
    df = pd.DataFrame({"area": area, "severity": sev})
    model = rm.GLMRelativities(family="gamma").fit(
        df, response="severity", predictors=["area"], base_levels={"area": "A"}
    )
    # B/A severity ratio ~1.5
    assert model.relativities_["area"]["B"] == pytest.approx(1.5, rel=0.15)


def test_tweedie_requires_power():
    with pytest.raises(ValueError):
        rm.GLMRelativities(family="tweedie").fit(
            sample_rating_data(n=500), response="claims",
            predictors=["area"], exposure="exposure",
        )


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def test_sample_claims_has_heavy_tail():
    claims = sample_claims(n=1000, seed=0)
    # top claim should dwarf the median (heavy tail present)
    assert claims.max() > 20 * np.median(claims)


# --------------------------------------------------------------------------- #
# retention / loading
# --------------------------------------------------------------------------- #
def test_retention_gross_up_with_fixed_and_variable():
    ret = rm.RetentionLoad(
        fixed_expense_pmpm=20.0, variable_expense_ratio=0.15, profit_margin=0.05
    )
    # P = (400 + 20) / (1 - 0.20) = 525
    assert ret.gross_rate(400.0) == pytest.approx(525.0)
    # loss ratio is an OUTPUT: 400 / 525
    assert ret.implied_loss_ratio(400.0) == pytest.approx(400.0 / 525.0)


def test_retention_no_fixed_gives_constant_plr():
    ret = rm.RetentionLoad(variable_expense_ratio=0.18, profit_margin=0.04)
    # with no fixed expense the permissible loss ratio is 1 - V - Q for any level
    assert ret.implied_loss_ratio(300.0) == pytest.approx(0.78)
    assert ret.implied_loss_ratio(900.0) == pytest.approx(0.78)
    assert ret.gross_rate(390.0) == pytest.approx(390.0 / 0.78)


def test_retention_lae_loads_claims():
    ret = rm.RetentionLoad(variable_expense_ratio=0.20, lae_ratio=0.10)
    # P = 400 * 1.10 / (1 - 0.20) = 550
    assert ret.gross_rate(400.0) == pytest.approx(550.0)


def test_retention_from_items_sums_variable():
    ret = rm.RetentionLoad.from_items(
        fixed_expense_pmpm=15.0,
        variable_items={"commission": 0.04, "premium_tax": 0.023, "aca_fees": 0.005},
        profit_margin=0.03,
    )
    assert ret.variable_expense_ratio == pytest.approx(0.068)
    assert ret.variable_and_profit == pytest.approx(0.098)


def test_retention_rejects_infeasible_load():
    with pytest.raises(ValueError):
        rm.RetentionLoad(variable_expense_ratio=0.8, profit_margin=0.25)


def test_manual_rate_with_retention_overrides_loss_ratio():
    ret = rm.RetentionLoad(fixed_expense_pmpm=18.0, variable_expense_ratio=0.16, profit_margin=0.04)
    man = rm.ManualRate(base_pmpm=500, factors={"area": 1.1}, retention=ret)
    claims = 500 * 1.1
    assert man.rate() == pytest.approx(ret.gross_rate(claims))
    # fixed expense applied AFTER base*relativity (flat per member)
    assert man.rate() == pytest.approx((claims + 18.0) / (1 - 0.20))


def test_indication_with_retention():
    ret = rm.RetentionLoad(fixed_expense_pmpm=25.0, variable_expense_ratio=0.15, profit_margin=0.05)
    ind = rm.RateIndication(
        experience_claims_pmpm=420, manual_claims_pmpm=460,
        credibility=0.5, current_rate=600, retention=ret,
        trend_total_factor=1.08,
    )
    blended = 0.5 * 420 + 0.5 * 460  # 440
    assert ind.indicated_rate() == pytest.approx(ret.gross_rate(blended))
    assert ind.blended_rate() == pytest.approx((440 + 25.0) / 0.80)


# --------------------------------------------------------------------------- #
# base rate / off-balance
# --------------------------------------------------------------------------- #
def _book_with_known_base(base=300.0):
    # losses constructed so base*relativity*exposure reproduces them exactly
    rows = [
        {"exposure": 1000, "area": 1.0, "tier": 1.0},
        {"exposure": 600, "area": 1.20, "tier": 1.10},
        {"exposure": 400, "area": 0.85, "tier": 0.95},
        {"exposure": 800, "area": 1.05, "tier": 1.25},
    ]
    df = pd.DataFrame(rows)
    df["relativity"] = df["area"] * df["tier"]
    df["loss"] = base * df["relativity"] * df["exposure"]
    return df, base


def test_base_rate_recovers_known_base():
    df, base = _book_with_known_base(300.0)
    res = rm.base_rate_from_experience(df, exposure="exposure", loss="loss",
                                       relativity="relativity")
    assert res.base_loss_cost == pytest.approx(base)
    # base * relativities reproduces total losses
    rebuilt = (res.base_loss_cost * df["relativity"] * df["exposure"]).sum()
    assert rebuilt == pytest.approx(df["loss"].sum())


def test_base_rate_from_factor_cols_matches_relativity_col():
    df, _ = _book_with_known_base(250.0)
    a = rm.base_rate_from_experience(df, "exposure", "loss", relativity="relativity")
    b = rm.base_rate_from_experience(df, "exposure", "loss", factor_cols=["area", "tier"])
    assert a.base_loss_cost == pytest.approx(b.base_loss_cost)
    assert a.average_relativity == pytest.approx(b.average_relativity)


def test_average_relativity_exposure_weighted():
    df, _ = _book_with_known_base()
    avg = rm.average_relativity(df, "exposure", relativity="relativity")
    expected = (df["relativity"] * df["exposure"]).sum() / df["exposure"].sum()
    assert avg == pytest.approx(expected)


def test_rebalance_holds_level_then_applies_change():
    # revise relativities so average rises 1.00 -> 1.05; hold level, then +8%
    b1 = rm.rebalance_base_rate(current_base=300.0, current_avg_relativity=1.00,
                                new_avg_relativity=1.05, overall_change=0.08)
    assert b1 == pytest.approx(300.0 * (1.00 / 1.05) * 1.08)
    # off-balance factor alone
    assert rm.off_balance_factor(1.00, 1.05) == pytest.approx(1.0 / 1.05)


def test_base_rate_then_retention_reconciles_total_premium():
    # full chain: indicated base from experience, then gross to a charged rate;
    # total charged premium must equal the required premium from the equation
    df, base = _book_with_known_base(280.0)
    res = rm.base_rate_from_experience(df, "exposure", "loss", relativity="relativity")
    ret = rm.RetentionLoad(fixed_expense_pmpm=30.0, variable_expense_ratio=0.12,
                           profit_margin=0.05)
    # charged premium per cell = (base*rel*exp ... ) handled per member:
    total_claims = df["loss"].sum()
    total_exposure = df["exposure"].sum()
    charged = sum(
        ret.gross_rate(res.base_loss_cost * row["relativity"]) * row["exposure"]
        for _, row in df.iterrows()
    )
    required = (total_claims + ret.fixed_expense_pmpm * total_exposure) / (1 - 0.17)
    assert charged == pytest.approx(required)


# --------------------------------------------------------------------------- #
# build-up engine
# --------------------------------------------------------------------------- #
def test_buildup_basic_arithmetic_and_reconciles():
    res = rm.evaluate([
        rm.start("Base", 100.0),
        rm.multiply("Factor A", 1.10),
        rm.add("Copay credit", -5.0),
        rm.checkpoint("Subtotal"),
    ])
    # 100 * 1.10 - 5 = 105
    assert res.value == pytest.approx(105.0)
    assert res.subtotal("Subtotal") == pytest.approx(105.0)
    # the breakdown's last running total equals the value
    assert res.breakdown["running_total"].iloc[-1] == pytest.approx(res.value)
    assert list(res.breakdown.columns) == [
        "step", "operation", "label", "operand", "running_total"
    ]


def test_buildup_running_total_reconciles_stepwise():
    res = rm.evaluate([
        rm.start("Base", 1000.0),
        rm.add("credit", -40.0),
        rm.multiply("region", 1.05),
        rm.multiply("network", 0.98),
    ])
    expected = (1000.0 - 40.0) * 1.05 * 0.98
    assert res.value == pytest.approx(expected)


def test_segment_multiply_partial_application():
    # apply a 1.20 factor to half the cost: 100 * (1 + 0.5*(1.20-1)) = 110
    res = rm.evaluate([rm.start("Base", 100.0),
                       rm.segment_multiply("Half at 1.20", 1.20, 0.5)])
    assert res.value == pytest.approx(110.0)
    # weight 1 == plain multiply; weight 0 == no-op
    assert rm.evaluate([rm.start("b", 100.0),
                        rm.segment_multiply("all", 1.2, 1.0)]).value == pytest.approx(120.0)
    assert rm.evaluate([rm.start("b", 100.0),
                        rm.segment_multiply("none", 1.2, 0.0)]).value == pytest.approx(100.0)


def test_segment_multiply_weight_validated():
    with pytest.raises(ValueError):
        rm.segment_multiply("bad", 1.2, 1.5)


def test_checkpoint_does_not_change_total():
    res = rm.evaluate([
        rm.start("Base", 50.0),
        rm.checkpoint("after start"),
        rm.multiply("f", 2.0),
        rm.checkpoint("after multiply"),
    ])
    assert res.subtotal("after start") == pytest.approx(50.0)
    assert res.subtotal("after multiply") == pytest.approx(100.0)
    assert res.value == pytest.approx(100.0)


def test_fluent_builder_matches_list():
    steps = [rm.start("B", 200.0), rm.multiply("a", 1.1), rm.add("c", -10.0)]
    via_list = rm.evaluate(steps)
    builder = rm.BuildUp().start("B", 200.0).multiply("a", 1.1).add("c", -10.0)
    assert builder.steps() == steps
    assert builder.evaluate().value == pytest.approx(via_list.value)


def test_participation_blend_matches_worksheet():
    # par 975.32 @ 90% + non-par 1478.56 @ 10% = 1025.644
    res = rm.participation_blend(975.32, 1478.56, 0.90)
    assert res.value == pytest.approx(975.32 * 0.9 + 1478.56 * 0.1)
    assert res.value == pytest.approx(1025.644)


def test_combine_streams_sums_with_running_total():
    res = rm.combine_streams({"Medical": 1025.644, "Drug": 323.67}, label="Med + Drug")
    assert res.value == pytest.approx(1349.314)
    assert res.subtotal("Med + Drug") == pytest.approx(1349.314)
    assert res.breakdown["running_total"].iloc[-1] == pytest.approx(res.value)


def test_combine_accepts_buildupresult_inputs():
    med = rm.participation_blend(975.32, 1478.56, 0.90)
    drug = rm.evaluate([rm.start("Drug", 323.67)])
    res = rm.combine_streams({"Medical": med, "Drug": drug})
    assert res.value == pytest.approx(1025.644 + 323.67)


def test_manual_rate_breakdown_reconciles_to_claims():
    man = rm.ManualRate(base_pmpm=480.0,
                        factors={"area": 1.05, "industry": 0.97, "tier": 1.10})
    bd = man.breakdown()
    assert bd.value == pytest.approx(man.claims_pmpm())
    # one start + three factors + one checkpoint = 5 rows
    assert len(bd.breakdown) == 5
    assert bd.subtotal("Manual claims PMPM") == pytest.approx(man.claims_pmpm())
