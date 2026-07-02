import pytest

import ratingmodels as rm
from ratingmodels import PricingEvaluation, scenario_frame, uplift_for_target_margin


RETENTION = rm.RetentionLoad(
    fixed_expense_pmpm=8.0,
    variable_expense_ratio=0.10,
    profit_margin=0.03,
    lae_ratio=0.02,
)


def make_case(**overrides):
    kwargs = dict(
        claims_pmpm=400.0,
        current_rate=480.0,
        retention=RETENTION,
        member_months=12_000.0,
        persistency=0.85,
    )
    kwargs.update(overrides)
    return PricingEvaluation(**kwargs)


# --------------------------------------------------------------------------- #
# forward evaluation
# --------------------------------------------------------------------------- #
def test_margin_ratio_at_indicated_rate_equals_profit_margin():
    case = make_case()
    indicated = RETENTION.gross_rate(case.claims_pmpm)
    change = indicated / case.current_rate - 1.0
    outcome = case.at(change)
    assert outcome.margin_ratio == pytest.approx(RETENTION.profit_margin)


def test_forward_expense_algebra():
    case = make_case()
    outcome = case.at(0.05, name="issued")
    premium = 480.0 * 1.05
    benefit = 400.0 * 1.02
    admin = 8.0 + 0.10 * premium
    assert outcome.name == "issued"
    assert outcome.premium_pmpm == pytest.approx(premium)
    assert outcome.benefit_pmpm == pytest.approx(benefit)
    assert outcome.admin_pmpm == pytest.approx(admin)
    assert outcome.loss_ratio == pytest.approx(400.0 / premium)
    assert outcome.gross_margin_pmpm == pytest.approx(premium - benefit)
    assert outcome.margin_pmpm == pytest.approx(premium - benefit - admin)
    assert outcome.margin_ratio == pytest.approx(outcome.margin_pmpm / premium)


def test_dollar_and_persistency_fields():
    case = make_case()
    outcome = case.at(0.05)
    assert outcome.premium == pytest.approx(outcome.premium_pmpm * 12_000.0)
    assert outcome.margin == pytest.approx(outcome.margin_pmpm * 12_000.0)
    assert outcome.expected_premium == pytest.approx(outcome.premium * 0.85)
    assert outcome.expected_margin == pytest.approx(outcome.margin * 0.85)

    bare = make_case(member_months=None, persistency=None).at(0.05)
    assert bare.premium is None
    assert bare.expected_margin is None


def test_no_retention_margin_equals_gross_margin():
    case = make_case(retention=None)
    outcome = case.at(0.10)
    assert outcome.admin_pmpm == pytest.approx(0.0)
    assert outcome.margin_pmpm == pytest.approx(outcome.gross_margin_pmpm)
    assert outcome.benefit_pmpm == pytest.approx(case.claims_pmpm)


def test_nonpositive_premium_rejected():
    with pytest.raises(ValueError, match="premium"):
        make_case().at(-1.0)


# --------------------------------------------------------------------------- #
# inverse solves
# --------------------------------------------------------------------------- #
def test_inverse_forward_roundtrip():
    case = make_case()
    for target in (-0.05, 0.0, 0.02, 0.06):
        change = case.rate_change_for_margin(target)
        assert case.at(change).margin_ratio == pytest.approx(target)


def test_zero_margin_rate_change():
    case = make_case()
    outcome = case.at(case.zero_margin_rate_change())
    assert outcome.margin_pmpm == pytest.approx(0.0, abs=1e-9)


def test_indication_is_the_profit_margin_special_case():
    case = make_case()
    at_profit = case.premium_for_margin(RETENTION.profit_margin)
    assert at_profit == pytest.approx(RETENTION.gross_rate(case.claims_pmpm))


def test_no_retention_inverse_reduces_to_loss_ratio_form():
    case = make_case(retention=None)
    premium = case.premium_for_margin(0.15)
    assert premium == pytest.approx(400.0 / 0.85)
    assert case.at(case.rate_change_for_margin(0.15)).loss_ratio == pytest.approx(0.85)


def test_infeasible_margin_target_rejected():
    case = make_case()
    with pytest.raises(ValueError, match="target_margin"):
        case.premium_for_margin(0.95)


# --------------------------------------------------------------------------- #
# indication bridge
# --------------------------------------------------------------------------- #
def test_from_indication_adopts_blend_and_retention():
    indication = rm.RateIndication(
        experience_claims_pmpm=420.0,
        manual_claims_pmpm=380.0,
        credibility=0.6,
        current_rate=480.0,
        retention=RETENTION,
    )
    case = PricingEvaluation.from_indication(indication, member_months=6_000.0)
    assert case.claims_pmpm == pytest.approx(indication.blended_claims_pmpm())
    outcome = case.at(indication.indicated_rate_change())
    assert outcome.margin_ratio == pytest.approx(RETENTION.profit_margin)


# --------------------------------------------------------------------------- #
# scenario_frame
# --------------------------------------------------------------------------- #
def test_scenario_frame_tidy_shape():
    cases = {"a": make_case(), "b": make_case(claims_pmpm=350.0, persistency=0.5)}
    scenarios = {
        "formula": {"a": 0.16, "b": 0.08},
        "issued": {"a": 0.11, "b": 0.05},
        "plan": 0.12,
    }
    frame = scenario_frame(cases, scenarios)
    assert frame.shape[0] == 6
    assert set(frame["scenario"]) == {"formula", "issued", "plan"}
    plan_a = frame[(frame["case"] == "a") & (frame["scenario"] == "plan")].iloc[0]
    assert plan_a["rate_change"] == pytest.approx(0.12)
    assert plan_a["expected_premium"] == pytest.approx(
        480.0 * 1.12 * 12_000.0 * 0.85
    )
    # a cohort rollup is a groupby of the tidy table
    aggregate = frame[frame["scenario"] == "issued"]
    ratio = aggregate["margin"].sum() / aggregate["premium"].sum()
    assert -1.0 < ratio < 1.0


def test_scenario_frame_missing_case_action_raises():
    cases = {"a": make_case(), "b": make_case()}
    with pytest.raises(KeyError, match="no rate change"):
        scenario_frame(cases, {"issued": {"a": 0.1}})


def test_scenario_frame_drops_absent_optional_columns():
    cases = {"a": make_case(member_months=None, persistency=None)}
    frame = scenario_frame(cases, {"plan": 0.1})
    assert "premium" not in frame.columns
    assert "expected_margin" not in frame.columns
    assert "premium_pmpm" in frame.columns


# --------------------------------------------------------------------------- #
# uplift_for_target_margin
# --------------------------------------------------------------------------- #
def small_book():
    return {
        "a": make_case(claims_pmpm=400.0, current_rate=480.0, member_months=12_000.0, persistency=0.85),
        "b": make_case(claims_pmpm=360.0, current_rate=430.0, member_months=4_000.0, persistency=0.60),
        "c": make_case(claims_pmpm=300.0, current_rate=390.0, member_months=7_000.0, persistency=0.90),
    }


def aggregate_margin_ratio(cases, changes):
    margin = premium = 0.0
    for case_id, case in cases.items():
        outcome = case.at(changes[case_id])
        margin += outcome.expected_margin
        premium += outcome.expected_premium
    return margin / premium


def test_multiplicative_uplift_hits_target_exactly():
    cases = small_book()
    base = {"a": 0.11, "b": 0.05, "c": 0.02}
    target = 0.03
    base_ratio = aggregate_margin_ratio(cases, base)
    uplift = uplift_for_target_margin(cases, base, target)
    # direction must oppose the shortfall/excess at the base actions
    assert (uplift > 0) == (base_ratio < target)
    lifted = {k: (1 + a) * (1 + uplift) - 1 for k, a in base.items()}
    assert aggregate_margin_ratio(cases, lifted) == pytest.approx(target)


def test_additive_uplift_hits_target_exactly():
    cases = small_book()
    base = {"a": 0.11, "b": 0.05, "c": 0.02}
    target = 0.03
    uplift = uplift_for_target_margin(cases, base, target, mode="additive")
    lifted = {k: a + uplift for k, a in base.items()}
    assert aggregate_margin_ratio(cases, lifted) == pytest.approx(target)


def test_uplift_negative_when_already_above_target():
    cases = small_book()
    base = {"a": 0.30, "b": 0.30, "c": 0.30}
    uplift = uplift_for_target_margin(cases, base, 0.01)
    assert uplift < 0


def test_uplift_broadcasts_constant_base_change():
    cases = small_book()
    uplift = uplift_for_target_margin(cases, 0.05, 0.02)
    lifted = {k: (1 + 0.05) * (1 + uplift) - 1 for k in cases}
    assert aggregate_margin_ratio(cases, lifted) == pytest.approx(0.02)


def test_uplift_infeasible_target_rejected():
    with pytest.raises(ValueError, match="not attainable"):
        uplift_for_target_margin(small_book(), 0.05, 0.95)


def test_uplift_missing_base_change_raises():
    cases = small_book()
    with pytest.raises(KeyError, match="base_changes"):
        uplift_for_target_margin(cases, {"a": 0.1, "b": 0.1}, 0.02)
