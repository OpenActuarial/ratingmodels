"""On-level factors: exact parallelogram geometry on hand-checked cases."""
import numpy as np
import pytest

import ratingmodels as rm


def test_classic_textbook_case():
    # +10% at mid-year, annual policies, calendar-year period:
    # new-rate triangle = 1/8 of earned premium -> average index 1.0125
    tab = rm.on_level_factors(
        periods=[(0.0, 1.0)], rate_changes=[(0.5, 0.10)], policy_term=1.0
    )
    row = tab.iloc[0]
    assert row["average_earned_index"] == pytest.approx(1.0125, abs=1e-12)
    assert row["current_index"] == pytest.approx(1.10)
    assert row["on_level_factor"] == pytest.approx(1.10 / 1.0125, rel=1e-12)


def test_instant_earning_term_zero():
    tab = rm.on_level_factors(
        periods=[(0.0, 1.0)], rate_changes=[(0.5, 0.10)], policy_term=0.0
    )
    assert tab.iloc[0]["average_earned_index"] == pytest.approx(1.05, abs=1e-12)
    assert tab.iloc[0]["on_level_factor"] == pytest.approx(1.10 / 1.05, rel=1e-12)


def test_period_fully_on_level_has_factor_one():
    tab = rm.on_level_factors(
        periods=[(3.0, 4.0)], rate_changes=[(0.5, 0.10), (1.0, 0.05)],
        policy_term=1.0,
    )
    assert tab.iloc[0]["average_earned_index"] == pytest.approx(1.10 * 1.05, rel=1e-12)
    assert tab.iloc[0]["on_level_factor"] == pytest.approx(1.0, rel=1e-12)


def test_period_before_any_change_gets_full_lift():
    tab = rm.on_level_factors(
        periods=[(0.0, 1.0)], rate_changes=[(2.0, 0.08)], policy_term=1.0
    )
    assert tab.iloc[0]["average_earned_index"] == pytest.approx(1.0)
    assert tab.iloc[0]["on_level_factor"] == pytest.approx(1.08)


def test_multiple_periods_and_compounding():
    tab = rm.on_level_factors(
        periods=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
        rate_changes=[(0.5, 0.10), (0.5, 0.05), (1.5, -0.02)],
        policy_term=1.0,
    )
    assert len(tab) == 3
    current = 1.10 * 1.05 * 0.98
    np.testing.assert_allclose(tab["current_index"], current, rtol=1e-12)
    # later periods sit closer to current level -> factors decrease toward 1
    assert tab["on_level_factor"].iloc[0] > tab["on_level_factor"].iloc[1]
    assert tab["on_level_factor"].iloc[2] == pytest.approx(
        current / tab["average_earned_index"].iloc[2], rel=1e-12
    )


def test_datetime_inputs_match_float_equivalents():
    f = rm.on_level_factors(
        periods=[(0.0, 1.0)], rate_changes=[(0.5, 0.10)], policy_term=1.0
    )
    d = rm.on_level_factors(
        periods=[("2024-01-01", "2024-12-31")],
        rate_changes=[("2024-07-01", 0.10)],
        policy_term=1.0,
    )
    # not identical (365 vs 365.25-day year) but the same geometry
    assert d.iloc[0]["on_level_factor"] == pytest.approx(
        f.iloc[0]["on_level_factor"], rel=2e-3
    )
    assert d.iloc[0]["period_start"] == "2024-01-01"


def test_current_date_excludes_later_changes():
    tab = rm.on_level_factors(
        periods=[(0.0, 1.0)],
        rate_changes=[(0.5, 0.10), (5.0, 0.50)],
        policy_term=1.0,
        current_date=2.0,
    )
    assert tab.iloc[0]["current_index"] == pytest.approx(1.10)


def test_no_changes_is_all_ones():
    tab = rm.on_level_factors(periods=[(0.0, 1.0)], rate_changes=[])
    assert tab.iloc[0]["on_level_factor"] == pytest.approx(1.0)


def test_guards():
    with pytest.raises(ValueError, match="end > start"):
        rm.on_level_factors(periods=[(1.0, 1.0)], rate_changes=[])
    with pytest.raises(ValueError, match="-100%"):
        rm.on_level_factors(periods=[(0.0, 1.0)], rate_changes=[(0.5, -1.0)])
    with pytest.raises(ValueError, match="numeric or all datetime"):
        rm.on_level_factors(
            periods=[(0.0, 1.0)], rate_changes=[("not-a-date-xx", 0.1)]
        )


def test_same_date_changes_compound():
    two = rm.on_level_factors(
        periods=[(1.0, 2.0)], rate_changes=[(0.5, 0.10), (0.5, 0.05)],
        policy_term=0.0,
    )
    one = rm.on_level_factors(
        periods=[(1.0, 2.0)], rate_changes=[(0.5, 0.155)], policy_term=0.0,
    )
    assert two.iloc[0]["on_level_factor"] == pytest.approx(
        one.iloc[0]["on_level_factor"], rel=1e-12)


def test_current_date_before_all_changes():
    tab = rm.on_level_factors(
        periods=[(0.0, 1.0)], rate_changes=[(5.0, 0.20)], policy_term=1.0,
        current_date=1.0,
    )
    assert tab.iloc[0]["current_index"] == pytest.approx(1.0)
    assert tab.iloc[0]["on_level_factor"] == pytest.approx(1.0)


def test_translation_invariance():
    base = rm.on_level_factors(
        periods=[(0.0, 1.0), (2.0, 3.0)],
        rate_changes=[(0.5, 0.08), (2.2, 0.05)], policy_term=1.0,
    )
    shift = 7.3
    moved = rm.on_level_factors(
        periods=[(0.0 + shift, 1.0 + shift), (2.0 + shift, 3.0 + shift)],
        rate_changes=[(0.5 + shift, 0.08), (2.2 + shift, 0.05)],
        policy_term=1.0,
    )
    np.testing.assert_allclose(moved["on_level_factor"],
                               base["on_level_factor"], rtol=1e-12)


def test_exact_geometry_matches_fine_grid_integration():
    """The closed-form parallelogram against brute force: integrate the
    same earned index E(t) on a 20,001-point grid and demand agreement.
    Five random change histories, seeded."""
    from ratingmodels.onlevel import _RateIndex

    rng = np.random.default_rng(9)
    for _ in range(5):
        k = rng.integers(2, 6)
        dates = np.sort(rng.uniform(0.0, 4.0, k))
        changes = rng.uniform(-0.10, 0.20, k)
        a, b = 1.0, 3.5
        exact = rm.on_level_factors(
            periods=[(a, b)], rate_changes=list(zip(dates, changes)),
            policy_term=1.0,
        ).iloc[0]["average_earned_index"]
        index = _RateIndex(dates, changes)
        grid = np.linspace(a, b, 20_001)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        brute = trapz(index.earned(grid, 1.0), grid) / (b - a)
        np.testing.assert_allclose(exact, brute, rtol=1e-7)
