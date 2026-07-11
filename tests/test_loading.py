"""Contract-pinned loss ratios: the named RetentionLoad constructors."""

import numpy as np
import pandas as pd
import pytest

from ratingmodels import RateIndication, RetentionLoad


class TestFromGrossLossRatio:
    def test_pins_the_ratio(self):
        retention = RetentionLoad.from_gross_loss_ratio(0.85)
        assert retention.gross_rate(450.0) == pytest.approx(450.0 / 0.85, abs=1e-12)
        assert retention.implied_loss_ratio(450.0) == pytest.approx(0.85, abs=1e-12)

    def test_itemization_splits_but_does_not_move_premium(self):
        plain = RetentionLoad.from_gross_loss_ratio(0.85)
        itemized = RetentionLoad.from_gross_loss_ratio(
            0.85, variable_items={"commission": 0.03, "premium_tax": 0.023}
        )
        assert itemized.gross_rate(450.0) == pytest.approx(plain.gross_rate(450.0), abs=1e-12)
        assert itemized.variable_expense_ratio == pytest.approx(0.053)
        assert itemized.profit_margin == pytest.approx(0.15 - 0.053)
        assert itemized.variable_and_profit == pytest.approx(0.15)

    def test_items_exceeding_contractual_retention_raise(self):
        with pytest.raises(ValueError, match="exceeds the contractual retention"):
            RetentionLoad.from_gross_loss_ratio(0.95, variable_items={"commission": 0.08})

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.2, -0.1])
    def test_loss_ratio_bounds(self, bad):
        with pytest.raises(ValueError):
            RetentionLoad.from_gross_loss_ratio(bad)

    def test_series_loss_ratio_rates_a_book(self):
        ratios = pd.Series([0.80, 0.85, 0.90], index=["g1", "g2", "g3"])
        claims = pd.Series([400.0, 450.0, 500.0], index=["g1", "g2", "g3"])
        rates = RetentionLoad.from_gross_loss_ratio(ratios).gross_rate(claims)
        pd.testing.assert_series_equal(rates, claims / ratios, check_names=False)


class TestFromNetLossRatio:
    def test_closed_form(self):
        retention = RetentionLoad.from_net_loss_ratio(
            0.87, fixed_expense=25.0, variable_items={"commission": 0.03}
        )
        expected = (450.0 / 0.87 + 25.0) / (1.0 - 0.03)
        assert retention.gross_rate(450.0) == pytest.approx(expected, abs=1e-12)

    def test_contract_check_is_identical(self):
        retention = RetentionLoad.from_net_loss_ratio(
            0.87, fixed_expense=25.0, variable_items={"commission": 0.03}
        )
        for loss_cost in (300.0, 450.0, 800.0):
            assert retention.implied_net_loss_ratio(loss_cost) == pytest.approx(0.87, abs=1e-12)

    def test_reduces_to_gross_without_expenses(self):
        net = RetentionLoad.from_net_loss_ratio(0.85)
        gross = RetentionLoad.from_gross_loss_ratio(0.85)
        assert net.gross_rate(450.0) == pytest.approx(gross.gross_rate(450.0), abs=1e-12)

    def test_margin_is_claims_proportional(self):
        retention = RetentionLoad.from_net_loss_ratio(
            0.87, fixed_expense=25.0, variable_items={"commission": 0.03}
        )
        for loss_cost in (300.0, 450.0, 800.0):
            premium = retention.gross_rate(loss_cost)
            expenses = 25.0 + 0.03 * premium
            margin = premium - expenses - loss_cost
            assert margin == pytest.approx(loss_cost * (1 - 0.87) / 0.87, abs=1e-9)

    def test_lae_slot_carries_the_gross_up(self):
        retention = RetentionLoad.from_net_loss_ratio(0.87)
        assert retention.lae_ratio == pytest.approx((1 - 0.87) / 0.87)
        assert retention.profit_margin == 0.0


class TestImpliedNetLossRatio:
    def test_generic_retention(self):
        retention = RetentionLoad.from_items(
            fixed_expense=25.0,
            variable_items={"commission": 0.03, "premium_tax": 0.023},
            profit_margin=0.02,
        )
        loss_cost = 450.0
        premium = retention.gross_rate(loss_cost)
        expected = loss_cost / (premium - 25.0 - 0.053 * premium)
        assert retention.implied_net_loss_ratio(loss_cost) == pytest.approx(expected, abs=1e-12)

    def test_zero_claims_is_safe(self):
        retention = RetentionLoad.from_items(fixed_expense=25.0, profit_margin=0.02)
        assert retention.implied_net_loss_ratio(0.0) == 0.0
        assert np.isfinite(RetentionLoad(fixed_expense=25.0).implied_net_loss_ratio(0.0))


class TestIndicationIntegration:
    def test_contract_retention_matches_target_loss_ratio_path(self):
        common = dict(
            experience_loss_cost=480.0,
            manual_loss_cost=460.0,
            credibility=0.9,
            current_rate=540.0,
        )
        via_retention = RateIndication(
            **common, retention=RetentionLoad.from_gross_loss_ratio(0.85)
        )
        via_target = RateIndication(**common, target_loss_ratio=0.85)
        assert via_retention.indicated_rate() == pytest.approx(
            via_target.indicated_rate(), abs=1e-12
        )
        assert via_retention.indicated_rate_change() == pytest.approx(
            via_target.indicated_rate_change(), abs=1e-12
        )
