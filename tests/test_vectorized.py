"""Vectorization contract tests.

Every public numeric entry point: scalar in -> float out (unchanged 0.4.x
behavior); Series in -> Series out on the same index, numerically identical
to a row-by-row scalar loop; scalars broadcast; one bad row fails the call
with a labeled error; reducing helpers raise on mismatched Series indexes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ratingmodels as rm
from ratingmodels._utils import product

IDX = pd.Index(["g1", "g2", "g3"], name="group")


def series(*vals) -> pd.Series:
    return pd.Series(list(vals), index=IDX)


def assert_matches_loop(vec, scalars, index=IDX):
    """Vector result equals the elementwise scalar-loop reference and keeps the index."""
    assert isinstance(vec, pd.Series)
    assert vec.index.equals(index)
    np.testing.assert_allclose(vec.to_numpy(), np.asarray(scalars, dtype=float), rtol=1e-12)


# --------------------------------------------------------------------------- #
# _utils
# --------------------------------------------------------------------------- #
class TestUtils:
    def test_validators_scalar_and_vector(self):
        from ratingmodels._utils import require_positive

        assert require_positive(2, "x") == 2.0
        out = require_positive(series(1.0, 2.0, 3.0), "x")
        assert isinstance(out, pd.Series) and out.index.equals(IDX)
        arr = require_positive([1.0, 2.0], "x")
        assert isinstance(arr, np.ndarray)

    def test_validator_error_names_label(self):
        from ratingmodels._utils import require_positive

        with pytest.raises(ValueError, match="'g2'"):
            require_positive(series(1.0, -1.0, 2.0), "exposure")

    def test_product_scalar_matches_04x(self):
        assert product([1.05, 1.10, 0.97]) == pytest.approx(1.05 * 1.10 * 0.97)
        assert product([]) == 1.0

    def test_product_reduces_across_factors_not_rows(self):
        f1, f2 = series(1.05, 0.98, 1.12), series(1.10, 1.00, 0.95)
        out = product([f1, f2, 2.0])  # scalar broadcasts
        assert_matches_loop(out, [a * b * 2.0 for a, b in zip(f1, f2)])

    def test_product_mismatched_index_raises(self):
        other = pd.Series([1.0, 1.0, 1.0], index=["x", "y", "z"])
        with pytest.raises(ValueError, match="share one index"):
            product([series(1.0, 1.0, 1.0), other])


# --------------------------------------------------------------------------- #
# trend
# --------------------------------------------------------------------------- #
class TestTrend:
    def test_trend_factor_vector(self):
        t = series(0.05, 0.07, -0.02)
        assert_matches_loop(rm.trend_factor(t, 1.5), [rm.trend_factor(x, 1.5) for x in t])

    def test_apply_and_combine(self):
        v, t, y = series(100.0, 200.0, 50.0), 0.06, series(1.0, 2.0, 0.5)
        assert_matches_loop(
            rm.apply_trend(v, t, y),
            [rm.apply_trend(a, t, b) for a, b in zip(v, y)],
        )
        u = series(0.03, 0.04, 0.05)
        assert_matches_loop(rm.combine_trend(u, 0.02), [rm.combine_trend(x, 0.02) for x in u])
        assert_matches_loop(
            rm.split_total_trend(0.10, u), [rm.split_total_trend(0.10, x) for x in u]
        )

    def test_vector_dates(self):
        starts = pd.Series(pd.to_datetime(["2025-01-01", "2025-07-01", "2025-01-01"]), index=IDX)
        ends = pd.Series(pd.to_datetime(["2025-12-31", "2026-06-30", "2026-12-30"]), index=IDX)
        yrs = rm.years_between(starts, ends)
        ref = [rm.years_between(s.date(), e.date()) for s, e in zip(starts, ends)]
        assert_matches_loop(yrs, ref)
        mid = rm.period_midpoint(starts, ends)
        ref_mid = [rm.period_midpoint(s.date(), e.date()) for s, e in zip(starts, ends)]
        assert mid.dt.date.tolist() == ref_mid
        f = rm.trend_factor_between(0.07, (starts, ends), ("2027-01-01", "2027-12-31"))
        assert isinstance(f, pd.Series) and f.index.equals(IDX)

    def test_trend_mismatched_index_raises(self):
        with pytest.raises(ValueError, match="share one index"):
            rm.trend_factor(series(0.05, 0.05, 0.05), pd.Series([1.0, 2.0, 3.0]))

    def test_negative_base_rejected_elementwise(self):
        with pytest.raises(ValueError, match="exceed -1"):
            rm.trend_factor(series(0.05, -1.5, 0.02), 1.0)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
class TestRetentionLoad:
    RET = rm.RetentionLoad(fixed_expense=12.0, variable_expense_ratio=0.11,
                           profit_margin=0.03, lae_ratio=0.02)

    def test_gross_rate_vector_matches_loop(self):
        lc = series(430.0, 460.0, 500.0)
        assert_matches_loop(self.RET.gross_rate(lc), [self.RET.gross_rate(x) for x in lc])

    def test_per_row_retention_fields(self):
        ret = rm.RetentionLoad(variable_expense_ratio=series(0.10, 0.12, 0.11))
        out = ret.gross_rate(430.0)
        ref = [rm.RetentionLoad(variable_expense_ratio=v).gross_rate(430.0)
               for v in (0.10, 0.12, 0.11)]
        assert_matches_loop(out, ref)

    def test_implied_loss_ratio_zero_and_vector(self):
        assert self.RET.implied_loss_ratio(0.0) == 0.0
        lc = series(0.0, 430.0, 500.0)
        out = self.RET.implied_loss_ratio(lc)
        ref = [self.RET.implied_loss_ratio(float(x)) for x in lc]
        assert_matches_loop(out, ref)

    def test_vector_validation(self):
        with pytest.raises(ValueError, match="non-negative"):
            rm.RetentionLoad(fixed_expense=series(1.0, -2.0, 0.0))
        with pytest.raises(ValueError, match="< 1"):
            rm.RetentionLoad(variable_expense_ratio=series(0.1, 0.99, 0.1),
                             profit_margin=0.03)


# --------------------------------------------------------------------------- #
# constraints / renewal
# --------------------------------------------------------------------------- #
class TestConstraintsRenewal:
    def test_constraints_elementwise(self):
        c = series(0.004, -0.03, 0.16)
        assert_matches_loop(rm.cap_change(c, cap=0.10, floor=-0.02),
                            [rm.cap_change(x, 0.10, -0.02) for x in c])
        assert_matches_loop(rm.band(c, deadband=0.005, step=0.01),
                            [rm.band(x, 0.005, 0.01) for x in c])
        cur, ind = series(500.0, 480.0, 610.0), series(575.0, 470.0, 700.0)
        assert_matches_loop(rm.corridor(cur, ind, 0.10, 0.02),
                            [rm.corridor(a, b, 0.10, 0.02) for a, b in zip(cur, ind)])

    def test_per_row_caps(self):
        cur, ind = series(500.0, 480.0, 610.0), series(575.0, 470.0, 700.0)
        caps = series(0.10, 0.10, 0.12)
        out = rm.apply_cap(cur, ind, cap=caps, floor=0.0)
        ref = [rm.apply_cap(a, b, cap=k, floor=0.0) for a, b, k in zip(cur, ind, caps)]
        assert_matches_loop(out, ref)

    def test_renew_vector_matches_loop(self):
        cur, ind = series(500.0, 480.0, 610.0), series(575.0, 470.0, 700.0)
        act = rm.renew(cur, ind, cap=0.10, floor=0.0)
        refs = [rm.renew(a, b, cap=0.10, floor=0.0) for a, b in zip(cur, ind)]
        for fld in ("proposed_rate", "indicated_change", "proposed_change"):
            assert_matches_loop(getattr(act, fld), [getattr(r, fld) for r in refs])
        assert isinstance(act.capped, pd.Series)
        assert act.capped.tolist() == [r.capped for r in refs]
        frame = act.to_frame()
        assert list(frame.index) == list(IDX) and "capped" in frame.columns

    def test_renew_scalar_types_unchanged(self):
        a = rm.renew(500.0, 575.0, cap=0.10)
        assert isinstance(a.proposed_rate, float) and isinstance(a.capped, bool)
        assert len(a.to_frame()) == 1

    def test_unit_level_renewal_vectorized_matches_manual(self):
        census = pd.DataFrame({"age": [1.1, 0.9, 1.0], "area": [1.05, 1.2, 0.8],
                               "count": [10, 4, 6]})
        out = rm.unit_level_renewal(census, 400.0, ["age", "area"])
        np.testing.assert_allclose(
            out["unit_rate"], 400.0 * census["age"] * census["area"], rtol=1e-12
        )
        np.testing.assert_allclose(out["premium"], out["unit_rate"] * census["count"])


# --------------------------------------------------------------------------- #
# credibility / blend
# --------------------------------------------------------------------------- #
class TestCredibilityBlend:
    def test_limited_fluctuation_vector(self):
        n = series(820.0, 1450.0, 260.0)
        assert_matches_loop(rm.limited_fluctuation_credibility(n, 1082.0),
                            [rm.limited_fluctuation_credibility(x, 1082.0) for x in n])

    def test_buhlmann_vector(self):
        n = series(820.0, 1450.0, 260.0)
        assert_matches_loop(rm.buhlmann_credibility(n, 4.0, 0.02),
                            [rm.buhlmann_credibility(x, 4.0, 0.02) for x in n])

    def test_blend_vector_and_alignment_guard(self):
        e, m, z = series(430.0, 420.0, 460.0), 445.0, series(0.87, 1.0, 0.49)
        assert_matches_loop(rm.blend(e, m, z), [rm.blend(a, m, b) for a, b in zip(e, z)])
        with pytest.raises(ValueError, match="share one index"):
            rm.blend(e, m, pd.Series([0.5, 0.5, 0.5]))


# --------------------------------------------------------------------------- #
# manual rate
# --------------------------------------------------------------------------- #
class TestManualRate:
    BOOK = pd.DataFrame(
        {"base": [420.0, 435.0, 410.0], "area": [1.05, 0.98, 1.12],
         "industry": [1.10, 1.00, 0.95]},
        index=IDX,
    )

    def loop(self, method):
        return [getattr(rm.ManualRate(r.base, {"a": r.area, "i": r.industry}), method)()
                for r in self.BOOK.itertuples()]

    def make_vec(self):
        b = self.BOOK
        return rm.ManualRate(b["base"], {"a": b["area"], "i": b["industry"]})

    def test_loss_cost_and_rate(self):
        assert_matches_loop(self.make_vec().loss_cost(), self.loop("loss_cost"))
        assert_matches_loop(self.make_vec().rate(), self.loop("rate"))

    def test_manual_loss_cost_function(self):
        b = self.BOOK
        out = rm.manual_loss_cost(b["base"], [b["area"], b["industry"]])
        assert_matches_loop(out, [r.base * r.area * r.industry for r in b.itertuples()])

    def test_breakdown_long_format(self):
        r = self.make_vec().breakdown()
        assert isinstance(r.value, pd.Series)
        assert "entity" in r.breakdown.columns
        final = r.breakdown[r.breakdown["step"] == r.breakdown["step"].max()]
        np.testing.assert_allclose(
            final.set_index("entity")["running_total"].reindex(IDX),
            r.value, rtol=1e-12,
        )

    def test_with_factor_keeps_retention(self):
        ret = rm.RetentionLoad(10.0, 0.1, 0.03)
        m = rm.ManualRate(430.0, {"area": 1.05}, retention=ret)
        assert m.with_factor("ind", 1.1).retention is ret

    def test_aggregate_demographic_factor_by(self):
        census = pd.DataFrame({"grp": ["A", "A", "B"], "f": [0.9, 1.3, 1.0],
                               "count": [10, 5, 7]})
        out = rm.aggregate_demographic_factor(census, "f", by="grp")
        assert out["A"] == pytest.approx(np.average([0.9, 1.3], weights=[10, 5]))
        assert out["B"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# experience rate
# --------------------------------------------------------------------------- #
class TestExperienceRate:
    KW = dict(trend_annual=0.07, trend_years=1.5, pooling_charge=28.0,
              benefit_factor=1.02, demographic_factor=0.99)
    EXP = pd.DataFrame(
        {"claims": [4_128_000.0, 6_048_000.0, 2_760_000.0],
         "exposure": [9_600.0, 14_400.0, 6_000.0],
         "pooled_x": [310_000.0, 0.0, 505_000.0]},
        index=IDX,
    )

    def test_loss_cost_and_rate_match_loop(self):
        e = self.EXP
        vec = rm.ExperienceRate(e["claims"], e["exposure"], pooled_excess=e["pooled_x"], **self.KW)
        refs = [rm.ExperienceRate(r.claims, r.exposure, pooled_excess=r.pooled_x, **self.KW)
                for r in e.itertuples()]
        assert_matches_loop(vec.loss_cost(), [r.loss_cost() for r in refs])
        assert_matches_loop(vec.rate(), [r.rate() for r in refs])

    def test_retention_path_vectorized(self):
        ret = rm.RetentionLoad(12.0, 0.11, 0.03, 0.02)
        e = self.EXP
        vec = rm.ExperienceRate(e["claims"], e["exposure"], retention=ret, **self.KW)
        refs = [rm.ExperienceRate(r.claims, r.exposure, retention=ret, **self.KW).rate()
                for r in e.itertuples()]
        assert_matches_loop(vec.rate(), refs)

    def test_bad_row_labeled(self):
        e = self.EXP
        with pytest.raises(ValueError, match="'g2'"):
            rm.ExperienceRate(e["claims"], series(9600.0, -5.0, 6000.0))

    def test_pool_claims_by(self):
        claims = pd.Series([120e3, 40e3, 310e3, 15e3, 90e3])
        grp = ["G1", "G1", "G2", "G2", "G2"]
        capped, excess = rm.pool_claims(claims, 100e3, by=grp)
        c1, x1 = rm.pool_claims([120e3, 40e3], 100e3)
        c2, x2 = rm.pool_claims([310e3, 15e3, 90e3], 100e3)
        assert capped.to_dict() == {"G1": c1, "G2": c2}
        assert excess.to_dict() == {"G1": x1, "G2": x2}
        charge = rm.expected_excess_charge(claims, 100e3, {"G1": 9600.0, "G2": 6000.0}, by=grp)
        assert charge["G1"] == pytest.approx(x1 / 9600.0)
        assert charge["G2"] == pytest.approx(x2 / 6000.0)

    def test_pool_claims_scalar_unchanged(self):
        capped, excess = rm.pool_claims([120e3, 40e3], 100e3)
        assert isinstance(capped, float) and isinstance(excess, float)


# --------------------------------------------------------------------------- #
# base rate
# --------------------------------------------------------------------------- #
class TestBaseRate:
    BOOK = pd.DataFrame(
        {"seg": ["A", "A", "B", "B"], "mm": [3000.0, 5000.0, 2000.0, 4000.0],
         "loss": [1.29e6, 2.2e6, 0.96e6, 1.72e6],
         "f1": [1.1, 0.95, 1.2, 1.0], "f2": [1.0, 1.05, 0.9, 1.1]}
    )

    def test_factor_cols_matches_relativity_column(self):
        b = self.BOOK.assign(rel=lambda d: d.f1 * d.f2)
        via_cols = rm.base_rate_from_experience(b, "mm", "loss", factor_cols=["f1", "f2"])
        via_rel = rm.base_rate_from_experience(b, "mm", "loss", relativity="rel")
        assert via_cols.base_loss_cost == pytest.approx(via_rel.base_loss_cost, rel=1e-12)

    def test_by_groups_match_per_segment_calls(self):
        out = rm.base_rate_from_experience(self.BOOK, "mm", "loss",
                                           factor_cols=["f1", "f2"], by="seg")
        assert isinstance(out, pd.DataFrame) and list(out.index) == ["A", "B"]
        for seg in ("A", "B"):
            one = rm.base_rate_from_experience(
                self.BOOK[self.BOOK.seg == seg], "mm", "loss", factor_cols=["f1", "f2"]
            )
            assert out.loc[seg, "base_loss_cost"] == pytest.approx(one.base_loss_cost)
            assert out.loc[seg, "total_exposure"] == pytest.approx(one.total_exposure)

    def test_average_relativity_by(self):
        out = rm.average_relativity(self.BOOK, "mm", factor_cols=["f1", "f2"], by="seg")
        one = rm.average_relativity(self.BOOK[self.BOOK.seg == "A"], "mm",
                                    factor_cols=["f1", "f2"])
        assert out["A"] == pytest.approx(one)

    def test_off_balance_vector(self):
        cur, new = series(1.02, 1.05, 1.00), series(1.00, 1.10, 1.04)
        assert_matches_loop(rm.off_balance_factor(cur, new),
                            [rm.off_balance_factor(a, b) for a, b in zip(cur, new)])
        assert_matches_loop(
            rm.rebalance_base_rate(series(420.0, 430.0, 410.0), cur, new, 0.03),
            [rm.rebalance_base_rate(b, a, c, 0.03)
             for b, a, c in zip(series(420.0, 430.0, 410.0), cur, new)],
        )


# --------------------------------------------------------------------------- #
# build-up engine
# --------------------------------------------------------------------------- #
class TestBuildUp:
    def test_scalar_breakdown_shape_unchanged(self):
        r = (rm.BuildUp().start("Base", 941.63).add("copay", -11.44)
             .multiply("Region", 1.083).checkpoint("Med").evaluate())
        assert list(r.breakdown.columns) == ["step", "operation", "label", "operand", "running_total"]
        assert isinstance(r.value, float)

    def test_vector_matches_scalar_loop(self):
        base = series(941.63, 812.10, 1004.55)
        region = series(1.083, 0.981, 1.140)
        w = series(0.25, 0.30, 0.20)
        vec = (rm.BuildUp().start("Base", base).add("copay", -11.44)
               .multiply("Region", region).segment_multiply("Rx", 0.92, w)
               .checkpoint("Med").evaluate())
        refs = [
            (rm.BuildUp().start("Base", b).add("copay", -11.44)
             .multiply("Region", g).segment_multiply("Rx", 0.92, ww)
             .checkpoint("Med").evaluate())
            for b, g, ww in zip(base, region, w)
        ]
        assert_matches_loop(vec.value, [r.value for r in refs])
        assert_matches_loop(vec.subtotal("Med"), [r.subtotal("Med") for r in refs])
        assert "entity" in vec.breakdown.columns
        assert len(vec.breakdown) == 5 * 3

    def test_streams_and_participation_vector(self):
        med = series(950.0, 760.0, 1100.0)
        drug = series(210.0, 190.0, 240.0)
        combined = rm.combine_streams({"Medical": med, "Drug": drug})
        assert_matches_loop(combined.value, med + drug)
        p = series(0.9, 0.85, 0.95)
        pb = rm.participation_blend(med, 700.0, p)
        assert_matches_loop(pb.value, [m * q + 700.0 * (1 - q) for m, q in zip(med, p)])

    def test_mixed_lengths_raise(self):
        with pytest.raises(ValueError, match="length"):
            rm.evaluate([rm.start("a", np.array([1.0, 2.0])),
                         rm.multiply("b", np.array([1.0, 2.0, 3.0]))])

    def test_mismatched_indexes_raise(self):
        with pytest.raises(ValueError, match="share one index"):
            rm.evaluate([rm.start("a", series(1.0, 2.0, 3.0)),
                         rm.multiply("b", pd.Series([1.0, 1.0, 1.0]))])


# --------------------------------------------------------------------------- #
# decomposition / indication
# --------------------------------------------------------------------------- #
class TestDecompositionIndication:
    def test_vector_decomposition_matches_scalar_loop(self):
        trend = series(1.075, 1.06, 1.09)
        expf = series(0.96, 1.02, 0.99)
        total = series(1.05, 1.10, 1.06)
        d = rm.decompose_rate_change({"trend": trend, "experience": expf}, total_factor=total)
        assert isinstance(d.factors, pd.DataFrame)
        for g in IDX:
            ds = rm.decompose_rate_change(
                {"trend": float(trend[g]), "experience": float(expf[g])},
                total_factor=float(total[g]),
            )
            np.testing.assert_allclose(
                d.contributions.loc[g].to_numpy(), ds.contributions.to_numpy(), rtol=1e-12
            )
        np.testing.assert_allclose(
            d.contributions.sum(axis=1), np.asarray(total) - 1.0, rtol=1e-12
        )
        long = d.to_frame()
        assert long.index.names == ["case", "driver"]

    def test_unity_total_rows_zero(self):
        d = rm.decompose_rate_change({"a": series(1.0, 1.1, 1.0),
                                      "b": series(1.0, 1 / 1.1, 1.0)})
        assert d.contributions.loc["g1"].sum() == pytest.approx(0.0)
        assert d.contributions.loc["g3"].sum() == pytest.approx(0.0)

    def make_indication(self):
        return rm.RateIndication(
            experience_loss_cost=series(472.5, 497.4, 448.1),
            manual_loss_cost=series(470.5, 439.1, 440.6),
            credibility=series(0.87, 1.0, 0.49),
            current_rate=series(545.0, 560.0, 530.0),
            current_premium=series(5.2e6, 8.0e6, 3.1e6),
            exposure=series(9600.0, 14400.0, 6000.0),
            trend_total_factor=1.07 ** 1.5,
        )

    def test_indication_matches_loop(self):
        ind = self.make_indication()
        refs = [
            rm.RateIndication(
                float(ind.experience_loss_cost[g]), float(ind.manual_loss_cost[g]),
                float(ind.credibility[g]), float(ind.current_rate[g]),
                current_premium=float(ind.current_premium[g]),
                exposure=float(ind.exposure[g]),
                trend_total_factor=ind.trend_total_factor,
            )
            for g in IDX
        ]
        for meth in ("blended_loss_cost", "indicated_rate", "indicated_rate_change",
                     "experience_loss_ratio", "loss_ratio_indication"):
            assert_matches_loop(getattr(ind, meth)(), [getattr(r, meth)() for r in refs])

    def test_indication_decomposition_reconciles_per_case(self):
        d = self.make_indication().rate_change_decomposition()
        total = np.asarray(d.total_factor)
        np.testing.assert_allclose(d.contributions.sum(axis=1), total - 1.0, rtol=1e-9)


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
class TestScenarios:
    RET = rm.RetentionLoad(12.0, 0.11, 0.03, 0.02)

    def make_book(self):
        return rm.PricingEvaluation(
            loss_cost=series(472.5, 497.4, 448.1),
            current_rate=series(545.0, 560.0, 530.0),
            retention=self.RET,
            exposure=series(9600.0, 14400.0, 6000.0),
            persistency=series(0.90, 0.95, 0.80),
        )

    def make_cases(self):
        b = self.make_book()
        return {
            g: rm.PricingEvaluation(
                float(b.loss_cost[g]), float(b.current_rate[g]), self.RET,
                float(b.exposure[g]), float(b.persistency[g]),
            )
            for g in IDX
        }

    def test_at_matches_loop_every_field(self):
        out = self.make_book().at(0.05, name="issued")
        refs = [c.at(0.05, name="issued") for c in self.make_cases().values()]
        for fld in ("premium_rate", "loss_ratio", "gross_margin_rate", "margin_rate",
                    "margin_ratio", "premium", "margin", "expected_premium",
                    "expected_margin"):
            assert_matches_loop(getattr(out, fld), [getattr(r, fld) for r in refs])

    def test_outcome_to_frame(self):
        f = self.make_book().at(0.05, name="issued").to_frame()
        assert list(f.index) == list(IDX)
        assert f["scenario"].eq("issued").all()
        scalar_f = self.make_cases()["g1"].at(0.05).to_frame()
        assert len(scalar_f) == 1

    def test_inverse_solves_vectorized(self):
        book, cases = self.make_book(), self.make_cases()
        assert_matches_loop(book.zero_margin_rate_change(),
                            [c.zero_margin_rate_change() for c in cases.values()])
        assert_matches_loop(book.premium_for_margin(0.03),
                            [c.premium_for_margin(0.03) for c in cases.values()])

    def test_scenario_frame_vector_matches_mapping(self):
        actions = {"formula": {"g1": 0.062, "g2": 0.045, "g3": 0.083}, "issued": 0.05}
        via_map = rm.scenario_frame(self.make_cases(), actions).sort_values(
            ["scenario", "case"]).reset_index(drop=True)
        via_vec = rm.scenario_frame(self.make_book(), actions).sort_values(
            ["scenario", "case"]).reset_index(drop=True)
        shared = [c for c in via_map.columns if c in via_vec.columns]
        pd.testing.assert_frame_equal(
            via_map[shared], via_vec[shared], check_dtype=False
        )

    def test_scenario_frame_missing_case_raises(self):
        with pytest.raises(KeyError, match="g3"):
            rm.scenario_frame(self.make_book(), {"partial": {"g1": 0.01, "g2": 0.02}})

    def test_uplift_vector_matches_mapping(self):
        for mode in ("multiplicative", "additive"):
            u_map = rm.uplift_for_target_margin(self.make_cases(), 0.01, 0.03, mode=mode)
            u_vec = rm.uplift_for_target_margin(self.make_book(), 0.01, 0.03, mode=mode)
            assert u_vec == pytest.approx(u_map, rel=1e-12)

    def test_margin_ratio_at_indication_equals_profit_margin(self):
        book = self.make_book()
        change = book.rate_change_for_margin(self.RET.profit_margin)
        out = book.at(change)
        np.testing.assert_allclose(out.margin_ratio, self.RET.profit_margin, rtol=1e-9)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
class TestEvaluation:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.n = 400
        self.seg = np.repeat(["A", "B"], self.n // 2)
        self.pred = rng.gamma(2, 200, self.n)
        self.act = self.pred * rng.lognormal(0, 0.4, self.n)
        self.w = rng.uniform(0.5, 2, self.n)

    def test_gini_by_matches_per_group(self):
        out = rm.gini_coefficient(self.act, self.pred, self.w, by=self.seg)
        for g in ("A", "B"):
            m = self.seg == g
            assert out[g] == pytest.approx(
                rm.gini_coefficient(self.act[m], self.pred[m], self.w[m])
            )

    def test_lift_table_by_multiindex(self):
        out = rm.lift_table(self.act, self.pred, self.w, n_bands=4, by=self.seg)
        assert out.index.names == ["group", "band"]
        m = self.seg == "A"
        one = rm.lift_table(self.act[m], self.pred[m], self.w[m], n_bands=4)
        pd.testing.assert_frame_equal(out.loc["A"], one)


# --------------------------------------------------------------------------- #
# relativity
# --------------------------------------------------------------------------- #
class TestFactorTable:
    def test_apply_preserves_series_index(self):
        ft = rm.FactorTable("area", {"north": 1.1, "south": 0.9})
        levels = pd.Series(["north", "west", "south"], index=IDX)
        out = ft.apply(levels)
        assert isinstance(out, pd.Series) and out.index.equals(IDX)
        assert out.tolist() == [1.1, 1.0, 0.9]
        arr = ft.apply(["north", "south"])
        assert isinstance(arr, np.ndarray)


# --------------------------------------------------------------------------- #
# end-to-end: DataFrame in -> DataFrame out
# --------------------------------------------------------------------------- #
def test_full_pipeline_dataframe_in_dataframe_out():
    book = pd.DataFrame(
        {
            "claims": [4_128_000.0, 6_048_000.0, 2_760_000.0],
            "exposure": [9_600.0, 14_400.0, 6_000.0],
            "pooled_x": [310_000.0, 0.0, 505_000.0],
            "base": [420.0, 435.0, 410.0],
            "area": [1.05, 0.98, 1.12],
            "industry": [1.10, 1.00, 0.95],
            "n_claims": [820.0, 1450.0, 260.0],
            "current": [545.0, 560.0, 530.0],
        },
        index=IDX,
    )
    exp = rm.ExperienceRate(book["claims"], book["exposure"],
                            pooled_excess=book["pooled_x"],
                            trend_annual=0.07, trend_years=1.5, pooling_charge=28.0)
    man = rm.ManualRate(book["base"], {"area": book["area"], "industry": book["industry"]})
    z = rm.limited_fluctuation_credibility(book["n_claims"], 1082.0)
    ind = rm.RateIndication(
        experience_loss_cost=exp.loss_cost(), manual_loss_cost=man.loss_cost(),
        credibility=z, current_rate=book["current"],
        trend_total_factor=exp.trend_factor(),
    )
    out = book.assign(
        experience_lc=exp.loss_cost(), manual_lc=man.loss_cost(), Z=z,
        blended_lc=ind.blended_loss_cost(), change=ind.indicated_rate_change(),
    )
    action = rm.renew(out["current"], ind.indicated_rate(), cap=0.10, floor=0.0)
    out["proposed"] = action.proposed_rate
    assert out.index.equals(IDX)
    for g in IDX:
        one = rm.RateIndication(
            float(out.loc[g, "experience_lc"]), float(out.loc[g, "manual_lc"]),
            float(out.loc[g, "Z"]), float(out.loc[g, "current"]),
            trend_total_factor=float(exp.trend_factor()),
        )
        assert out.loc[g, "change"] == pytest.approx(one.indicated_rate_change())


def test_plain_lists_accepted_everywhere():
    """Plain Python lists satisfy the contract (coerced to ndarray)."""
    L = [0.05, 0.06, 0.07]
    P = [500.0, 480.0, 610.0]
    assert isinstance(rm.trend_factor(L, 1.5), np.ndarray)
    assert isinstance(rm.apply_trend(P, L, 1.0), np.ndarray)
    assert isinstance(rm.combine_trend(L, 0.02), np.ndarray)
    assert isinstance(rm.split_total_trend(0.10, L), np.ndarray)
    np.testing.assert_allclose(
        rm.combine_trend(L, 0.02),
        [rm.combine_trend(x, 0.02) for x in L], rtol=1e-12,
    )
