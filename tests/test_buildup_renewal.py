"""Build-up arithmetic, renewal rollups, and the small rating helpers."""
import pandas as pd
import pytest

from ratingmodels import BuildUp, apply_trend, average_relativity, blend, cap_change, unit_level_renewal


def test_buildup_value_and_subtotal():
    result = (BuildUp()
              .start("base loss cost", 100.0)
              .multiply("trend", 1.10)
              .add("pooling charge", 5.0)
              .checkpoint("net claim cost")
              .multiply("lae", 1.05)
              .evaluate())
    assert result.value == pytest.approx((100.0 * 1.10 + 5.0) * 1.05, rel=1e-12)
    assert result.subtotal("net claim cost") == pytest.approx(115.0, rel=1e-12)
    assert len(result.to_frame()) == len(result.steps)


def test_unit_level_renewal_multiplies_factors_and_counts():
    census = pd.DataFrame({"age_f": [1.1, 0.9], "area_f": [1.0, 1.2], "count": [3, 5]})
    out = unit_level_renewal(census, base_rate=100.0, factor_cols=["age_f", "area_f"])
    assert out["unit_rate"].tolist() == pytest.approx([110.0, 108.0], rel=1e-12)
    assert out["premium"].tolist() == pytest.approx([330.0, 540.0], rel=1e-12)


def test_average_relativity_is_the_exposure_weighted_mean():
    df = pd.DataFrame({"exposure": [3.0, 5.0], "age_f": [1.1, 0.9], "area_f": [1.0, 1.2]})
    got = average_relativity(df, exposure="exposure", factor_cols=["age_f", "area_f"])
    assert got == pytest.approx((3 * 1.1 * 1.0 + 5 * 0.9 * 1.2) / 8.0, rel=1e-12)


def test_small_helpers():
    assert blend(0.9, 0.5, 0.25) == pytest.approx(0.25 * 0.9 + 0.75 * 0.5, rel=1e-12)
    assert apply_trend(100.0, 0.05, 3.0) == pytest.approx(100.0 * 1.05**3, rel=1e-12)
    assert cap_change(0.20, cap=0.10) == 0.10
    assert cap_change(-0.20, cap=0.10, floor=-0.05) == -0.05
    assert cap_change(0.03, cap=0.10, floor=-0.05) == 0.03
