r"""Trend: bringing historical experience to the rating-period cost level.

The trend factor compounds an annual rate over the gap between the midpoint of
the experience period and the midpoint of the rating period:

.. math::
    \text{factor} = (1 + t)^{\Delta}, \qquad
    \Delta = \frac{m_{\text{rate}} - m_{\text{exp}}}{365.25}\ \text{years}.

A total trend is often decomposed into utilization and unit-cost components,
which combine multiplicatively: :math:`(1+t) = (1+t_u)(1+t_c)`.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Union

DateLike = Union[str, date, datetime]


def _to_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.fromisoformat(str(d)).date()


def years_between(start: DateLike, end: DateLike) -> float:
    """Fractional years between two dates using a 365.25-day year."""
    s, e = _to_date(start), _to_date(end)
    return (e - s).days / 365.25


def period_midpoint(start: DateLike, end: DateLike) -> date:
    """Midpoint date of a period ``[start, end]`` (inclusive endpoints)."""
    s, e = _to_date(start), _to_date(end)
    if e < s:
        raise ValueError("end must not precede start")
    return s + (e - s) / 2


def trend_factor(annual_trend: float, years: float) -> float:
    r""":math:`(1 + \text{annual\_trend})^{\text{years}}`."""
    base = 1.0 + annual_trend
    if base <= 0:
        raise ValueError("annual_trend must exceed -1")
    return float(base**years)


def trend_factor_between(
    annual_trend: float,
    experience_period: tuple[DateLike, DateLike],
    rating_period: tuple[DateLike, DateLike],
) -> float:
    """Midpoint-to-midpoint trend factor from two date ranges."""
    m_exp = period_midpoint(*experience_period)
    m_rate = period_midpoint(*rating_period)
    return trend_factor(annual_trend, years_between(m_exp, m_rate))


def apply_trend(value: float, annual_trend: float, years: float) -> float:
    """Trend a value forward (or back, for negative ``years``)."""
    return float(value) * trend_factor(annual_trend, years)


def combine_trend(util_trend: float, cost_trend: float) -> float:
    r"""Combine utilization and unit-cost trends: :math:`(1+t_u)(1+t_c)-1`."""
    return float((1 + util_trend) * (1 + cost_trend) - 1)


def split_total_trend(total_trend: float, util_trend: float) -> float:
    """Back out the unit-cost trend implied by a total and a utilization trend."""
    return float((1 + total_trend) / (1 + util_trend) - 1)
