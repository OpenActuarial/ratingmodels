r"""Trend: bringing historical experience to the rating-period cost level.

The trend factor compounds an annual rate over the gap between the midpoint of
the experience period and the midpoint of the rating period:

.. math::
    \text{factor} = (1 + t)^{\Delta}, \qquad
    \Delta = \frac{m_{\text{rate}} - m_{\text{exp}}}{365.25}\ \text{years}.

A total trend is often decomposed into frequency and severity components,
which combine multiplicatively: :math:`(1+t) = (1+t_f)(1+t_s)` (a health
shop's utilization / unit-cost split is the same identity under its own
labels).

All numeric arguments follow the ratingmodels vectorization contract: scalar
in gives float out; a Series or array of trends / years / values gives a
Series or array out, elementwise, with scalars broadcasting. Date arguments
likewise accept datetime-like Series for per-row periods.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Union

import numpy as np
import pandas as pd

from ._utils import Numeric, as_numeric, common_index, maybe_float

DateLike = Union[str, date, datetime, pd.Series]


def _to_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.fromisoformat(str(d)).date()


def _is_datelike_vector(d) -> bool:
    return isinstance(d, (pd.Series, pd.DatetimeIndex, np.ndarray)) and np.ndim(d) > 0


def years_between(start: DateLike, end: DateLike) -> Numeric:
    """Fractional years between two dates using a 365.25-day year.

    Accepts scalar dates (returns float) or datetime-like Series/arrays
    (returns a Series/array of year gaps, elementwise; scalars broadcast).
    """
    if _is_datelike_vector(start) or _is_datelike_vector(end):
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        days = (e - s) / pd.Timedelta(days=1)
        return days / 365.25
    s, e = _to_date(start), _to_date(end)
    return (e - s).days / 365.25


def period_midpoint(start: DateLike, end: DateLike):
    """Midpoint date of a period ``[start, end]`` (inclusive endpoints).

    Vectorized over datetime-like Series/arrays (returns Timestamps).
    """
    if _is_datelike_vector(start) or _is_datelike_vector(end):
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        if np.any(np.asarray(e < s)):
            raise ValueError("end must not precede start")
        return s + (e - s) / 2
    s, e = _to_date(start), _to_date(end)
    if e < s:
        raise ValueError("end must not precede start")
    return s + (e - s) / 2


def trend_factor(annual_trend: Numeric, years: Numeric) -> Numeric:
    r""":math:`(1 + \text{annual\_trend})^{\text{years}}`, elementwise."""
    common_index([annual_trend, years])
    annual_trend = as_numeric(annual_trend, "annual_trend")
    years = as_numeric(years, "years")
    base = 1.0 + annual_trend
    if np.any(np.asarray(base) <= 0):
        raise ValueError("annual_trend must exceed -1")
    return maybe_float(base**years)


def trend_factor_between(
    annual_trend: Numeric,
    experience_period: tuple[DateLike, DateLike],
    rating_period: tuple[DateLike, DateLike],
) -> Numeric:
    """Midpoint-to-midpoint trend factor from two date ranges."""
    m_exp = period_midpoint(*experience_period)
    m_rate = period_midpoint(*rating_period)
    return trend_factor(annual_trend, years_between(m_exp, m_rate))


def apply_trend(value: Numeric, annual_trend: Numeric, years: Numeric) -> Numeric:
    """Trend a value forward (or back, for negative ``years``), elementwise."""
    common_index([value, annual_trend, years])
    value = as_numeric(value, "value")
    return maybe_float(value * trend_factor(annual_trend, years))


def combine_trend(frequency_trend: Numeric, severity_trend: Numeric) -> Numeric:
    r"""Combine frequency and severity trends: :math:`(1+t_f)(1+t_s)-1`."""
    common_index([frequency_trend, severity_trend])
    frequency_trend = as_numeric(frequency_trend, "frequency_trend")
    severity_trend = as_numeric(severity_trend, "severity_trend")
    return maybe_float((1 + frequency_trend) * (1 + severity_trend) - 1)


def split_total_trend(total_trend: Numeric, frequency_trend: Numeric) -> Numeric:
    """Back out the severity trend implied by a total and a frequency trend."""
    common_index([total_trend, frequency_trend])
    total_trend = as_numeric(total_trend, "total_trend")
    frequency_trend = as_numeric(frequency_trend, "frequency_trend")
    return maybe_float((1 + total_trend) / (1 + frequency_trend) - 1)
