r"""On-level factors: restate historical premium at current rate level.

A rate indication compares losses to premium, but historical premium was
earned at historical rates. The on-level factor for an experience period is

.. math::
    \text{OLF} = \frac{\text{current rate index}}
                       {\text{average rate index earned in the period}},

so that ``premium x OLF`` is what the same exposure would have produced at
today's rates. The average earned index is the *parallelogram method* made
exact: with policies of term :math:`T` written uniformly, the rate index
earned at calendar time :math:`t` is the average of the written-rate index
over the writing window :math:`[t-T, t]`, and the period average integrates
that. Both integrals are piecewise polynomial and are computed exactly --
no simulation, no discretization error.

Dates may be floats (in years, or any consistent unit shared with
``policy_term``) or anything :class:`pandas.Timestamp` accepts, converted
internally at 365.25 days per year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["on_level_factors"]

_DAYS_PER_YEAR = 365.25


def _to_float_time(value, anchor):
    ts = pd.Timestamp(value)
    return (ts - anchor).days / _DAYS_PER_YEAR + (ts - anchor).seconds / (
        _DAYS_PER_YEAR * 86400.0
    )


def _coerce_times(periods, change_dates, current_date):
    """Everything to float time on one axis; datetimes via an anchor."""
    flat = [p[0] for p in periods] + [p[1] for p in periods] + list(change_dates)
    if current_date is not None:
        flat.append(current_date)
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in flat):
        conv = float
    else:
        try:
            anchor = min(pd.Timestamp(v) for v in flat)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "dates must be all numeric or all datetime-like"
            ) from exc

        def conv(v):
            return _to_float_time(v, anchor)

    per = [(conv(a), conv(b)) for a, b in periods]
    ch = [conv(d) for d in change_dates]
    cur = conv(current_date) if current_date is not None else None
    return per, ch, cur


class _RateIndex:
    """Written-rate index: a step function with exact integrals."""

    def __init__(self, dates, changes):
        order = np.argsort(dates, kind="stable")
        self.dates = np.asarray(dates, dtype=float)[order]
        factors = 1.0 + np.asarray(changes, dtype=float)[order]
        if np.any(factors <= 0):
            raise ValueError("rate changes must be greater than -100%")
        # values[j] applies on (dates[j-1], dates[j]] ... i.e. segment j is
        # [dates[j-1], dates[j]) with value values[j]; values[0] before all
        self.values = np.concatenate([[1.0], np.cumprod(factors)])

    def level(self, t):
        """I(t): the written-rate index at time t (right-continuous)."""
        idx = np.searchsorted(self.dates, np.asarray(t, dtype=float), side="right")
        return self.values[idx]

    def integral(self, t):
        r"""":math:`\int_a^t I(s)\,ds` from a fixed anchor (piecewise linear)."""
        t = np.asarray(t, dtype=float)
        if self.dates.size == 0:
            return t
        seg_cum = np.concatenate(
            [[0.0], np.cumsum(self.values[:-1][1:] * np.diff(self.dates))]
        ) if self.dates.size > 1 else np.array([0.0])
        # cumulative integral evaluated at each change date, anchored at dates[0]
        idx = np.searchsorted(self.dates, t, side="right")
        base = np.where(idx == 0, (t - self.dates[0]) * self.values[0], 0.0)
        inner = np.where(
            idx > 0,
            seg_cum[np.maximum(idx - 1, 0)]
            + (t - self.dates[np.maximum(idx - 1, 0)]) * self.values[idx],
            0.0,
        )
        return np.where(idx == 0, base, inner)

    def earned(self, t, term):
        """E(t): index earned at t, averaging writes over [t - term, t]."""
        if term == 0:
            return self.level(t)
        t = np.asarray(t, dtype=float)
        return (self.integral(t) - self.integral(t - term)) / term

    @property
    def current(self):
        return float(self.values[-1])


def on_level_factors(
    periods,
    rate_changes,
    policy_term: float = 1.0,
    current_date=None,
) -> pd.DataFrame:
    """On-level factors per experience period, by the exact parallelogram.

    Parameters
    ----------
    periods : sequence of (start, end)
        Experience periods, ``start < end``. Floats or datetime-likes.
    rate_changes : sequence of (date, change)
        Rate-change history as decimal changes (``0.08`` for +8%). Multiple
        changes on one date compound. May be empty (all factors 1.0).
    policy_term : float
        Policy term in the same units as float dates (years for datetime
        input). ``0`` means premium is earned the instant it is written --
        the in-force approximation; ``1.0`` is the classic annual-policy
        parallelogram.
    current_date : optional
        The "as of" for the current rate level. Default: after every listed
        change (the full cumulative index). Pass a date to exclude changes
        after it.

    Returns
    -------
    pandas.DataFrame
        One row per period: ``period_start``, ``period_end``,
        ``average_earned_index``, ``current_index``, ``on_level_factor``.

    Examples
    --------
    The textbook case -- one +10% change mid-year, annual policies, calendar
    year period: an eighth of the earned premium sits in the new-rate
    triangle, so the average index is 1.0125 and the factor 1.1/1.0125.
    """
    periods = list(periods)
    if not periods:
        raise ValueError("at least one period is required")
    changes = [(d, c) for d, c in rate_changes]
    if policy_term < 0:
        raise ValueError("policy_term must be nonnegative")
    per, dates, cur_t = _coerce_times(
        periods, [d for d, _ in changes], current_date
    )
    for a, b in per:
        if not b > a:
            raise ValueError("each period must have end > start")

    index = _RateIndex(dates, [c for _, c in changes])
    if cur_t is None:
        current = index.current
    else:
        current = float(index.level(cur_t))

    rows = []
    for (a, b), (raw_a, raw_b) in zip(per, periods):
        breaks = {a, b}
        for d in index.dates:
            for point in (d, d + policy_term):
                if a < point < b:
                    breaks.add(float(point))
        grid = np.array(sorted(breaks))
        if policy_term == 0:
            # E is the step function itself: integrate exactly panel by
            # panel (level is constant on [g_i, g_{i+1}) -- every change
            # date is a breakpoint), where trapezoid would smear the jump
            avg = float(
                np.sum(index.level(grid[:-1]) * np.diff(grid)) / (b - a)
            )
        else:
            # E is continuous piecewise-linear with these breakpoints, so
            # the trapezoid rule is exact
            e = index.earned(grid, policy_term)
            trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy<2
            avg = float(trapz(e, grid) / (b - a))
        rows.append(
            {
                "period_start": raw_a,
                "period_end": raw_b,
                "average_earned_index": avg,
                "current_index": current,
                "on_level_factor": current / avg,
            }
        )
    return pd.DataFrame(rows)
