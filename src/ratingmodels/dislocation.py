r"""Rate dislocation: the distribution of rate changes across a book.

An average rate change hides everything that matters operationally -- who
takes a large increase, how much premium sits in each band, and what the
constraints (caps, floors, concessions) cost against the indication.
:func:`rate_dislocation` bands the book by rate change;
:func:`constraint_impact` quantifies the gap between indicated and proposed
rates. Both are pure comparisons of rate vectors, so they work with any
source of "current" and "proposed" -- a renewal run
(:func:`ratingmodels.renew`), a re-rated plan, or scenario output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rate_dislocation", "constraint_impact"]


def _rate_arrays(*rates, exposure=None):
    out = []
    for r in rates:
        a = np.asarray(r, dtype=float)
        if a.ndim != 1:
            raise ValueError("rates must be 1-D")
        out.append(a)
    n = out[0].size
    if n == 0:
        raise ValueError("inputs must not be empty")
    if any(a.shape != out[0].shape for a in out):
        raise ValueError("rate inputs must have equal length")
    if exposure is None:
        w = np.ones(n)
    else:
        w = np.asarray(exposure, dtype=float)
        if w.shape != out[0].shape:
            raise ValueError("exposure must match rates in length")
        if np.any(w < 0):
            raise ValueError("exposure must be nonnegative")
    if not all(np.all(np.isfinite(a)) for a in (*out, w)):
        raise ValueError("inputs must be finite")
    return (*out, w)


def _fmt_pct(x: float) -> str:
    pct = 100.0 * x
    s = f"{pct:+.1f}".rstrip("0").rstrip(".")
    return f"{s}%"


def rate_dislocation(
    current_rate,
    proposed_rate,
    exposure=None,
    bands=(-0.10, -0.05, 0.0, 0.05, 0.10),
    include_total: bool = True,
) -> pd.DataFrame:
    """Band the book by rate change and report premium in each band.

    Parameters
    ----------
    current_rate, proposed_rate : array-like
        Per-case rates; the change is ``proposed/current - 1``.
    exposure : array-like, optional
        Units each rate applies to, so ``rate * exposure`` is premium.
        Premium equals rate (and counts weight equally) when omitted.
    bands : sequence of float
        Interior band edges as decimal changes, e.g. ``-0.05`` for -5%.
        Edges are extended with ``-inf``/``+inf``, so ``k`` edges give
        ``k + 1`` bands; a band's interval is half-open, ``(low, high]``.
        The default edges include 0.0, so increases and decreases are
        always separated.
    include_total : bool
        Append an ``"All"`` row. Default True.

    Returns
    -------
    pandas.DataFrame
        One row per band (empty bands kept, so the exhibit shape is stable)
        with columns ``n``, ``exposure``, ``current_premium``,
        ``proposed_premium``, ``avg_change`` (premium-weighted:
        proposed/current - 1), ``exposure_share``.
    """
    cur, prop, w = _rate_arrays(current_rate, proposed_rate, exposure=exposure)
    if np.any(cur <= 0):
        raise ValueError("current_rate must be positive to define a change")
    edges = np.asarray(sorted(bands), dtype=float)
    if edges.size == 0:
        raise ValueError("bands must contain at least one edge")
    if np.unique(edges).size != edges.size:
        raise ValueError("band edges must be distinct")

    change = prop / cur - 1.0
    # a change of exactly -5% computed as 95/100 - 1 lands a few ULPs off the
    # edge; snap within floating-point tolerance so boundary cases band
    # deterministically into the lower band, (low, high]
    for e in edges:
        near = np.isclose(change, e, rtol=1e-9, atol=1e-12)
        if near.any():
            change = np.where(near, e, change)
    labels = (
        [f"below {_fmt_pct(edges[0])}"]
        + [f"{_fmt_pct(lo)} to {_fmt_pct(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]
        + [f"above {_fmt_pct(edges[-1])}"]
    )
    band_ix = np.searchsorted(edges, change, side="left")  # (low, high] bands

    cur_prem = cur * w
    prop_prem = prop * w
    total_w = w.sum()

    rows = []
    for b, label in enumerate(labels):
        m = band_ix == b
        cw = float(cur_prem[m].sum())
        pw = float(prop_prem[m].sum())
        rows.append(
            {
                "band": label,
                "n": int(m.sum()),
                "exposure": float(w[m].sum()),
                "current_premium": cw,
                "proposed_premium": pw,
                "avg_change": pw / cw - 1.0 if cw > 0 else np.nan,
                "exposure_share": float(w[m].sum() / total_w) if total_w > 0 else np.nan,
            }
        )
    if include_total:
        cw, pw = float(cur_prem.sum()), float(prop_prem.sum())
        rows.append(
            {
                "band": "All",
                "n": int(len(change)),
                "exposure": float(total_w),
                "current_premium": cw,
                "proposed_premium": pw,
                "avg_change": pw / cw - 1.0 if cw > 0 else np.nan,
                "exposure_share": 1.0 if total_w > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("band")


def constraint_impact(
    indicated_rate,
    proposed_rate,
    exposure=None,
    current_rate=None,
    by=None,
) -> "pd.Series | pd.DataFrame":
    """What the gap between indicated and proposed rates costs.

    Caps, floors, and concessions move issued rates off the indication; this
    quantifies the move in premium terms -- the shortfall left on the table
    where proposed sits below indicated, the excess where it sits above, and
    the further average change still needed to reach the indication.

    Parameters
    ----------
    indicated_rate, proposed_rate : array-like
        The formula answer and the rate actually proposed/issued.
    exposure : array-like, optional
        Units per case; premium is ``rate * exposure``. Omitted = 1 per case.
    current_rate : array-like, optional
        When given, ``indicated_change`` and ``realized_change`` (both
        premium-weighted against current) are also reported.
    by : array-like, optional
        Group labels; returns one row per group (a DataFrame) instead of a
        Series -- which segments absorbed the capping is usually the
        actionable question.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Metrics: ``n``, ``exposure``, ``n_below`` / ``exposure_below`` /
        ``premium_shortfall`` (proposed < indicated), ``n_above`` /
        ``exposure_above`` / ``premium_excess`` (proposed > indicated),
        ``indicated_premium``, ``proposed_premium``, ``remaining_change``
        (indicated/proposed - 1, the future rate action still owed), and --
        with ``current_rate`` -- ``indicated_change`` and ``realized_change``.
    """
    if current_rate is None:
        ind, prop, w = _rate_arrays(indicated_rate, proposed_rate, exposure=exposure)
        cur = None
    else:
        ind, prop, cur, w = _rate_arrays(
            indicated_rate, proposed_rate, current_rate, exposure=exposure
        )

    if by is not None:
        keys = np.asarray(by)
        if keys.shape != ind.shape:
            raise ValueError("by must match rates in length")
        rows = {}
        for lvl in pd.unique(keys):
            m = keys == lvl
            rows[lvl] = constraint_impact(
                ind[m], prop[m],
                exposure=w[m],
                current_rate=None if cur is None else cur[m],
            )
        out = pd.DataFrame(rows).T.sort_index(kind="stable")
        out.index.name = "group"
        for col in ("n", "n_below", "n_above"):
            out[col] = out[col].astype(int)
        return out

    ind_prem = ind * w
    prop_prem = prop * w
    below = prop < ind
    above = prop > ind

    metrics = {
        "n": int(ind.size),
        "exposure": float(w.sum()),
        "n_below": int(below.sum()),
        "exposure_below": float(w[below].sum()),
        "premium_shortfall": float(((ind - prop) * w)[below].sum()),
        "n_above": int(above.sum()),
        "exposure_above": float(w[above].sum()),
        "premium_excess": float(((prop - ind) * w)[above].sum()),
        "indicated_premium": float(ind_prem.sum()),
        "proposed_premium": float(prop_prem.sum()),
        "remaining_change": (
            float(ind_prem.sum() / prop_prem.sum() - 1.0) if prop_prem.sum() > 0 else np.nan
        ),
    }
    if cur is not None:
        if np.any(cur <= 0):
            raise ValueError("current_rate must be positive to define a change")
        cur_prem = float((cur * w).sum())
        metrics["indicated_change"] = ind_prem.sum() / cur_prem - 1.0 if cur_prem > 0 else np.nan
        metrics["realized_change"] = prop_prem.sum() / cur_prem - 1.0 if cur_prem > 0 else np.nan
    return pd.Series(metrics, name="constraint_impact")
