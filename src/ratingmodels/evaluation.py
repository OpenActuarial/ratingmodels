r"""Pricing-model evaluation: lift tables and the ordered-Lorenz Gini.

These diagnostics measure a model's ability to *segment* risk -- to order
policies from best to worst -- which is the property a rating plan monetizes.
Both are exposure-weighted throughout.

Gini here is the pricing convention (Frees, Meyers & Cummings): policies are
sorted by predicted risk, the Lorenz curve plots cumulative share of exposure
against cumulative share of actual losses, and the Gini coefficient is twice
the area between the curve and the diagonal. ``normalize=True`` divides by the
Gini of a hypothetical perfect model (one that sorts by actual outcome), giving
a 0-to-1 scale comparable across books.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["gini_coefficient", "lift_table"]


def _as_arrays(actual, predicted, exposure):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape or a.ndim != 1:
        raise ValueError("actual and predicted must be 1D arrays of equal length")
    if a.size == 0:
        raise ValueError("inputs must not be empty")
    if exposure is None:
        w = np.ones_like(a)
    else:
        w = np.asarray(exposure, dtype=float)
        if w.shape != a.shape:
            raise ValueError("exposure must match actual/predicted in length")
        if np.any(w < 0):
            raise ValueError("exposure must be nonnegative")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(p)) and np.all(np.isfinite(w))):
        raise ValueError("inputs must be finite")
    return a, p, w


def _lorenz_gini(order_key, actual, exposure):
    """Gini from the Lorenz curve of actual losses ordered by ``order_key``."""
    idx = np.argsort(order_key, kind="stable")
    w = exposure[idx]
    loss = actual[idx]
    total_w = w.sum()
    total_loss = loss.sum()
    if total_w <= 0:
        raise ValueError("total exposure must be positive")
    if total_loss <= 0:
        return 0.0
    cum_w = np.concatenate([[0.0], np.cumsum(w)]) / total_w
    cum_loss = np.concatenate([[0.0], np.cumsum(loss)]) / total_loss
    # area under the Lorenz curve by trapezoid; Gini = 1 - 2 * area
    area = np.trapezoid(cum_loss, cum_w)
    return float(1.0 - 2.0 * area)


def gini_coefficient(actual, predicted, exposure=None, normalize: bool = True) -> float:
    """Ordered-Lorenz Gini of ``predicted`` as a risk ranker for ``actual``.

    Parameters
    ----------
    actual : array-like
        Observed outcome per record (losses, claim counts, pure premium).
    predicted : array-like
        Model prediction used to order records from lowest to highest risk.
    exposure : array-like, optional
        Weights (earned exposure). Equal weights if omitted.
    normalize : bool
        If True (default), divide by the Gini of the perfect model that sorts
        by ``actual`` itself, so 1.0 means perfect segmentation and 0.0 means
        no segmentation. If False, return the raw ordered-Lorenz Gini.
    """
    a, p, w = _as_arrays(actual, predicted, exposure)
    g = _lorenz_gini(p, a, w)
    if not normalize:
        return g
    g_perfect = _lorenz_gini(a, a, w)
    if g_perfect <= 0:
        return 0.0
    return float(g / g_perfect)


def lift_table(
    actual,
    predicted,
    exposure=None,
    n_bands: int = 10,
) -> pd.DataFrame:
    """Exposure-weighted lift table: records banded by predicted risk.

    Records are sorted by ``predicted`` and split into ``n_bands`` bands of
    (approximately) equal total exposure. Within each band the table reports
    exposure, the exposure-weighted actual and predicted means, and ``lift`` --
    the band's actual mean relative to the overall actual mean. A model that
    segments well shows lift rising monotonically across bands.

    Returns
    -------
    pandas.DataFrame
        Indexed 1..n_bands with columns ``n``, ``exposure``,
        ``predicted_mean``, ``actual_mean``, ``lift``.
    """
    a, p, w = _as_arrays(actual, predicted, exposure)
    if n_bands < 2:
        raise ValueError("n_bands must be at least 2")
    idx = np.argsort(p, kind="stable")
    a, p, w = a[idx], p[idx], w[idx]
    total_w = w.sum()
    if total_w <= 0:
        raise ValueError("total exposure must be positive")

    # band edges at equal cumulative exposure
    cum_w = np.cumsum(w)
    band = np.minimum((cum_w / total_w * n_bands - 1e-12).astype(int), n_bands - 1)

    overall_actual = a.sum() / total_w
    rows = []
    for b in range(n_bands):
        m = band == b
        wb = w[m].sum()
        if wb <= 0:
            rows.append((b + 1, int(m.sum()), 0.0, np.nan, np.nan, np.nan))
            continue
        actual_mean = a[m].sum() / wb
        predicted_mean = float(np.sum(p[m] * w[m]) / wb)
        lift = actual_mean / overall_actual if overall_actual > 0 else np.nan
        rows.append((b + 1, int(m.sum()), float(wb), predicted_mean, float(actual_mean), float(lift)))
    out = pd.DataFrame(
        rows, columns=["band", "n", "exposure", "predicted_mean", "actual_mean", "lift"]
    ).set_index("band")
    return out
