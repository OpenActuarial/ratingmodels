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

from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "gini_coefficient",
    "lift_table",
    "calibration_table",
    "actual_expected_table",
    "compare_models",
]


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
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy<2
    area = trapz(cum_loss, cum_w)
    return float(1.0 - 2.0 * area)


def gini_coefficient(
    actual, predicted, exposure=None, normalize: bool = True, by=None
) -> "float | pd.Series":
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
    by : array-like, optional
        Group labels aligned with ``actual``. When given, the Gini is
        computed within each group and a Series indexed by group is
        returned -- one call scores every segment of a validation frame.
    """
    if by is not None:
        return _grouped(
            by, actual, predicted, exposure,
            lambda a, p, w: gini_coefficient(a, p, w, normalize),
        ).rename("gini")
    a, p, w = _as_arrays(actual, predicted, exposure)
    g = _lorenz_gini(p, a, w)
    if not normalize:
        return g
    g_perfect = _lorenz_gini(a, a, w)
    if g_perfect <= 0:
        return 0.0
    return float(g / g_perfect)


def _grouped(by, actual, predicted, exposure, fn) -> pd.Series:
    """Apply ``fn(actual, predicted, exposure)`` within each group of ``by``."""
    a, p, w = _as_arrays(actual, predicted, exposure)
    keys = np.asarray(by)
    if keys.shape != a.shape:
        raise ValueError("by must match actual/predicted in length")
    frame = pd.DataFrame({"a": a, "p": p, "w": w, "g": keys})
    return frame.groupby("g", sort=True).apply(
        lambda d: fn(d["a"].to_numpy(), d["p"].to_numpy(), d["w"].to_numpy()),
        include_groups=False,
    ).rename_axis(index=None)


def lift_table(
    actual,
    predicted,
    exposure=None,
    n_bands: int = 10,
    by=None,
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
        ``predicted_mean``, ``actual_mean``, ``lift``. With ``by`` (group
        labels aligned with ``actual``), one table is built per group and
        the result carries a ``(group, band)`` MultiIndex.
    """
    if by is not None:
        a, p, w = _as_arrays(actual, predicted, exposure)
        keys = np.asarray(by)
        if keys.shape != a.shape:
            raise ValueError("by must match actual/predicted in length")
        frame = pd.DataFrame({"a": a, "p": p, "w": w, "g": keys})
        pieces = {
            g: lift_table(d["a"].to_numpy(), d["p"].to_numpy(), d["w"].to_numpy(), n_bands)
            for g, d in frame.groupby("g", sort=True)
        }
        return pd.concat(pieces, names=["group", "band"])
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


def calibration_table(
    actual,
    predicted,
    exposure=None,
    n_bands: int = 10,
    by=None,
) -> pd.DataFrame:
    """Calibration across the prediction range: actual vs. predicted by band.

    The companion to :func:`lift_table`: lift asks whether predictions *order*
    risks; calibration asks whether they are *right on the level*. Records are
    banded into ``n_bands`` groups of (approximately) equal exposure by
    predicted value, and each band reports per-unit actual and predicted means
    (band totals over band exposure) and their ratio -- so ``actual`` and
    ``predicted`` are treated symmetrically and should both be on the *total*
    scale, as from ``model.predict(df, exposure=...)``. A well-calibrated
    model has ``ae_ratio`` near 1.0 in every band; a systematic drift (low
    bands above 1, high bands below) is the classic signature of over-shrunk
    predictions.

    Returns
    -------
    pandas.DataFrame
        Indexed 1..n_bands with columns ``n``, ``exposure``,
        ``predicted_mean``, ``actual_mean``, ``ae_ratio``. With ``by``
        (group labels aligned with ``actual``), one table per group under a
        ``(group, band)`` MultiIndex.
    """
    if by is not None:
        a, p, w = _as_arrays(actual, predicted, exposure)
        keys = np.asarray(by)
        if keys.shape != a.shape:
            raise ValueError("by must match actual/predicted in length")
        frame = pd.DataFrame({"a": a, "p": p, "w": w, "g": keys})
        pieces = {
            g: calibration_table(d["a"].to_numpy(), d["p"].to_numpy(), d["w"].to_numpy(), n_bands)
            for g, d in frame.groupby("g", sort=True)
        }
        return pd.concat(pieces, names=["group", "band"])
    a, p, w = _as_arrays(actual, predicted, exposure)
    if n_bands < 2:
        raise ValueError("n_bands must be at least 2")
    idx = np.argsort(p, kind="stable")
    a, p, w = a[idx], p[idx], w[idx]
    total_w = w.sum()
    if total_w <= 0:
        raise ValueError("total exposure must be positive")
    cum_w = np.cumsum(w)
    band = np.minimum((cum_w / total_w * n_bands - 1e-12).astype(int), n_bands - 1)

    rows = []
    for b in range(n_bands):
        m = band == b
        wb = w[m].sum()
        if wb <= 0:
            rows.append((b + 1, int(m.sum()), 0.0, np.nan, np.nan, np.nan))
            continue
        psum = float(p[m].sum())
        asum = float(a[m].sum())
        predicted_mean = psum / wb
        actual_mean = asum / wb
        ae = asum / psum if psum > 0 else np.nan
        rows.append((b + 1, int(m.sum()), float(wb), predicted_mean, actual_mean, ae))
    return pd.DataFrame(
        rows, columns=["band", "n", "exposure", "predicted_mean", "actual_mean", "ae_ratio"]
    ).set_index("band")


def actual_expected_table(
    actual,
    expected,
    exposure=None,
    by=None,
    include_total: bool = True,
) -> pd.DataFrame:
    """Actual-to-expected exhibit: totals, means, and A/E ratio by segment.

    The workhorse validation exhibit: for each segment, the total actual,
    total expected, their exposure-weighted means, and the A/E ratio. An A/E
    near 1.0 in every segment of a variable means the model has captured that
    variable's effect; a pattern across levels means residual signal.

    Parameters
    ----------
    actual, expected : array-like
        Observed outcomes and model expectations, row-aligned. ``expected``
        should be on the same total scale as ``actual`` (e.g. include
        exposure), as from ``model.predict(df, exposure=...)``.
    exposure : array-like, optional
        Weights for the mean columns. Row counts when omitted.
    by : array-like, mapping, or DataFrame, optional
        * omitted -- a single overall row.
        * array of labels -- one row per level.
        * mapping/DataFrame of ``name -> labels`` -- one block per variable,
          stacked tidily under a ``(variable, level)`` MultiIndex; one call
          audits every rating variable of a validation frame.
    include_total : bool
        Append an overall row (labelled ``"All"``). Default True.

    Returns
    -------
    pandas.DataFrame
        Columns ``n``, ``exposure``, ``actual``, ``expected``,
        ``actual_mean``, ``expected_mean``, ``ae_ratio``.
    """
    a, e, w = _as_arrays(actual, expected, exposure)

    def _row(mask):
        n = int(mask.sum())
        wsum = float(w[mask].sum())
        asum = float(a[mask].sum())
        esum = float(e[mask].sum())
        return (
            n,
            wsum,
            asum,
            esum,
            asum / wsum if wsum > 0 else np.nan,
            esum / wsum if wsum > 0 else np.nan,
            asum / esum if esum > 0 else np.nan,
        )

    cols = ["n", "exposure", "actual", "expected", "actual_mean", "expected_mean", "ae_ratio"]
    all_mask = np.ones(len(a), dtype=bool)

    if by is None:
        return pd.DataFrame([_row(all_mask)], columns=cols, index=pd.Index(["All"]))

    if isinstance(by, (dict, pd.DataFrame)):
        groups = dict(by) if isinstance(by, dict) else {c: by[c] for c in by.columns}
        rows, index = [], []
        for var, labels in groups.items():
            keys = np.asarray(labels)
            if keys.shape != a.shape:
                raise ValueError(f"by[{var!r}] must match actual/expected in length")
            for lvl in pd.unique(keys):
                rows.append(_row(keys == lvl))
                index.append((var, lvl))
        if include_total:
            rows.append(_row(all_mask))
            index.append(("All", ""))
        return pd.DataFrame(
            rows,
            columns=cols,
            index=pd.MultiIndex.from_tuples(index, names=["variable", "level"]),
        )

    keys = np.asarray(by)
    if keys.shape != a.shape:
        raise ValueError("by must match actual/expected in length")
    levels = pd.Series(keys).drop_duplicates().sort_values(kind="stable").to_list()
    rows = [_row(keys == lvl) for lvl in levels]
    index = pd.Index(levels)
    if include_total:
        rows.append(_row(all_mask))
        index = pd.Index(list(levels) + ["All"])
    return pd.DataFrame(rows, columns=cols, index=index)


def compare_models(
    models,
    data: pd.DataFrame,
    response: str,
    exposure: str | None = None,
    offset: str | None = None,
    weights: str | None = None,
    n_bands: int = 10,
) -> pd.DataFrame:
    """Side-by-side scorecard for fitted GLMs on one evaluation frame.

    Every model is scored on the *same* data -- pass a held-out validation
    frame (see :func:`ratingmodels.temporal_split` /
    :func:`ratingmodels.group_split`) for an honest comparison, or the
    training frame for an in-sample one.

    Parameters
    ----------
    models : mapping or sequence
        ``name -> fitted GLMRelativities`` (or a sequence, auto-named
        ``model_1``, ``model_2``, ...). Each model must expose the fitted
        interface (``predict``, family deviance); i.e. any
        :class:`GLMRelativities`-compatible object.
    data, response, exposure, offset, weights : DataFrame / str
        Evaluation frame and its column names, as in ``GLMRelativities.fit``.
    n_bands : int
        Bands for the calibration-error summary.

    Returns
    -------
    pandas.DataFrame
        One row per model: ``family``, ``n_params``, ``converged``,
        ``dispersion`` (training), then evaluation-frame metrics
        ``deviance``, ``null_deviance``, ``deviance_explained``, ``gini``,
        ``ae_ratio``, and ``calibration_error`` (the exposure-weighted mean
        absolute deviation of band-level A/E from 1.0).

    Notes
    -----
    Deviance is family-specific: it is comparable between models of the same
    family, while ``gini``, ``ae_ratio``, and ``calibration_error`` are
    comparable across families. No AIC is reported -- the standard errors are
    quasi-likelihood, so a true likelihood is not available.
    """
    if isinstance(models, Mapping):
        named = list(models.items())
    else:
        named = [(f"model_{i + 1}", m) for i, m in enumerate(models)]
    if not named:
        raise ValueError("no models given")

    y = data[response].to_numpy(dtype=float)
    n = len(data)
    prior_w = data[weights].to_numpy(dtype=float) if weights is not None else np.ones(n)
    expo = data[exposure].to_numpy(dtype=float) if exposure is not None else None

    eta_offset = np.zeros(n)
    if exposure is not None:
        if np.any(expo <= 0):
            raise ValueError("exposure must be positive")
        eta_offset += np.log(expo)
    if offset is not None:
        eta_offset += data[offset].to_numpy(dtype=float)
    rate0 = np.sum(prior_w * y) / np.sum(prior_w * np.exp(eta_offset))
    mu0 = max(rate0, 1e-12) * np.exp(eta_offset)

    rows = []
    for name, model in named:
        if getattr(model, "coefficients_", None) is None:
            raise RuntimeError(f"model {name!r} is not fit")
        mu = model.predict(data, exposure=exposure, offset=offset)
        p = model._power()
        dev = model._deviance(y, mu, prior_w, p)
        null_dev = model._deviance(y, mu0, prior_w, p)
        cal = calibration_table(y, mu, exposure=expo, n_bands=n_bands)
        ok = cal["ae_ratio"].notna()
        cal_err = (
            float(np.average(np.abs(cal.loc[ok, "ae_ratio"] - 1.0),
                             weights=cal.loc[ok, "exposure"]))
            if ok.any()
            else np.nan
        )
        rows.append(
            {
                "family": model.family,
                "n_params": int(len(model.coefficients_)),
                "converged": bool(model.converged_),
                "dispersion": float(model.dispersion_),
                "deviance": dev,
                "null_deviance": null_dev,
                "deviance_explained": 1.0 - dev / null_dev if null_dev > 0 else np.nan,
                "gini": gini_coefficient(y, mu, exposure=expo),
                "ae_ratio": float(y.sum() / mu.sum()) if mu.sum() > 0 else np.nan,
                "calibration_error": cal_err,
            }
        )
    return pd.DataFrame(rows, index=pd.Index([nm for nm, _ in named], name="model"))
