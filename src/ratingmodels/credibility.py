r"""Credibility — thin adapters over :mod:`actuarialpy`.

Credibility is a shared ecosystem primitive, and its home is ``actuarialpy``,
"where credibility sits next to the experience and ratemaking workflows that
consume it." To avoid a second copy of the math drifting out of sync, this
module does **not** reimplement the estimators: it delegates to ``actuarialpy``
and adapts the results to the names and shapes the rest of ``ratingmodels``
expects.

In particular :func:`buhlmann_straub` calls
:meth:`actuarialpy.BuhlmannStraub.from_frame`, which uses the general unbiased
estimators (handling unequal period counts), and repackages the fit as a
:class:`BuhlmannStraubResult` keyed by group.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import actuarialpy as ap

from ._utils import Numeric, maybe_float, require_nonnegative, require_positive


def full_credibility_standard(
    p: float = 0.90,
    k: float = 0.05,
    cv_severity: float | None = None,
) -> float:
    r"""Expected claim count required for full credibility.

    Delegates to :func:`actuarialpy.full_credibility_claims`. Returns
    :math:`(z_{(1+p)/2}/k)^2`, inflated by :math:`1 + \mathrm{cv}^2` when
    ``cv_severity`` is supplied (aggregate losses rather than pure frequency).

    >>> round(full_credibility_standard(0.90, 0.05))
    1082
    """
    return maybe_float(
        ap.full_credibility_claims(confidence=p, tolerance=k, severity_cv=cv_severity)
    )


def limited_fluctuation_credibility(n: Numeric, n_full: Numeric) -> Numeric:
    r"""Partial credibility by the square-root rule, ``min(1, sqrt(n / n_full))``.

    Delegates to :func:`actuarialpy.limited_fluctuation_z`. ``n`` and ``n_full``
    are in consistent units (claims, policies, exposure units, ...).
    Elementwise: a Series of ``n`` returns a Series of ``Z``.
    """
    return maybe_float(ap.limited_fluctuation_z(n, n_full))


def buhlmann_credibility(exposure: Numeric, epv: Numeric, vhm: Numeric) -> Numeric:
    r"""Bühlmann credibility factor :math:`Z = n / (n + k)`, ``k = EPV/VHM``.

    This is the credibility *factor* given structural parameters; the
    greatest-accuracy *estimators* (fitting EPV/VHM from data) live in
    :class:`actuarialpy.Buhlmann` / :class:`actuarialpy.BuhlmannStraub`.
    Elementwise: a Series of exposures returns a Series of ``Z``.
    """
    exposure = require_nonnegative(exposure, "exposure")
    epv = require_positive(epv, "epv")
    vhm = require_positive(vhm, "vhm")
    k = epv / vhm
    return maybe_float(exposure / (exposure + k))


@dataclass
class BuhlmannStraubResult:
    """Result of an empirical Bühlmann-Straub fit, keyed by group."""

    k: float
    epv: float
    vhm: float
    overall_mean: float
    group_means: pd.Series
    credibility: pd.Series
    credibility_weighted: pd.Series

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BuhlmannStraubResult(k={self.k:.4g}, epv={self.epv:.4g}, "
            f"vhm={self.vhm:.4g}, n_groups={len(self.group_means)})"
        )


def buhlmann_straub(
    data: pd.DataFrame,
    group: str,
    period: str,
    value: str,
    exposure: str,
) -> BuhlmannStraubResult:
    r"""Empirical Bühlmann-Straub credibility from grouped exposure data.

    Thin wrapper over :meth:`actuarialpy.BuhlmannStraub.from_frame` (the general
    unbiased estimators) that returns a :class:`BuhlmannStraubResult` with
    per-group credibility and credibility-weighted means.

    Parameters
    ----------
    data : DataFrame
        Long-format data: one row per (group, period).
    group, period, value, exposure : str
        Column names. ``value`` is the per-unit observation (e.g. loss per
        member-month); ``exposure`` is the weight :math:`m_{ij}`.
    """
    model = ap.BuhlmannStraub.from_frame(
        data, group=group, value=value, weight=exposure, period=period
    )
    groups = model.groups_
    group_means = pd.Series(model.risk_means_, index=groups, name="group_mean")
    z = pd.Series(
        np.atleast_1d(model.z(model.weights)), index=groups, name="credibility"
    )
    cred_weighted = (z * group_means + (1 - z) * model.overall_mean).rename(
        "credibility_weighted"
    )
    return BuhlmannStraubResult(
        k=float(model.k),
        epv=float(model.epv),
        vhm=float(model.vhm),
        overall_mean=float(model.overall_mean),
        group_means=group_means,
        credibility=z,
        credibility_weighted=cred_weighted,
    )
