r"""Rate-change decomposition (contribution-to-change).

A total rate change factor :math:`F` is the product of identifiable driver
factors. Two attributions are provided:

* **Multiplicative** -- the factors themselves, :math:`F = \prod_i f_i`.
* **Additive (percentage points)** -- a log-share normalization so the parts
  sum exactly to the total percentage change:

  .. math::
      c_i = \frac{\ln f_i}{\ln F}\,(F - 1), \qquad \sum_i c_i = F - 1.

If the supplied factors do not multiply to an independently computed total, a
``residual`` factor is added so the decomposition is exact and the unexplained
movement is explicit rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ._utils import product


@dataclass
class RateChangeDecomposition:
    """Result of :func:`decompose_rate_change`."""

    total_factor: float
    factors: pd.Series          # multiplicative, incl. any residual
    contributions: pd.Series    # additive percentage points, sum == total-1

    @property
    def total_change(self) -> float:
        return self.total_factor - 1.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"factor": self.factors, "pct_point_contribution": self.contributions}
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RateChangeDecomposition(total_change={self.total_change:+.4%}, "
            f"drivers={list(self.factors.index)})"
        )


def decompose_rate_change(
    factors: Mapping[str, float],
    total_factor: float | None = None,
) -> RateChangeDecomposition:
    r"""Attribute a rate change to multiplicative drivers.

    Parameters
    ----------
    factors : mapping
        Named driver factors (e.g. ``{"trend": 1.075, "experience": 0.96,
        "benefit": 1.02, "demographic": 1.01}``). Each must be positive.
    total_factor : float, optional
        Independently computed total change factor (indicated / current). If
        given and it differs from the product of ``factors``, a ``residual``
        factor is appended so the decomposition reconciles exactly. If omitted,
        the total is taken to be the product of the supplied factors.
    """
    names = list(factors.keys())
    vals = np.array([float(v) for v in factors.values()], dtype=float)
    if np.any(vals <= 0):
        raise ValueError("all factors must be positive")

    fac = dict(zip(names, vals))
    prod = product(vals)
    if total_factor is None:
        total = prod
    else:
        total = float(total_factor)
        if total <= 0:
            raise ValueError("total_factor must be positive")
        residual = total / prod
        if not np.isclose(residual, 1.0, rtol=1e-9, atol=1e-12):
            fac["residual"] = residual

    fseries = pd.Series(fac, name="factor")

    # additive log-share attribution; handle the F == 1 limit gracefully
    log_total = np.log(total)
    if abs(log_total) < 1e-12:
        contrib = pd.Series(0.0, index=fseries.index, name="pct_point_contribution")
    else:
        shares = np.log(fseries.to_numpy()) / log_total
        contrib = pd.Series(
            shares * (total - 1.0),
            index=fseries.index,
            name="pct_point_contribution",
        )

    return RateChangeDecomposition(
        total_factor=total, factors=fseries, contributions=contrib
    )
