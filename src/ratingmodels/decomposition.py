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

Driver values follow the vectorization contract: pass columns (one value per
case) and the decomposition is computed row-by-row. ``factors`` and
``contributions`` then come back as DataFrames -- one row per case, one
column per driver -- and :meth:`RateChangeDecomposition.to_frame` stacks them
into a tidy ``(case, driver)`` long table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np
import pandas as pd

from ._utils import Numeric, common_index, is_arraylike, maybe_float, product


@dataclass
class RateChangeDecomposition:
    """Result of :func:`decompose_rate_change`.

    For a scalar decomposition ``factors`` and ``contributions`` are Series
    indexed by driver. For a vectorized one they are DataFrames (rows =
    cases, columns = drivers) and ``total_factor`` is a Series/array.
    """

    total_factor: Numeric
    factors: Union[pd.Series, pd.DataFrame]          # multiplicative, incl. any residual
    contributions: Union[pd.Series, pd.DataFrame]    # additive percentage points, sum == total-1

    @property
    def total_change(self) -> Numeric:
        return maybe_float(self.total_factor - 1.0)

    def to_frame(self) -> pd.DataFrame:
        if isinstance(self.factors, pd.DataFrame):
            out = pd.concat(
                {
                    "factor": self.factors.stack(),
                    "pct_point_contribution": self.contributions.stack(),
                },
                axis=1,
            )
            out.index = out.index.set_names(["case", "driver"])
            return out
        return pd.DataFrame(
            {"factor": self.factors, "pct_point_contribution": self.contributions}
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if isinstance(self.factors, pd.DataFrame):
            return (
                f"RateChangeDecomposition(n={len(self.factors)}, "
                f"drivers={list(self.factors.columns)})"
            )
        return (
            f"RateChangeDecomposition(total_change={self.total_change:+.4%}, "
            f"drivers={list(self.factors.index)})"
        )


def _scalar_decompose(
    names: list[str], vals: np.ndarray, total_factor: float | None
) -> RateChangeDecomposition:
    fac = dict(zip(names, vals, strict=True))
    prod = product(vals)
    if total_factor is None:
        total = prod
    else:
        total = float(total_factor)
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


def _vector_decompose(
    names: list[str], values: list, total_factor
) -> RateChangeDecomposition:
    idx = common_index(list(values) + ([total_factor] if total_factor is not None else []))
    arrs = [np.asarray(v, dtype=float) for v in values]
    n = max(a.shape[0] for a in arrs if a.ndim == 1)
    mat = np.vstack([np.broadcast_to(a, (n,)) for a in arrs])  # (k, n)
    if np.any(mat <= 0):
        raise ValueError("all factors must be positive")
    case_index = idx if idx is not None else pd.RangeIndex(n)

    prod = np.exp(np.sum(np.log(mat), axis=0))
    if total_factor is None:
        total = prod
    else:
        total = np.broadcast_to(np.asarray(total_factor, dtype=float), (n,)).copy()
        if np.any(total <= 0):
            raise ValueError("total_factor must be positive")
        residual = total / prod
        if not np.allclose(residual, 1.0, rtol=1e-9, atol=1e-12):
            names = names + ["residual"]
            mat = np.vstack([mat, residual])

    factors = pd.DataFrame(mat.T, index=case_index, columns=names)

    log_total = np.log(total)
    small = np.abs(log_total) < 1e-12
    denom = np.where(small, 1.0, log_total)
    shares = np.log(mat) / denom            # (k, n)
    contrib = shares * (total - 1.0)
    contrib = np.where(small, 0.0, contrib)
    contributions = pd.DataFrame(contrib.T, index=case_index, columns=names)

    total_out = pd.Series(total, index=case_index, name="total_factor") if idx is not None else total
    return RateChangeDecomposition(
        total_factor=total_out, factors=factors, contributions=contributions
    )


def decompose_rate_change(
    factors: Mapping[str, Numeric],
    total_factor: Numeric | None = None,
) -> RateChangeDecomposition:
    r"""Attribute a rate change to multiplicative drivers.

    Parameters
    ----------
    factors : mapping
        Named driver factors (e.g. ``{"trend": 1.075, "experience": 0.96,
        "benefit": 1.02, "demographic": 1.01}``). Each must be positive.
        Values may be scalars (one decomposition) or vectors under the
        vectorization contract (one decomposition per row; scalars
        broadcast).
    total_factor : float or array-like, optional
        Independently computed total change factor (indicated / current). If
        given and it differs from the product of ``factors``, a ``residual``
        factor is appended so the decomposition reconciles exactly. If omitted,
        the total is taken to be the product of the supplied factors.
    """
    names = list(factors.keys())
    values = list(factors.values())
    if any(is_arraylike(v) for v in values) or is_arraylike(total_factor):
        return _vector_decompose(names, values, total_factor)

    vals = np.array([float(v) for v in values], dtype=float)
    if np.any(vals <= 0):
        raise ValueError("all factors must be positive")
    if total_factor is not None and float(total_factor) <= 0:
        raise ValueError("total_factor must be positive")
    return _scalar_decompose(names, vals, total_factor)
