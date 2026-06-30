r"""Base-rate construction from book experience, with off-balancing.

The base rate is the cost level for the reference cell (all relativities equal
1). It is backed out from the book so that base times relativities reproduces
the book's losses. For risks :math:`i` with exposure :math:`e_i`, relativity
:math:`r_i = \prod_k f_{ki}`, and trended/developed loss :math:`L_i`:

.. math::
    \bar r = \frac{\sum_i e_i r_i}{\sum_i e_i}, \qquad
    B = \frac{\sum_i L_i}{\sum_i e_i r_i} = \frac{\bar L}{\bar r}.

By construction :math:`\sum_i e_i\, B r_i = \sum_i L_i`. When relativities are
revised, the average relativity moves and the overall premium level drifts
unless the base is **off-balanced**. Moving from average :math:`\bar r_0` to
:math:`\bar r_1`, with an intended overall change :math:`\Delta`:

.. math::
    B_1 = B_0 \, \frac{\bar r_0}{\bar r_1} \, (1 + \Delta),

where :math:`\bar r_0 / \bar r_1` is the off-balance correction that holds the
book level neutral.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import product, require_positive


@dataclass
class BaseRateResult:
    """Result of :func:`base_rate_from_experience`."""

    base_loss_cost: float
    average_relativity: float
    average_loss_cost: float
    total_exposure: float

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BaseRateResult(base_loss_cost={self.base_loss_cost:.4f}, "
            f"average_relativity={self.average_relativity:.4f}, "
            f"average_loss_cost={self.average_loss_cost:.4f})"
        )


def _relativity_vector(
    data: pd.DataFrame,
    relativity: str | None,
    factor_cols: Sequence[str] | None,
) -> np.ndarray:
    if relativity is not None:
        rel = data[relativity].to_numpy(dtype=float)
    elif factor_cols:
        rel = np.array(
            [product(row[c] for c in factor_cols) for _, row in data.iterrows()],
            dtype=float,
        )
    else:
        raise ValueError("provide either `relativity` or `factor_cols`")
    if np.any(rel <= 0):
        raise ValueError("relativities must be positive")
    return rel


def average_relativity(
    data: pd.DataFrame,
    exposure: str,
    relativity: str | None = None,
    factor_cols: Sequence[str] | None = None,
) -> float:
    r"""Exposure-weighted average relativity :math:`\bar r = \sum e_i r_i / \sum e_i`.

    Supply relativities either as a single ``relativity`` column or as
    ``factor_cols`` (per-row factors that are multiplied together).
    """
    e = data[exposure].to_numpy(dtype=float)
    if np.any(e <= 0):
        raise ValueError("exposures must be positive")
    rel = _relativity_vector(data, relativity, factor_cols)
    return float(np.sum(e * rel) / np.sum(e))


def base_rate_from_experience(
    data: pd.DataFrame,
    exposure: str,
    loss: str,
    relativity: str | None = None,
    factor_cols: Sequence[str] | None = None,
) -> BaseRateResult:
    r"""Indicated base loss cost from book experience (off-balance method).

    Returns :math:`B = \sum_i L_i / \sum_i e_i r_i` together with the average
    relativity and average loss cost. Gross ``base_loss_cost`` to a charged base
    rate with a :class:`ratingmodels.RetentionLoad`.

    Parameters
    ----------
    data : DataFrame
        One row per risk or rating cell.
    exposure, loss : str
        Column names for exposure (e.g. member-months) and trended/developed
        loss.
    relativity : str, optional
        Column of precomputed relativities :math:`r_i`.
    factor_cols : sequence of str, optional
        Columns of individual rating factors to multiply into :math:`r_i`
        (used when ``relativity`` is not supplied).
    """
    e = data[exposure].to_numpy(dtype=float)
    if np.any(e <= 0):
        raise ValueError("exposures must be positive")
    losses = data[loss].to_numpy(dtype=float)
    rel = _relativity_vector(data, relativity, factor_cols)

    total_exposure = float(np.sum(e))
    exposure_weighted_rel = float(np.sum(e * rel))  # = sum(e_i r_i)
    base = float(np.sum(losses) / exposure_weighted_rel)
    return BaseRateResult(
        base_loss_cost=base,
        average_relativity=exposure_weighted_rel / total_exposure,
        average_loss_cost=float(np.sum(losses) / total_exposure),
        total_exposure=total_exposure,
    )


def off_balance_factor(current_avg_relativity: float, new_avg_relativity: float) -> float:
    r"""Off-balance correction :math:`\bar r_0 / \bar r_1` from revising relativities."""
    require_positive(current_avg_relativity, "current_avg_relativity")
    require_positive(new_avg_relativity, "new_avg_relativity")
    return current_avg_relativity / new_avg_relativity


def rebalance_base_rate(
    current_base: float,
    current_avg_relativity: float,
    new_avg_relativity: float,
    overall_change: float = 0.0,
) -> float:
    r"""Off-balanced new base rate :math:`B_1 = B_0 (\bar r_0/\bar r_1)(1+\Delta)`.

    Holds the overall premium level neutral when relativities change, then
    applies the intended overall rate change ``overall_change`` (:math:`\Delta`).
    """
    require_positive(current_base, "current_base")
    factor = off_balance_factor(current_avg_relativity, new_avg_relativity)
    return current_base * factor * (1.0 + overall_change)
