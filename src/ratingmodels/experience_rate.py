r"""Experience rate construction.

The experience rate develops a group's own claims into a charged rate:

1. **Pool** large claims at a pooling point :math:`P`, removing the excess
   :math:`\sum_i \max(0, c_i - P)` so a few catastrophic claims don't distort
   the manual-comparable base.
2. **Normalize** to a loss cost by dividing pooled claims by exposure units.
3. **Trend** forward to the rating-period cost level.
4. **Add back** a pooling charge (the expected cost of the excess layer,
   spread across the book) and apply benefit/demographic adjustments.
5. **Load** for expenses and margin via the target loss ratio.

.. math::
    \text{exp loss cost}
      = \frac{C - \text{excess}}{E}\cdot(1+t)^{\Delta}
        \cdot f_{\text{ben}} f_{\text{demo}} + \text{pooling charge}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pandas as pd

from ._utils import (
    Numeric,
    as_float_array,
    maybe_float,
    require_nonnegative,
    require_positive,
    require_unit_interval,
)
from .trend import trend_factor
from .loading import RetentionLoad


def pool_claims(
    claims, pooling_point: float, by=None
) -> "tuple[float, float] | tuple[pd.Series, pd.Series]":
    r"""Split claims into a pooled (capped) total and the excess above ``P``.

    Returns ``(capped_total, excess)`` where
    ``excess = sum(max(0, claim - pooling_point))``.

    With ``by`` (group labels aligned with ``claims``), pooling is applied
    within each group and both returns are Series indexed by group -- one
    ``groupby`` pass pools a whole claim file.
    """
    require_positive(pooling_point, "pooling_point")
    arr = as_float_array(claims, "claims")
    if np.any(arr < 0):
        raise ValueError("claims must be non-negative")
    excess_by_claim = np.maximum(0.0, arr - pooling_point)
    if by is not None:
        frame = pd.DataFrame(
            {"capped": arr - excess_by_claim, "excess": excess_by_claim},
            index=getattr(claims, "index", None),
        )
        keys = by if not isinstance(by, str) else pd.Series(claims)[by]
        grouped = frame.groupby(np.asarray(keys), sort=True).sum()
        return grouped["capped"].rename("capped_total"), grouped["excess"].rename("excess")
    excess = float(excess_by_claim.sum())
    return float(arr.sum() - excess), excess


def expected_excess_charge(
    claims, pooling_point: float, exposure: Numeric, by=None
) -> Numeric:
    """Naive pooling charge per exposure unit: observed excess spread over exposure.

    A filed pooling charge is normally derived from book-wide excess
    experience or an EVT tail model (see the ``extremeloss`` package); this
    helper gives the simple group-level estimate. With ``by``, the charge is
    computed per group (``exposure`` then aligns to the group index --
    a Series/mapping keyed by group, or a scalar broadcast to all groups).
    """
    _, excess = pool_claims(claims, pooling_point, by=by)
    if by is not None and not isinstance(exposure, (int, float)):
        exposure = pd.Series(exposure).reindex(excess.index)
    exposure = require_positive(exposure, "exposure")
    return maybe_float(excess / exposure)


@dataclass
class ExperienceRate:
    """Develop an experience rate from incurred claims and exposure.

    Every numeric field follows the vectorization contract: pass columns
    (Series of claims, exposures, per-group trends...) and every derived
    quantity -- :meth:`pooled_loss_cost`, :meth:`loss_cost`, :meth:`rate` --
    comes back as a Series on the same index. Scalars broadcast, so a single
    trend assumption prices against per-group claims.

    Parameters
    ----------
    incurred_claims : float
        Total incurred (completed) claims over the experience period.
    exposure : float
        Exposure units (member-months, policy months, earned exposures, ...).
    trend_annual : float
        Annual claims trend.
    trend_years : float
        Years from experience midpoint to rating midpoint.
    pooled_excess : float
        Claim dollars removed by pooling (from :func:`pool_claims`). Default 0.
    pooling_charge : float
        Pooling charge added back, per exposure unit. Default 0.
    benefit_factor, demographic_factor : float
        Multiplicative adjustments for benefit/demographic changes between the
        experience and rating periods. Default 1.0.
    target_loss_ratio : float
        Claims / premium target used to load to a charged rate.
    """

    incurred_claims: Numeric
    exposure: Numeric
    trend_annual: Numeric = 0.0
    trend_years: Numeric = 1.0
    pooled_excess: Numeric = 0.0
    pooling_charge: Numeric = 0.0
    benefit_factor: Numeric = 1.0
    demographic_factor: Numeric = 1.0
    target_loss_ratio: Numeric = 0.85
    retention: "RetentionLoad | None" = None

    def __post_init__(self) -> None:
        self.incurred_claims = require_nonnegative(self.incurred_claims, "incurred_claims")
        self.exposure = require_positive(self.exposure, "exposure")
        self.pooled_excess = require_nonnegative(self.pooled_excess, "pooled_excess")
        self.pooling_charge = require_nonnegative(self.pooling_charge, "pooling_charge")
        self.benefit_factor = require_positive(self.benefit_factor, "benefit_factor")
        self.demographic_factor = require_positive(self.demographic_factor, "demographic_factor")
        if self.retention is None:
            self.target_loss_ratio = require_unit_interval(
                self.target_loss_ratio, "target_loss_ratio", closed=False
            )

    def pooled_loss_cost(self) -> Numeric:
        """Pooled (capped) claims per exposure unit, before trend."""
        return (self.incurred_claims - self.pooled_excess) / self.exposure

    def trend_factor(self) -> Numeric:
        return trend_factor(self.trend_annual, self.trend_years)

    def loss_cost(self) -> Numeric:
        """Trended, pooled, adjusted experience loss cost (charge added back)."""
        trended = (
            self.pooled_loss_cost()
            * self.trend_factor()
            * self.benefit_factor
            * self.demographic_factor
        )
        return maybe_float(trended + self.pooling_charge)

    def rate(self) -> Numeric:
        """Charged experience rate per exposure unit.

        Uses ``retention`` (the full gross-up) when supplied, otherwise
        ``loss cost / target_loss_ratio``.
        """
        if self.retention is not None:
            return self.retention.gross_rate(self.loss_cost())
        return maybe_float(self.loss_cost() / self.target_loss_ratio)
