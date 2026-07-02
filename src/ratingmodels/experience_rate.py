r"""Experience rate construction.

The experience rate develops a group's own claims into a charged rate:

1. **Pool** large claims at a pooling point :math:`P`, removing the excess
   :math:`\sum_i \max(0, c_i - P)` so a few catastrophic claims don't distort
   the manual-comparable base.
2. **Normalize** to a PMPM by dividing pooled claims by member-months.
3. **Trend** forward to the rating-period cost level.
4. **Add back** a pooling charge (the expected cost of the excess layer,
   spread across the book) and apply benefit/demographic adjustments.
5. **Load** for expenses and margin via the target loss ratio.

.. math::
    \text{exp claims PMPM}
      = \frac{C - \text{excess}}{E}\cdot(1+t)^{\Delta}
        \cdot f_{\text{ben}} f_{\text{demo}} + \text{pooling charge}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._utils import (
    as_float_array,
    require_nonnegative,
    require_positive,
    require_unit_interval,
)
from .trend import trend_factor
from .loading import RetentionLoad


def pool_claims(claims, pooling_point: float) -> tuple[float, float]:
    r"""Split claims into a pooled (capped) total and the excess above ``P``.

    Returns ``(capped_total, excess)`` where
    ``excess = sum(max(0, claim - pooling_point))``.
    """
    require_positive(pooling_point, "pooling_point")
    arr = as_float_array(claims, "claims")
    if np.any(arr < 0):
        raise ValueError("claims must be non-negative")
    excess = float(np.sum(np.maximum(0.0, arr - pooling_point)))
    return float(arr.sum() - excess), excess


def expected_excess_charge(claims, pooling_point: float, exposure: float) -> float:
    """Naive pooling charge PMPM: observed excess spread over exposure.

    A filed pooling charge is normally derived from book-wide excess
    experience or an EVT tail model (see the ``extremeloss`` package); this
    helper gives the simple group-level estimate.
    """
    _, excess = pool_claims(claims, pooling_point)
    return excess / require_positive(exposure, "exposure")


@dataclass
class ExperienceRate:
    """Develop an experience rate from incurred claims and exposure.

    Parameters
    ----------
    incurred_claims : float
        Total incurred (completed) claims over the experience period.
    exposure : float
        Member-months (or other PMPM exposure base).
    trend_annual : float
        Annual claims trend.
    trend_years : float
        Years from experience midpoint to rating midpoint.
    pooled_excess : float
        Claim dollars removed by pooling (from :func:`pool_claims`). Default 0.
    pooling_charge_pmpm : float
        Pooling charge added back, PMPM. Default 0.
    benefit_factor, demographic_factor : float
        Multiplicative adjustments for benefit/demographic changes between the
        experience and rating periods. Default 1.0.
    target_loss_ratio : float
        Claims / premium target used to load to a charged rate.
    """

    incurred_claims: float
    exposure: float
    trend_annual: float = 0.0
    trend_years: float = 1.0
    pooled_excess: float = 0.0
    pooling_charge_pmpm: float = 0.0
    benefit_factor: float = 1.0
    demographic_factor: float = 1.0
    target_loss_ratio: float = 0.85
    retention: "RetentionLoad | None" = None

    def __post_init__(self) -> None:
        require_nonnegative(self.incurred_claims, "incurred_claims")
        require_positive(self.exposure, "exposure")
        require_nonnegative(self.pooled_excess, "pooled_excess")
        require_nonnegative(self.pooling_charge_pmpm, "pooling_charge_pmpm")
        require_positive(self.benefit_factor, "benefit_factor")
        require_positive(self.demographic_factor, "demographic_factor")
        if self.retention is None:
            require_unit_interval(self.target_loss_ratio, "target_loss_ratio", closed=False)

    def pooled_claims_pmpm(self) -> float:
        """Pooled (capped) claims per member-month, before trend."""
        return (self.incurred_claims - self.pooled_excess) / self.exposure

    def trend_factor(self) -> float:
        return trend_factor(self.trend_annual, self.trend_years)

    def claims_pmpm(self) -> float:
        """Trended, pooled, adjusted experience claims PMPM (charge added back)."""
        trended = (
            self.pooled_claims_pmpm()
            * self.trend_factor()
            * self.benefit_factor
            * self.demographic_factor
        )
        return trended + self.pooling_charge_pmpm

    def rate(self) -> float:
        """Charged experience rate PMPM.

        Uses ``retention`` (the full gross-up) when supplied, otherwise
        ``claims PMPM / target_loss_ratio``.
        """
        if self.retention is not None:
            return self.retention.gross_rate(self.claims_pmpm())
        return self.claims_pmpm() / self.target_loss_ratio
