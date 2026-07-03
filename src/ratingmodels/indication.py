r"""Rate indication: the central object that turns experience and manual
inputs into an indicated rate and a rate change.

Two standard methods are exposed.

**Build-up (loss-ratio loaded).**  Blend experience and manual *claims* by
credibility, then gross up by the target loss ratio:

.. math::
    \text{indicated rate} = \frac{Z\,\text{exp claims} + (1-Z)\,\text{man claims}}
                                 {\text{target LR}}, \qquad
    \text{change} = \frac{\text{indicated}}{\text{current}} - 1.

**Loss-ratio (credibility-weighted indication).**  Weight the experience
indication against a trend-only ("no experience") indication:

.. math::
    \text{change} = Z\!\left(\frac{\text{exp LR}}{\text{target LR}} - 1\right)
                    + (1 - Z)\,\big((1+t)^{\Delta}-1\big).
"""
from __future__ import annotations

from dataclasses import dataclass

from ._utils import (
    Numeric,
    as_numeric,
    maybe_float,
    require_positive,
    require_unit_interval,
)
from .blend import blend
from .decomposition import RateChangeDecomposition, decompose_rate_change
from .loading import RetentionLoad


@dataclass
class RateIndication:
    r"""Develop an indicated rate from experience and manual inputs.

    Every numeric field follows the vectorization contract: pass columns
    (Series of loss costs, credibilities, current rates...) to run the whole
    book's indication in one object; each method returns a Series on the
    shared index, and :meth:`rate_change_decomposition` returns per-case
    driver tables.

    Parameters
    ----------
    experience_loss_cost : float or array-like
        Trended, pooled, adjusted experience loss cost (per exposure unit)
        (see :class:`ratingmodels.ExperienceRate`).
    manual_loss_cost : float
        Manual loss cost at the rating-period level
        (see :class:`ratingmodels.ManualRate`).
    credibility : float
        Credibility ``Z`` assigned to experience, in [0, 1].
    current_rate : float
        Current charged rate per exposure unit.
    target_loss_ratio : float
        Claims / premium target used to load claims to a charged rate.
    current_premium : float, optional
        On-level earned premium over the experience period; required only for
        the loss-ratio method.
    exposure : float, optional
        Exposure units over the experience period; required for the loss-ratio
        method (with ``current_premium``) to form an experience loss ratio.
    trend_total_factor : float
        Total claims trend factor :math:`(1+t)^\Delta`; used by the loss-ratio
        method's trend-only side and by the decomposition. Default 1.0.
    benefit_factor, demographic_factor : float
        Driver factors for the rate-change decomposition. Default 1.0.
    """

    experience_loss_cost: Numeric
    manual_loss_cost: Numeric
    credibility: Numeric
    current_rate: Numeric
    target_loss_ratio: Numeric = 0.85
    current_premium: Numeric | None = None
    exposure: Numeric | None = None
    trend_total_factor: Numeric = 1.0
    benefit_factor: Numeric = 1.0
    demographic_factor: Numeric = 1.0
    retention: "RetentionLoad | None" = None

    def __post_init__(self) -> None:
        self.experience_loss_cost = as_numeric(
            self.experience_loss_cost, "experience_loss_cost"
        )
        require_positive(self.experience_loss_cost + 1e-12, "experience_loss_cost")
        self.manual_loss_cost = require_positive(self.manual_loss_cost, "manual_loss_cost")
        self.credibility = require_unit_interval(self.credibility, "credibility")
        self.current_rate = require_positive(self.current_rate, "current_rate")
        self.target_loss_ratio = require_unit_interval(
            self.target_loss_ratio, "target_loss_ratio", closed=False
        )

    # ----- claims-level blending ----- #
    def blended_loss_cost(self) -> Numeric:
        return blend(
            self.experience_loss_cost, self.manual_loss_cost, self.credibility
        )

    # ----- charged rates ----- #
    def _gross(self, loss_cost: Numeric) -> Numeric:
        """Gross claims to a charged rate via retention, else target loss ratio."""
        if self.retention is not None:
            return self.retention.gross_rate(loss_cost)
        return maybe_float(loss_cost / self.target_loss_ratio)

    def experience_rate(self) -> Numeric:
        return self._gross(self.experience_loss_cost)

    def manual_rate(self) -> Numeric:
        return self._gross(self.manual_loss_cost)

    def blended_rate(self) -> Numeric:
        return self._gross(self.blended_loss_cost())

    # ----- indication (build-up) ----- #
    def indicated_rate(self) -> Numeric:
        """Indicated charged rate (build-up method)."""
        return self.blended_rate()

    def indicated_rate_change(self) -> Numeric:
        """Proportional change implied by the build-up indicated rate."""
        return maybe_float(self.indicated_rate() / self.current_rate - 1.0)

    # ----- indication (loss-ratio method) ----- #
    def experience_loss_ratio(self) -> Numeric:
        if self.current_premium is None or self.exposure is None:
            raise ValueError(
                "current_premium and exposure are required for the loss-ratio method"
            )
        premium = require_positive(self.current_premium, "current_premium")
        exposure = require_positive(self.exposure, "exposure")
        experience_claims = self.experience_loss_cost * exposure
        return maybe_float(experience_claims / premium)

    def loss_ratio_indication(self) -> Numeric:
        r"""Credibility-weighted loss-ratio rate change.

        Experience side: ``exp_LR / target_LR - 1``.
        Trend-only side: ``trend_total_factor - 1``.
        """
        exp_side = self.experience_loss_ratio() / self.target_loss_ratio - 1.0
        trend_side = self.trend_total_factor - 1.0
        z = self.credibility
        return maybe_float(z * exp_side + (1 - z) * trend_side)

    # ----- decomposition ----- #
    def rate_change_decomposition(self) -> RateChangeDecomposition:
        r"""Decompose the build-up indicated change into named drivers.

        Drivers:

        * ``trend`` -- the claims trend factor,
        * ``experience`` -- the credibility effect, blended/manual claims
          (equals 1 when ``Z = 0``),
        * ``benefit`` and ``demographic`` -- supplied adjustment factors.

        Any remaining movement (rate adequacy / loading) is absorbed by an
        explicit ``residual`` factor so the parts reconcile to the total.
        """
        experience_factor = self.blended_loss_cost() / self.manual_loss_cost
        drivers = {
            "trend": self.trend_total_factor,
            "experience": experience_factor,
            "benefit": self.benefit_factor,
            "demographic": self.demographic_factor,
        }
        total = self.indicated_rate() / self.current_rate
        return decompose_rate_change(drivers, total_factor=total)
