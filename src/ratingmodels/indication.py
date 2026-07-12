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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from actuarialpy import Experience

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

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


def _as_period_array(value, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(
            f"{name} must be a scalar or a length-{n} sequence; got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} has non-finite values")
    return arr


@dataclass
class ExperienceExhibit:
    r"""Assemble experience periods into the inputs a rate indication takes.

    :class:`RateIndication` consumes *point* inputs -- a trended, developed
    experience loss cost and an on-level premium. This is the object that
    produces them from per-period columns, with every adjustment a visible
    column of the worksheet: premium times its on-level factor, losses
    times development and trend, a loss ratio per period, and the weighted
    total. The natural factor producers are
    :func:`~ratingmodels.on_level_factors` (the ``on_level_factor``
    column) and ``actuarialpy.reserving.ChainLadder`` (per-origin
    development factors), but any source works -- this object composes, it
    does not re-derive.

    Parameters
    ----------
    earned_premium : array-like
        Earned premium per experience period, at historical rate level.
    losses : array-like
        Incurred losses per period (at whatever development and trend
        level the factor arguments then adjust for).
    on_level_factors, trend_factors, development_factors : scalar or array
        Per-period multiplicative adjustments. Default 1.0.
    weights : array-like, optional
        Weights for the *diagnostic* weighted loss ratio. Default: on-level
        premium, making the weighted ratio identical to the aggregate
        ratio -- which is also the convention :meth:`to_indication` uses,
        since the indication is built from period totals.
    period_labels : sequence, optional
        Exhibit index; defaults to ``0..n-1``.
    """

    earned_premium: object
    losses: object
    on_level_factors: object = 1.0
    trend_factors: object = 1.0
    development_factors: object = 1.0
    weights: object | None = None
    period_labels: Sequence | None = None
    _n: int = field(init=False, repr=False)

    def __post_init__(self):
        premium = np.asarray(self.earned_premium, dtype=float)
        if premium.ndim == 0:
            premium = premium.reshape(1)
        self._n = n = premium.shape[0]
        if np.any(premium <= 0) or not np.all(np.isfinite(premium)):
            raise ValueError("earned_premium must be positive and finite")
        self.earned_premium = premium
        self.losses = _as_period_array(self.losses, n, "losses")
        if np.any(self.losses < 0):
            raise ValueError("losses must be nonnegative")
        for name in ("on_level_factors", "trend_factors", "development_factors"):
            arr = _as_period_array(getattr(self, name), n, name)
            if np.any(arr <= 0):
                raise ValueError(f"{name} must be positive")
            setattr(self, name, arr)
        if self.weights is not None:
            self.weights = _as_period_array(self.weights, n, "weights")
            if np.any(self.weights < 0) or self.weights.sum() <= 0:
                raise ValueError("weights must be nonnegative with positive sum")
        if self.period_labels is not None and len(self.period_labels) != n:
            raise ValueError("period_labels must match the number of periods")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_experience(
        cls,
        exp: "Experience",
        *,
        freq: str = "YE",
        on_level_factors: object = 1.0,
        trend_factors: object = 1.0,
        development_factors: object = 1.0,
        weights: object | None = None,
    ) -> "ExperienceExhibit":
        """Build the worksheet from the canonical Experience.

        Premium comes from the bound ``revenue`` role and losses from the
        bound ``expense`` role, summed per ``freq`` period of the bound
        ``date`` role (annual by default). The factor arguments stay explicit
        -- on-leveling, development, and trend are judgment this object
        composes, not derives.
        """
        from actuarialpy import resolve_date, single_role

        revenue = single_role(exp.revenue, "revenue")
        expense = single_role(exp.expense, "expense")
        date_col = resolve_date(exp)
        dated = exp.data.assign(**{date_col: pd.to_datetime(exp.data[date_col])})
        per = (
            dated.set_index(date_col)[[revenue, expense]]
            .resample(freq)
            .sum()
        )
        labels = (
            list(per.index.year)
            if freq.upper().startswith(("Y", "A"))
            else list(per.index.astype(str))
        )
        return cls(
            earned_premium=per[revenue].to_numpy(),
            losses=per[expense].to_numpy(),
            on_level_factors=on_level_factors,
            trend_factors=trend_factors,
            development_factors=development_factors,
            weights=weights,
            period_labels=labels,
        )

    def exhibit(self) -> pd.DataFrame:
        """The worksheet: one row per period, every adjustment a column."""
        olp = self.earned_premium * self.on_level_factors
        adj = self.losses * self.development_factors * self.trend_factors
        w = self.weights if self.weights is not None else olp
        idx = pd.Index(
            self.period_labels if self.period_labels is not None else range(self._n),
            name="period",
        )
        return pd.DataFrame(
            {
                "earned_premium": self.earned_premium,
                "on_level_factor": self.on_level_factors,
                "on_level_premium": olp,
                "losses": self.losses,
                "development_factor": self.development_factors,
                "trend_factor": self.trend_factors,
                "adjusted_losses": adj,
                "loss_ratio": adj / olp,
                "weight": w,
            },
            index=idx,
        )

    @property
    def on_level_premium(self) -> float:
        """Total on-level earned premium across the periods."""
        return float((self.earned_premium * self.on_level_factors).sum())

    @property
    def adjusted_losses(self) -> float:
        """Total developed, trended losses across the periods."""
        return float(
            (self.losses * self.development_factors * self.trend_factors).sum()
        )

    @property
    def experience_loss_ratio(self) -> float:
        """Weighted per-period loss ratio (aggregate ratio by default)."""
        ex = self.exhibit()
        return float(np.average(ex["loss_ratio"], weights=ex["weight"]))

    # ------------------------------------------------------------------ #
    def to_indication(
        self,
        manual_loss_cost: float,
        credibility: float,
        current_rate: float,
        exposure: float,
        retention: RetentionLoad | None = None,
        target_loss_ratio: float = 0.85,
        **kwargs,
    ) -> RateIndication:
        """Wire the assembled totals into a :class:`RateIndication`.

        ``experience_loss_cost`` becomes ``adjusted_losses / exposure`` and
        ``current_premium`` becomes :attr:`on_level_premium`, so the
        indication's own ``experience_loss_ratio()`` reproduces this
        exhibit's aggregate ratio exactly. ``exposure`` is the total over
        the experience period, in the same units as ``manual_loss_cost``
        and ``current_rate``. Remaining keyword arguments
        (``trend_total_factor``, ``benefit_factor``, ...) pass through.
        """
        exposure = float(exposure)
        if exposure <= 0:
            raise ValueError("exposure must be positive")
        return RateIndication(
            experience_loss_cost=self.adjusted_losses / exposure,
            manual_loss_cost=manual_loss_cost,
            credibility=credibility,
            current_rate=current_rate,
            current_premium=self.on_level_premium,
            exposure=exposure,
            retention=retention,
            target_loss_ratio=target_loss_ratio,
            **kwargs,
        )
