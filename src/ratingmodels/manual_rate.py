r"""Manual rate construction.

The manual (book) rate is a base cost level scaled by the product of rating
relativities and then loaded for expenses and margin:

.. math::
    \text{manual loss cost} = \text{base} \times \prod_i f_i, \qquad
    \text{manual rate} = \frac{\text{manual loss cost}}{\text{target loss ratio}}.

For a group, unit-level demographic factors are aggregated to a single
relativity (exposure-weighted) before composing with group-level factors
(area, industry, group size, network, plan/benefit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import product, require_positive, require_unit_interval
from .buildup import BuildUpResult, checkpoint, evaluate, multiply, start
from .loading import RetentionLoad


def manual_loss_cost(base_loss_cost: float, factors: Sequence[float]) -> float:
    r"""Base loss cost scaled by the product of relativities."""
    require_positive(base_loss_cost, "base_loss_cost")
    return base_loss_cost * product(factors)


def aggregate_demographic_factor(
    census: pd.DataFrame,
    factor_col: str,
    weight_col: str = "count",
) -> float:
    """Weighted average of a unit-level demographic factor (e.g. an age/sex
    factor weighted by member counts)."""
    w = census[weight_col].to_numpy(dtype=float)
    f = census[factor_col].to_numpy(dtype=float)
    if w.sum() <= 0:
        raise ValueError("total weight must be positive")
    return float(np.average(f, weights=w))


@dataclass
class ManualRate:
    """Build a manual rate from a base and a set of named relativities.

    Parameters
    ----------
    base_loss_cost : float
        Base loss cost (per exposure unit) at the rating-period level (see
        :func:`ratingmodels.base_rate_from_experience` to derive it).
    factors : mapping
        Named relativities, e.g. ``{"area": 1.05, "industry": 0.97, ...}``.
    target_loss_ratio : float
        Claims / premium target used to gross up to a charged rate. Ignored
        when ``retention`` is supplied.
    retention : RetentionLoad, optional
        Full expense / profit loading. When provided, the charged rate is built
        with the fundamental insurance equation instead of a single loss ratio,
        and fixed expense is applied per exposure unit (flat across cells).
    """

    base_loss_cost: float
    factors: Mapping[str, float] = field(default_factory=dict)
    target_loss_ratio: float = 0.85
    retention: "RetentionLoad | None" = None

    def __post_init__(self) -> None:
        require_positive(self.base_loss_cost, "base_loss_cost")
        if self.retention is None:
            require_unit_interval(self.target_loss_ratio, "target_loss_ratio", closed=False)

    def total_relativity(self) -> float:
        return product(self.factors.values())

    def loss_cost(self) -> float:
        """Expected manual loss cost (before expense/margin loading)."""
        return self.base_loss_cost * self.total_relativity()

    def steps(self) -> list:
        """The manual claims build-up as an ordered list of steps."""
        s = [start("Base claims cost", self.base_loss_cost)]
        s += [multiply(name, factor) for name, factor in self.factors.items()]
        s.append(checkpoint("Manual loss cost"))
        return s

    def breakdown(self) -> "BuildUpResult":
        """Audit trail of the manual claims build-up (base x each relativity).

        The final running total equals :meth:`loss_cost` up to floating point.
        """
        return evaluate(self.steps())

    def rate(self) -> float:
        """Charged manual rate per exposure unit.

        Uses ``retention`` (the full gross-up) when supplied, otherwise
        ``loss cost / target_loss_ratio``.
        """
        if self.retention is not None:
            return self.retention.gross_rate(self.loss_cost())
        return self.loss_cost() / self.target_loss_ratio

    def with_factor(self, name: str, value: float) -> "ManualRate":
        new = dict(self.factors)
        new[name] = value
        return ManualRate(self.base_loss_cost, new, self.target_loss_ratio)
