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

from ._utils import Numeric, maybe_float, product, require_positive, require_unit_interval
from .buildup import BuildUpResult, checkpoint, evaluate, multiply, start
from .loading import RetentionLoad


def manual_loss_cost(base_loss_cost: Numeric, factors: Sequence[Numeric]) -> Numeric:
    r"""Base loss cost scaled by the product of relativities.

    Elementwise: pass columns (a Series of base rates and Series factors) to
    price every row at once; scalars broadcast.
    """
    base_loss_cost = require_positive(base_loss_cost, "base_loss_cost")
    return maybe_float(base_loss_cost * product(factors))


def aggregate_demographic_factor(
    census: pd.DataFrame,
    factor_col: str,
    weight_col: str = "count",
    by: str | Sequence[str] | None = None,
) -> "float | pd.Series":
    """Weighted average of a unit-level demographic factor (e.g. an age/sex
    factor weighted by member counts).

    With ``by`` (a column or list of columns), aggregates within each group
    and returns a Series indexed by group -- one demographic factor per
    group from a single census frame.
    """
    if by is not None:
        def _agg(g: pd.DataFrame) -> float:
            return aggregate_demographic_factor(g, factor_col, weight_col)
        return census.groupby(by, sort=True).apply(_agg, include_groups=False).rename(factor_col)
    w = census[weight_col].to_numpy(dtype=float)
    f = census[factor_col].to_numpy(dtype=float)
    if w.sum() <= 0:
        raise ValueError("total weight must be positive")
    return float(np.average(f, weights=w))


@dataclass
class ManualRate:
    """Build a manual rate from a base and a set of named relativities.

    Every numeric field follows the vectorization contract: Series-valued
    bases and factors build the whole book's manual rates in one object,
    and :meth:`loss_cost` / :meth:`rate` / :meth:`breakdown` come back
    per row.

    Parameters
    ----------
    base_loss_cost : float or array-like
        Base loss cost (per exposure unit) at the rating-period level (see
        :func:`ratingmodels.base_rate_from_experience` to derive it).
    factors : mapping
        Named relativities, e.g. ``{"area": 1.05, "industry": 0.97, ...}``;
        values may be scalars or Series columns.
    target_loss_ratio : float
        Claims / premium target used to gross up to a charged rate. Ignored
        when ``retention`` is supplied.
    retention : RetentionLoad, optional
        Full expense / profit loading. When provided, the charged rate is built
        with the fundamental insurance equation instead of a single loss ratio,
        and fixed expense is applied per exposure unit (flat across cells).
    """

    base_loss_cost: Numeric
    factors: Mapping[str, Numeric] = field(default_factory=dict)
    target_loss_ratio: Numeric = 0.85
    retention: "RetentionLoad | None" = None

    def __post_init__(self) -> None:
        self.base_loss_cost = require_positive(self.base_loss_cost, "base_loss_cost")
        if self.retention is None:
            self.target_loss_ratio = require_unit_interval(
                self.target_loss_ratio, "target_loss_ratio", closed=False
            )

    def total_relativity(self) -> Numeric:
        return product(self.factors.values())

    def loss_cost(self) -> Numeric:
        """Expected manual loss cost (before expense/margin loading)."""
        return maybe_float(self.base_loss_cost * self.total_relativity())

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

    def rate(self) -> Numeric:
        """Charged manual rate per exposure unit.

        Uses ``retention`` (the full gross-up) when supplied, otherwise
        ``loss cost / target_loss_ratio``.
        """
        if self.retention is not None:
            return self.retention.gross_rate(self.loss_cost())
        return maybe_float(self.loss_cost() / self.target_loss_ratio)

    def with_factor(self, name: str, value: Numeric) -> "ManualRate":
        new = dict(self.factors)
        new[name] = value
        return ManualRate(self.base_loss_cost, new, self.target_loss_ratio, self.retention)
