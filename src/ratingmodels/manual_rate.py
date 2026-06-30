r"""Manual rate construction.

The manual (book) rate is a base cost level scaled by the product of rating
relativities and then loaded for expenses and margin:

.. math::
    \text{manual PMPM} = \text{base} \times \prod_i f_i, \qquad
    \text{manual rate} = \frac{\text{manual PMPM}}{\text{target loss ratio}}.

For a group, member-level demographic factors are aggregated to a single
relativity (membership-weighted) before composing with group-level factors
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


def manual_pmpm(base_pmpm: float, factors: Sequence[float]) -> float:
    r"""Base PMPM scaled by the product of relativities."""
    require_positive(base_pmpm, "base_pmpm")
    return base_pmpm * product(factors)


def aggregate_demographic_factor(
    census: pd.DataFrame,
    factor_col: str,
    weight_col: str = "members",
) -> float:
    """Membership-weighted average of a member-level demographic factor."""
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
    base_pmpm : float
        Base claims cost PMPM at the rating-period level (see
        :func:`ratingmodels.base_rate_from_experience` to derive it).
    factors : mapping
        Named relativities, e.g. ``{"area": 1.05, "industry": 0.97, ...}``.
    target_loss_ratio : float
        Claims / premium target used to gross up to a charged rate. Ignored
        when ``retention`` is supplied.
    retention : RetentionLoad, optional
        Full expense / profit loading. When provided, the charged rate is built
        with the fundamental insurance equation instead of a single loss ratio,
        and fixed expense is applied per member (flat across cells).
    """

    base_pmpm: float
    factors: Mapping[str, float] = field(default_factory=dict)
    target_loss_ratio: float = 0.85
    retention: "RetentionLoad | None" = None

    def __post_init__(self) -> None:
        require_positive(self.base_pmpm, "base_pmpm")
        if self.retention is None:
            require_unit_interval(self.target_loss_ratio, "target_loss_ratio", closed=False)

    def total_relativity(self) -> float:
        return product(self.factors.values())

    def claims_pmpm(self) -> float:
        """Expected manual claims PMPM (before expense/margin loading)."""
        return self.base_pmpm * self.total_relativity()

    def steps(self) -> list:
        """The manual claims build-up as an ordered list of steps."""
        s = [start("Base claims cost", self.base_pmpm)]
        s += [multiply(name, factor) for name, factor in self.factors.items()]
        s.append(checkpoint("Manual claims PMPM"))
        return s

    def breakdown(self) -> "BuildUpResult":
        """Audit trail of the manual claims build-up (base x each relativity).

        The final running total equals :meth:`claims_pmpm` up to floating point.
        """
        return evaluate(self.steps())

    def rate(self) -> float:
        """Charged manual rate PMPM.

        Uses ``retention`` (the full gross-up) when supplied, otherwise
        ``claims PMPM / target_loss_ratio``.
        """
        if self.retention is not None:
            return self.retention.gross_rate(self.claims_pmpm())
        return self.claims_pmpm() / self.target_loss_ratio

    def with_factor(self, name: str, value: float) -> "ManualRate":
        new = dict(self.factors)
        new[name] = value
        return ManualRate(self.base_pmpm, new, self.target_loss_ratio)
