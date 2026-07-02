r"""Renewal actions: turn an indicated rate into a charged renewal rate.

A renewal action applies the indicated change, then the filed constraints
(caps, floors, rounding), and reports the realised change. A row-level
helper re-rates a census or schedule under new relativities and rolls up to
a book total.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils import product, require_positive
from .constraints import apply_cap, round_rate


@dataclass
class RenewalAction:
    """Result of :func:`renew`."""

    current_rate: float
    indicated_rate: float
    proposed_rate: float          # after caps/floors and rounding
    indicated_change: float
    proposed_change: float
    capped: bool

    def to_dict(self) -> dict:
        return {
            "current_rate": self.current_rate,
            "indicated_rate": self.indicated_rate,
            "proposed_rate": self.proposed_rate,
            "indicated_change": self.indicated_change,
            "proposed_change": self.proposed_change,
            "capped": self.capped,
        }


def renew(
    current_rate: float,
    indicated_rate: float,
    cap: float | None = None,
    floor: float | None = None,
    round_to: int | None = 2,
) -> RenewalAction:
    """Apply caps/floors (and optional rounding) to an indicated rate."""
    require_positive(current_rate, "current_rate")
    require_positive(indicated_rate, "indicated_rate")
    indicated_change = indicated_rate / current_rate - 1.0
    proposed = apply_cap(current_rate, indicated_rate, cap=cap, floor=floor)
    if round_to is not None:
        proposed = round_rate(proposed, round_to)
    proposed_change = proposed / current_rate - 1.0
    capped = not np.isclose(proposed, indicated_rate, rtol=1e-9, atol=1e-9)
    return RenewalAction(
        current_rate=float(current_rate),
        indicated_rate=float(indicated_rate),
        proposed_rate=float(proposed),
        indicated_change=float(indicated_change),
        proposed_change=float(proposed_change),
        capped=bool(capped),
    )


def unit_level_renewal(
    census: pd.DataFrame,
    base_rate: float,
    factor_cols: list[str],
    count_col: str = "count",
) -> pd.DataFrame:
    """Re-rate each census row as ``base_rate * product(factor_cols)``.

    Returns the census with ``unit_rate`` and ``premium`` columns; the group
    total is the sum of ``premium``.
    """
    require_positive(base_rate, "base_rate")
    out = census.copy()
    rates = []
    for _, row in out.iterrows():
        rates.append(base_rate * product(row[c] for c in factor_cols))
    out["unit_rate"] = rates
    out["premium"] = out["unit_rate"] * out[count_col].astype(float)
    return out
