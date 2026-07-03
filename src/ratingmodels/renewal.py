r"""Renewal actions: turn an indicated rate into a charged renewal rate.

A renewal action applies the indicated change, then the filed constraints
(caps, floors, rounding), and reports the realised change. A row-level
helper re-rates a census or schedule under new relativities and rolls up to
a book total.

:func:`renew` is elementwise under the vectorization contract: pass columns
of current and indicated rates (and, optionally, per-row caps/floors) and
the :class:`RenewalAction` fields come back as Series;
:meth:`RenewalAction.to_frame` lays the whole renewal run out as one tidy
DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils import (
    Numeric,
    first_series,
    is_arraylike,
    match_index,
    maybe_float,
    product,
    require_positive,
)
from .constraints import apply_cap, round_rate


@dataclass
class RenewalAction:
    """Result of :func:`renew`. Fields are floats for a scalar renewal and
    Series/arrays for a vectorized one."""

    current_rate: Numeric
    indicated_rate: Numeric
    proposed_rate: Numeric          # after caps/floors and rounding
    indicated_change: Numeric
    proposed_change: Numeric
    capped: "bool | np.ndarray | pd.Series"

    def to_dict(self) -> dict:
        return {
            "current_rate": self.current_rate,
            "indicated_rate": self.indicated_rate,
            "proposed_rate": self.proposed_rate,
            "indicated_change": self.indicated_change,
            "proposed_change": self.proposed_change,
            "capped": self.capped,
        }

    def to_frame(self) -> pd.DataFrame:
        """One tidy row per renewal (a single row for a scalar action)."""
        d = self.to_dict()
        if any(is_arraylike(v) for v in d.values()):
            return pd.DataFrame(d)
        return pd.DataFrame([d])


def renew(
    current_rate: Numeric,
    indicated_rate: Numeric,
    cap: Numeric | None = None,
    floor: Numeric | None = None,
    round_to: int | None = 2,
) -> RenewalAction:
    """Apply caps/floors (and optional rounding) to an indicated rate.

    Elementwise: Series in, Series-valued :class:`RenewalAction` out.
    ``cap`` and ``floor`` may be scalars or per-row vectors.
    """
    current_rate = require_positive(current_rate, "current_rate")
    indicated_rate = require_positive(indicated_rate, "indicated_rate")
    indicated_change = indicated_rate / current_rate - 1.0
    proposed = apply_cap(current_rate, indicated_rate, cap=cap, floor=floor)
    capped = ~np.isclose(np.asarray(proposed, dtype=float),
                         np.asarray(indicated_rate, dtype=float),
                         rtol=1e-9, atol=1e-9)
    if round_to is not None:
        proposed = round_rate(proposed, round_to)
    proposed_change = proposed / current_rate - 1.0
    template = first_series(current_rate, indicated_rate, cap, floor)
    if capped.ndim:
        capped_out = match_index(capped, template) if template is not None else capped
    else:
        capped_out = bool(capped[()])
    return RenewalAction(
        current_rate=maybe_float(current_rate),
        indicated_rate=maybe_float(indicated_rate),
        proposed_rate=maybe_float(proposed),
        indicated_change=maybe_float(indicated_change),
        proposed_change=maybe_float(proposed_change),
        capped=capped_out,
    )


def unit_level_renewal(
    census: pd.DataFrame,
    base_rate: Numeric,
    factor_cols: list[str],
    count_col: str = "count",
) -> pd.DataFrame:
    """Re-rate each census row as ``base_rate * product(factor_cols)``.

    Returns the census with ``unit_rate`` and ``premium`` columns; the group
    total is the sum of ``premium``. Fully vectorized -- ``base_rate`` may be
    a scalar or a per-row vector.
    """
    base_rate = require_positive(base_rate, "base_rate")
    out = census.copy()
    rel = product([out[c] for c in factor_cols]) if factor_cols else 1.0
    out["unit_rate"] = np.asarray(base_rate) * np.asarray(rel, dtype=float)
    out["premium"] = out["unit_rate"] * out[count_col].astype(float)
    return out
