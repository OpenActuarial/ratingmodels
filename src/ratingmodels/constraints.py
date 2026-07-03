r"""Constraints applied to indicated rates before they become rate actions.

Indicated rates are rarely charged as-is. Common adjustments:

* **Caps / floors** on the rate change to limit renewal shock.
* **Banding** -- snapping small changes to zero, or to discrete steps.
* **Rounding** to a filed precision.
* **Corridors** -- limiting how far a rate may move over successive renewals.

Every function is elementwise under the vectorization contract: pass whole
columns of current and indicated rates and get a column of constrained rates
back, with the pandas index preserved.
"""
from __future__ import annotations

import numpy as np

from ._utils import (
    Numeric,
    first_series,
    match_index,
    maybe_float,
    require_positive,
)


def cap_change(
    change: Numeric, cap: Numeric | None = None, floor: Numeric | None = None
) -> Numeric:
    """Clip a proportional rate change to ``[floor, cap]`` (either may be None).

    ``cap`` and ``floor`` may themselves be vectors for per-row limits.
    """
    out = np.asarray(change, dtype=float)
    if cap is not None:
        out = np.minimum(out, np.asarray(cap, dtype=float))
    if floor is not None:
        out = np.maximum(out, np.asarray(floor, dtype=float))
    template = first_series(change, cap, floor)
    if out.ndim:
        return match_index(out, template) if template is not None else out
    return float(out[()])


def apply_cap(
    current_rate: Numeric,
    indicated_rate: Numeric,
    cap: Numeric | None = None,
    floor: Numeric | None = None,
) -> Numeric:
    """Return the charged rate after capping the implied change, elementwise."""
    current_rate = require_positive(current_rate, "current_rate")
    change = indicated_rate / current_rate - 1.0
    return maybe_float(current_rate * (1.0 + cap_change(change, cap, floor)))


def band(change: Numeric, deadband: float = 0.0, step: float | None = None) -> Numeric:
    """Snap a change to zero within ``deadband``; optionally to ``step`` grid."""
    arr = np.asarray(change, dtype=float)
    out = np.where(np.abs(arr) <= deadband, 0.0, arr)
    if step is not None and step > 0:
        out = np.round(out / step) * step
    return maybe_float(match_index(out, change) if out.ndim else out[()])


def round_rate(rate: Numeric, ndigits: int = 2) -> Numeric:
    """Round a rate to a filed precision (default cents), elementwise."""
    return maybe_float(np.round(rate, ndigits))


def corridor(
    current_rate: Numeric,
    indicated_rate: Numeric,
    max_up: float,
    max_down: float,
) -> Numeric:
    """Limit a single renewal move to ``[-max_down, +max_up]`` proportionally."""
    return apply_cap(current_rate, indicated_rate, cap=max_up, floor=-abs(max_down))
