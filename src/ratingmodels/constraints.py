r"""Constraints applied to indicated rates before they become rate actions.

Indicated rates are rarely charged as-is. Common adjustments:

* **Caps / floors** on the rate change to limit renewal shock.
* **Banding** -- snapping small changes to zero, or to discrete steps.
* **Rounding** to a filed precision.
* **Corridors** -- limiting how far a rate may move over successive renewals.
"""
from __future__ import annotations

import numpy as np


def cap_change(change: float, cap: float | None = None, floor: float | None = None) -> float:
    """Clip a proportional rate change to ``[floor, cap]`` (either may be None)."""
    out = float(change)
    if cap is not None:
        out = min(out, float(cap))
    if floor is not None:
        out = max(out, float(floor))
    return out


def apply_cap(
    current_rate: float,
    indicated_rate: float,
    cap: float | None = None,
    floor: float | None = None,
) -> float:
    """Return the charged rate after capping the implied change."""
    if current_rate <= 0:
        raise ValueError("current_rate must be positive")
    change = indicated_rate / current_rate - 1.0
    return current_rate * (1.0 + cap_change(change, cap, floor))


def band(change: float, deadband: float = 0.0, step: float | None = None) -> float:
    """Snap a change to zero within ``deadband``; optionally to ``step`` grid."""
    out = 0.0 if abs(change) <= deadband else float(change)
    if step is not None and step > 0:
        out = round(out / step) * step
    return out


def round_rate(rate: float, ndigits: int = 2) -> float:
    """Round a rate to a filed precision (default cents)."""
    return float(np.round(rate, ndigits))


def corridor(
    current_rate: float,
    indicated_rate: float,
    max_up: float,
    max_down: float,
) -> float:
    """Limit a single renewal move to ``[-max_down, +max_up]`` proportionally."""
    return apply_cap(current_rate, indicated_rate, cap=max_up, floor=-abs(max_down))
