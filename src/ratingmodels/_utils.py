"""Internal helpers: validation and small numeric utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def as_float_array(x, name: str = "value") -> np.ndarray:
    """Coerce to a 1-D float array, raising a clear error on bad input."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def require_positive(x: float, name: str) -> float:
    x = float(x)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {x!r}")
    return x


def require_nonnegative(x: float, name: str) -> float:
    x = float(x)
    if not np.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be non-negative and finite, got {x!r}")
    return x


def require_unit_interval(x: float, name: str, *, closed: bool = True) -> float:
    x = float(x)
    lo_ok = (x >= 0) if closed else (x > 0)
    hi_ok = (x <= 1) if closed else (x < 1)
    if not (np.isfinite(x) and lo_ok and hi_ok):
        bound = "[0, 1]" if closed else "(0, 1)"
        raise ValueError(f"{name} must lie in {bound}, got {x!r}")
    return x


def product(values: Iterable[float]) -> float:
    """Numerically stable product via logs when all positive, else direct."""
    vals = list(values)
    if not vals:
        return 1.0
    arr = np.asarray(vals, dtype=float)
    if np.all(arr > 0):
        return float(np.exp(np.sum(np.log(arr))))
    return float(np.prod(arr))
