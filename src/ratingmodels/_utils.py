r"""Internal helpers: validation and small numeric utilities.

Vectorization contract
----------------------
Public numeric arguments documented as ``float or array-like`` accept
scalars, numpy arrays, pandas Series, and plain sequences interchangeably,
under one contract used throughout ratingmodels:

* **scalar in, float out** -- exact historical behavior;
* **vector in, vector out** -- elementwise over rows, so a whole book prices
  in one call; a pandas ``Series`` in gives a ``Series`` out with the index
  preserved, any other array-like gives a numpy array;
* **broadcasting** -- scalars and length-``n`` vectors mix freely;
* **elementwise validation** -- one bad row fails the whole call, and the
  error reports the first offending label/position.

Series inputs that are combined in one expression follow normal pandas
index alignment; draw them from a single DataFrame (or otherwise share an
index). Helpers that reduce *across* inputs (:func:`product`, the build-up
engine) check this explicitly and raise on mismatched indexes.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd

#: Accepted by vectorized numeric arguments.
Numeric = Union[float, int, np.ndarray, pd.Series, Sequence[float]]


def is_arraylike(x) -> bool:
    """True for vector inputs (ndarray / Series / list / tuple), False for
    scalars, strings, and mappings."""
    if isinstance(x, (np.ndarray, pd.Series, pd.Index)):
        return np.ndim(x) > 0
    if isinstance(x, (str, bytes, dict)):
        return False
    return isinstance(x, (list, tuple)) or (
        hasattr(x, "__len__") and hasattr(x, "__getitem__")
    )


def as_numeric(x, name: str = "value") -> Union[float, np.ndarray, pd.Series]:
    """Coerce to the working numeric type: float for scalars, float Series
    for Series (index kept), float ndarray for any other array-like."""
    if isinstance(x, pd.Series):
        return x.astype(float)
    if is_arraylike(x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be scalar or 1-D, got shape {arr.shape}")
        return arr
    return float(x)


def maybe_float(x):
    """float() for scalars and 0-d arrays; vectors pass through unchanged."""
    if is_arraylike(x):
        return x
    return float(x)


def match_index(values, template):
    """Wrap ``values`` in a Series on ``template``'s index when the template
    is a Series; otherwise return ``values`` unchanged."""
    if isinstance(template, pd.Series):
        return pd.Series(np.asarray(values), index=template.index)
    return values


def first_series(*candidates):
    """The first pandas Series among ``candidates``, else None. Used to pick
    the index carrier for a result assembled in numpy."""
    for c in candidates:
        if isinstance(c, pd.Series):
            return c
    return None


def common_index(values: Iterable) -> "pd.Index | None":
    """Shared index of all Series in ``values`` (None if there are none).

    Raises ``ValueError`` if two Series carry different indexes -- reducing
    across silently misaligned rows is the classic vectorization bug.
    """
    idx = None
    for v in values:
        if isinstance(v, pd.Series):
            if idx is None:
                idx = v.index
            elif not v.index.equals(idx):
                raise ValueError(
                    "Series inputs must share one index; got "
                    f"{list(idx[:4])!r}... vs {list(v.index[:4])!r}..."
                )
    return idx


def _first_bad_label(bad: np.ndarray, x) -> str:
    pos = int(np.argmax(bad))
    if isinstance(x, pd.Series):
        return f"at index {x.index[pos]!r}"
    return f"at position {pos}"


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


def require_positive(x, name: str):
    """Validate ``x > 0`` (elementwise for vectors) and return the coerced
    numeric (float / float ndarray / float Series)."""
    out = as_numeric(x, name)
    if is_arraylike(out):
        arr = np.asarray(out, dtype=float)
        bad = ~np.isfinite(arr) | (arr <= 0)
        if np.any(bad):
            raise ValueError(
                f"{name} must be positive and finite; first violation "
                f"{_first_bad_label(bad, out)} (value {float(arr[np.argmax(bad)])!r})"
            )
        return out
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {out!r}")
    return out


def require_nonnegative(x, name: str):
    """Validate ``x >= 0`` (elementwise for vectors) and return the coerced
    numeric."""
    out = as_numeric(x, name)
    if is_arraylike(out):
        arr = np.asarray(out, dtype=float)
        bad = ~np.isfinite(arr) | (arr < 0)
        if np.any(bad):
            raise ValueError(
                f"{name} must be non-negative and finite; first violation "
                f"{_first_bad_label(bad, out)} (value {float(arr[np.argmax(bad)])!r})"
            )
        return out
    if not np.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be non-negative and finite, got {out!r}")
    return out


def require_unit_interval(x, name: str, *, closed: bool = True):
    """Validate ``x`` in [0, 1] (or (0, 1) when ``closed=False``),
    elementwise for vectors, and return the coerced numeric."""
    out = as_numeric(x, name)
    bound = "[0, 1]" if closed else "(0, 1)"
    if is_arraylike(out):
        arr = np.asarray(out, dtype=float)
        lo = (arr >= 0) if closed else (arr > 0)
        hi = (arr <= 1) if closed else (arr < 1)
        bad = ~np.isfinite(arr) | ~(lo & hi)
        if np.any(bad):
            raise ValueError(
                f"{name} must lie in {bound}; first violation "
                f"{_first_bad_label(bad, out)} (value {float(arr[np.argmax(bad)])!r})"
            )
        return out
    lo_ok = (out >= 0) if closed else (out > 0)
    hi_ok = (out <= 1) if closed else (out < 1)
    if not (np.isfinite(out) and lo_ok and hi_ok):
        raise ValueError(f"{name} must lie in {bound}, got {out!r}")
    return out


def safe_divide(numerator, denominator, fill: float = 0.0):
    """Elementwise ``numerator / denominator`` with ``fill`` where the
    denominator is zero. Type-preserving under the vectorization contract."""
    template = first_series(numerator, denominator)
    n = np.asarray(numerator, dtype=float)
    d = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(d == 0, fill, n / np.where(d == 0, 1.0, d))
    if template is not None:
        return pd.Series(out, index=template.index)
    return maybe_float(out if out.ndim else out[()])


def product(values: Iterable):
    r"""Product of ``values`` under the vectorization contract.

    Scalars only -> a single float, the plain product (numerically stable
    via logs when all inputs are positive). If any value is a length-``n``
    vector, the product reduces **across the factors** and keeps the rows:
    ``product([f1, f2, f3])`` with Series ``f_k`` is the elementwise
    :math:`f_1 f_2 f_3`, returned as a Series on the shared index (or a
    numpy array when no Series is involved). Scalars broadcast.
    """
    vals = list(values)
    if not vals:
        return 1.0
    if any(is_arraylike(v) for v in vals):
        idx = common_index(vals)
        arrs = [np.asarray(v, dtype=float) for v in vals]
        n = max(a.shape[0] for a in arrs if a.ndim == 1)
        rows = [np.broadcast_to(a, (n,)) for a in arrs]
        mat = np.vstack(rows)
        if np.all(mat > 0):
            out = np.exp(np.sum(np.log(mat), axis=0))
        else:
            out = np.prod(mat, axis=0)
        if idx is not None:
            return pd.Series(out, index=idx)
        return out
    arr = np.asarray(vals, dtype=float)
    if np.all(arr > 0):
        return float(np.exp(np.sum(np.log(arr))))
    return float(np.prod(arr))
