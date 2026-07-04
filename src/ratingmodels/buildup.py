r"""Ordered rate build-up with an audit trail.

Every group rate is assembled the same way: start from a base claim cost and
apply an ordered sequence of operations -- multiply by a relativity, add or
subtract a dollar amount (a copay credit, a per-unit fee), apply a factor to
only a segment of the cost -- recording labeled subtotals along the way, then
combine streams by participation share or additively (a health book's
in-/out-of-network split and medical + drug are the classic cases). This
module provides that grammar; it ships **no factor values**. The numbers are
yours (filed tables, state amounts, vendor fees); the engine just applies them
and produces a reconciling, auditable breakdown.

Operations
----------
* ``start(label, value)``      -- set the running total.
* ``multiply(label, factor)``  -- ``running *= factor`` (a relativity / trend).
* ``add(label, amount)``       -- ``running += amount`` (copay credit < 0, fee > 0).
* ``segment_multiply(label, factor, weight)`` -- apply ``factor`` to a fraction
  ``weight`` of the running total:
  :math:`\text{running} \leftarrow \text{running}\,(1 - w + w f)`.
* ``checkpoint(label)``        -- record a labeled subtotal; total unchanged.

Combining streams
-----------------
* :func:`participation_blend` -- :math:`\text{par}\,p + \text{nonpar}\,(1-p)`.
* :func:`combine_streams`     -- additive combine (e.g. a health book's
  medical + drug).

Both return a :class:`BuildUpResult`, so intermediate results carry their own
breakdown and can be fed into credibility, trend, and retention.

Vectorized build-ups
--------------------
Every operand follows the vectorization contract: pass a Series (a column --
per-group bases, per-group area factors) anywhere a float is accepted and the
whole book builds up at once. ``value`` and each subtotal come back as a
Series on the shared index, and ``breakdown`` switches from one row per step
to tidy long format -- one row per ``(step, entity)`` -- so it pivots or
filters directly. Scalar operands broadcast; Series operands must share one
index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

from ._utils import (
    Numeric,
    as_numeric,
    common_index,
    is_arraylike,
    require_unit_interval,
)

_BREAKDOWN_COLUMNS = ["step", "operation", "label", "operand", "running_total"]
_BREAKDOWN_COLUMNS_VECTOR = ["step", "operation", "label", "entity", "operand", "running_total"]


@dataclass(frozen=True)
class Step:
    """A single build-up operation. ``operand`` is the factor (multiply /
    segment), amount (add), or value (start); ``weight`` is used by
    ``segment_multiply`` only. Operands may be scalars or vectors."""

    op: str
    label: str
    operand: Numeric = 1.0
    weight: Numeric = 1.0


def start(label: str, value: Numeric) -> Step:
    """Set the running total to ``value`` (normally the first step)."""
    return Step("start", label, as_numeric(value, "value"))


def multiply(label: str, factor: Numeric) -> Step:
    """Multiply the running total by ``factor`` (a relativity or trend)."""
    return Step("multiply", label, as_numeric(factor, "factor"))


def add(label: str, amount: Numeric) -> Step:
    """Add ``amount`` to the running total (negative for a copay credit)."""
    return Step("add", label, as_numeric(amount, "amount"))


def segment_multiply(label: str, factor: Numeric, weight: Numeric) -> Step:
    r"""Apply ``factor`` to a fraction ``weight`` of the running total.

    :math:`\text{running} \leftarrow \text{running}\,(1 - w + w f)`.
    """
    weight = require_unit_interval(weight, "weight")
    return Step("segment_multiply", label, as_numeric(factor, "factor"), weight)


def checkpoint(label: str) -> Step:
    """Record a labeled subtotal without changing the running total."""
    return Step("checkpoint", label, float("nan"))


@dataclass
class BuildUpResult:
    """Result of evaluating a build-up.

    Attributes
    ----------
    value : float or pandas.Series
        Final running total; a Series (index preserved) for a vectorized
        build-up.
    breakdown : pandas.DataFrame
        Scalar build-up: one row per step with columns ``step, operation,
        label, operand, running_total``. Vectorized build-up: tidy long
        format, one row per ``(step, entity)``, with an ``entity`` column
        carrying the shared Series index (or positions). For
        ``segment_multiply`` the ``operand`` shown is the *effective*
        factor :math:`(1 - w + w f)`, so the column reconciles by
        multiplication.
    subtotals : dict
        Ordered mapping of checkpoint label -> running total at that point
        (floats, or Series for a vectorized build-up).
    steps : list[Step]
        The raw steps (nominal factor and weight preserved).
    """

    value: Numeric
    breakdown: pd.DataFrame
    subtotals: dict
    steps: list = field(default_factory=list, repr=False)

    def subtotal(self, label: str) -> Numeric:
        """Running total recorded at the named checkpoint."""
        if label not in self.subtotals:
            raise KeyError(f"no checkpoint labeled {label!r}")
        return self.subtotals[label]

    def to_frame(self) -> pd.DataFrame:
        return self.breakdown

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if is_arraylike(self.value):
            v = np.asarray(self.value, dtype=float)
            return (
                f"BuildUpResult(n={v.size}, mean value={v.mean():.4f}, "
                f"steps={len(self.steps)})"
            )
        return f"BuildUpResult(value={self.value:.4f}, steps={len(self.steps)})"


def _vector_context(steps: Sequence[Step]):
    """(n, index) implied by any vector operands/weights, validating that
    all vectors agree on length and (for Series) on index."""
    vectors = [s.operand for s in steps if is_arraylike(s.operand)]
    vectors += [s.weight for s in steps if is_arraylike(s.weight)]
    if not vectors:
        return None, None
    idx = common_index(vectors)
    lengths = {np.asarray(v).shape[0] for v in vectors}
    if len(lengths) > 1:
        raise ValueError(f"vector operands must share one length, got {sorted(lengths)}")
    n = lengths.pop()
    if idx is not None and len(idx) != n:
        raise ValueError("vector operands must share one length")
    return n, idx


def evaluate(steps: Sequence[Step]) -> BuildUpResult:
    """Run an ordered sequence of :class:`Step` and return a :class:`BuildUpResult`.

    The running total starts at 0; a leading :func:`start` sets the base.
    Vector operands (Series / arrays) make the whole build-up elementwise;
    see the module notes on vectorized build-ups.
    """
    n, idx = _vector_context(steps)

    def _wrap(x):
        if n is None:
            return float(x)
        arr = np.broadcast_to(np.asarray(x, dtype=float), (n,))
        return pd.Series(arr, index=idx) if idx is not None else arr.copy()

    running = 0.0
    rows = []
    subtotals: dict = {}
    for i, s in enumerate(steps, start=1):
        operand = np.asarray(s.operand, dtype=float) if is_arraylike(s.operand) else s.operand
        weight = np.asarray(s.weight, dtype=float) if is_arraylike(s.weight) else s.weight
        if s.op == "start":
            running = operand
            shown = operand
        elif s.op == "multiply":
            running = running * operand
            shown = operand
        elif s.op == "add":
            running = running + operand
            shown = operand
        elif s.op == "segment_multiply":
            effective = 1.0 - weight + weight * operand
            running = running * effective
            shown = effective
        elif s.op == "checkpoint":
            subtotals[s.label] = _wrap(running)
            shown = np.nan
        else:
            raise ValueError(f"unknown operation {s.op!r}")
        if n is None:
            rows.append(
                {
                    "step": i,
                    "operation": s.op,
                    "label": s.label,
                    "operand": shown,
                    "running_total": running,
                }
            )
        else:
            shown_v = np.broadcast_to(np.asarray(shown, dtype=float), (n,))
            running_v = np.broadcast_to(np.asarray(running, dtype=float), (n,))
            entities = idx if idx is not None else np.arange(n)
            for e, o, r in zip(entities, shown_v, running_v):
                rows.append(
                    {
                        "step": i,
                        "operation": s.op,
                        "label": s.label,
                        "entity": e,
                        "operand": o,
                        "running_total": r,
                    }
                )
    columns = _BREAKDOWN_COLUMNS if n is None else _BREAKDOWN_COLUMNS_VECTOR
    breakdown = pd.DataFrame(rows, columns=columns)
    return BuildUpResult(
        value=_wrap(running) if n is not None else float(running),
        breakdown=breakdown,
        subtotals=subtotals,
        steps=list(steps),
    )


class BuildUp:
    """Fluent builder for a build-up; sugar over a list of :class:`Step`.

    >>> r = (BuildUp()
    ...      .start("Par Base", 941.63)
    ...      .add("$30 specialist copay", -11.44)
    ...      .multiply("Rating Region", 1.083)
    ...      .checkpoint("Medical Par Base Claim Cost")
    ...      .evaluate())
    """

    def __init__(self) -> None:
        self._steps: list[Step] = []

    def start(self, label: str, value: Numeric) -> "BuildUp":
        self._steps.append(start(label, value))
        return self

    def multiply(self, label: str, factor: Numeric) -> "BuildUp":
        self._steps.append(multiply(label, factor))
        return self

    def add(self, label: str, amount: Numeric) -> "BuildUp":
        self._steps.append(add(label, amount))
        return self

    def segment_multiply(self, label: str, factor: Numeric, weight: Numeric) -> "BuildUp":
        self._steps.append(segment_multiply(label, factor, weight))
        return self

    def checkpoint(self, label: str) -> "BuildUp":
        self._steps.append(checkpoint(label))
        return self

    def steps(self) -> list:
        return list(self._steps)

    def evaluate(self) -> BuildUpResult:
        return evaluate(self._steps)


# --------------------------------------------------------------------------- #
# combining streams
# --------------------------------------------------------------------------- #
ValueLike = Union[BuildUpResult, float, "pd.Series", np.ndarray]


def _val(x: ValueLike) -> Numeric:
    if isinstance(x, BuildUpResult):
        return x.value
    return as_numeric(x, "value")


def combine_streams(
    streams: Mapping[str, ValueLike],
    label: str = "Combined",
) -> BuildUpResult:
    """Additively combine named streams (e.g. ``{"Medical": ..., "Drug": ...}``).

    Implemented as a build-up (start + adds) so the result carries a running
    total and an audit trail. Vector-valued streams combine elementwise.
    """
    items = list(streams.items())
    if not items:
        raise ValueError("provide at least one stream")
    steps = [start(items[0][0], _val(items[0][1]))]
    for lab, v in items[1:]:
        steps.append(add(lab, _val(v)))
    steps.append(checkpoint(label))
    return evaluate(steps)


def participation_blend(
    par: ValueLike,
    nonpar: ValueLike,
    participation_rate: Numeric,
    label: str = "Blended Claim Cost",
) -> BuildUpResult:
    r"""Two-stream participation blend :math:`\text{par}\,p + \text{nonpar}\,(1-p)`
    (e.g. a health book's in-/out-of-network split).

    ``participation_rate`` is the participating share ``p``; it may be a
    Series for per-row participation.
    """
    p = require_unit_interval(participation_rate, "participation_rate")
    if is_arraylike(p):
        par_label, nonpar_label = "Par x participation", "Non-Par x (1 - participation)"
    else:
        par_label, nonpar_label = f"Par x {p:.1%}", f"Non-Par x {1 - p:.1%}"
    return combine_streams(
        {
            par_label: _val(par) * p,
            nonpar_label: _val(nonpar) * (1 - p),
        },
        label=label,
    )
