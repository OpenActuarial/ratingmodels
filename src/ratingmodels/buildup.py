r"""Ordered rate build-up with an audit trail.

Every group rate is assembled the same way: start from a base claim cost and
apply an ordered sequence of operations -- multiply by a relativity, add or
subtract a dollar amount (a copay credit, a per-unit fee), apply a factor to
only a segment of the cost -- recording labeled subtotals along the way, then
combine streams (in-/out-of-network by participation, medical + drug). This
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
* :func:`combine_streams`     -- additive combine (e.g. medical + drug).

Both return a :class:`BuildUpResult`, so intermediate results carry their own
breakdown and can be fed into credibility, trend, and retention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

_BREAKDOWN_COLUMNS = ["step", "operation", "label", "operand", "running_total"]


@dataclass(frozen=True)
class Step:
    """A single build-up operation. ``operand`` is the factor (multiply /
    segment), amount (add), or value (start); ``weight`` is used by
    ``segment_multiply`` only."""

    op: str
    label: str
    operand: float = 1.0
    weight: float = 1.0


def start(label: str, value: float) -> Step:
    """Set the running total to ``value`` (normally the first step)."""
    return Step("start", label, float(value))


def multiply(label: str, factor: float) -> Step:
    """Multiply the running total by ``factor`` (a relativity or trend)."""
    return Step("multiply", label, float(factor))


def add(label: str, amount: float) -> Step:
    """Add ``amount`` to the running total (negative for a copay credit)."""
    return Step("add", label, float(amount))


def segment_multiply(label: str, factor: float, weight: float) -> Step:
    r"""Apply ``factor`` to a fraction ``weight`` of the running total.

    :math:`\text{running} \leftarrow \text{running}\,(1 - w + w f)`.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    return Step("segment_multiply", label, float(factor), float(weight))


def checkpoint(label: str) -> Step:
    """Record a labeled subtotal without changing the running total."""
    return Step("checkpoint", label, float("nan"))


@dataclass
class BuildUpResult:
    """Result of evaluating a build-up.

    Attributes
    ----------
    value : float
        Final running total.
    breakdown : pandas.DataFrame
        One row per step: ``step, operation, label, operand, running_total``.
        For ``segment_multiply`` the ``operand`` shown is the *effective*
        factor :math:`(1 - w + w f)`, so the column reconciles by multiplication.
    subtotals : dict
        Ordered mapping of checkpoint label -> running total at that point.
    steps : list[Step]
        The raw steps (nominal factor and weight preserved).
    """

    value: float
    breakdown: pd.DataFrame
    subtotals: dict
    steps: list = field(default_factory=list, repr=False)

    def subtotal(self, label: str) -> float:
        """Running total recorded at the named checkpoint."""
        if label not in self.subtotals:
            raise KeyError(f"no checkpoint labeled {label!r}")
        return self.subtotals[label]

    def to_frame(self) -> pd.DataFrame:
        return self.breakdown

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BuildUpResult(value={self.value:.4f}, steps={len(self.steps)})"


def evaluate(steps: Sequence[Step]) -> BuildUpResult:
    """Run an ordered sequence of :class:`Step` and return a :class:`BuildUpResult`.

    The running total starts at 0; a leading :func:`start` sets the base.
    """
    running = 0.0
    rows = []
    subtotals: dict = {}
    for i, s in enumerate(steps, start=1):
        if s.op == "start":
            running = s.operand
            shown = s.operand
        elif s.op == "multiply":
            running *= s.operand
            shown = s.operand
        elif s.op == "add":
            running += s.operand
            shown = s.operand
        elif s.op == "segment_multiply":
            effective = 1.0 - s.weight + s.weight * s.operand
            running *= effective
            shown = effective
        elif s.op == "checkpoint":
            subtotals[s.label] = running
            shown = np.nan
        else:
            raise ValueError(f"unknown operation {s.op!r}")
        rows.append(
            {
                "step": i,
                "operation": s.op,
                "label": s.label,
                "operand": shown,
                "running_total": running,
            }
        )
    breakdown = pd.DataFrame(rows, columns=_BREAKDOWN_COLUMNS)
    return BuildUpResult(
        value=float(running),
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

    def start(self, label: str, value: float) -> "BuildUp":
        self._steps.append(start(label, value))
        return self

    def multiply(self, label: str, factor: float) -> "BuildUp":
        self._steps.append(multiply(label, factor))
        return self

    def add(self, label: str, amount: float) -> "BuildUp":
        self._steps.append(add(label, amount))
        return self

    def segment_multiply(self, label: str, factor: float, weight: float) -> "BuildUp":
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
ValueLike = Union[BuildUpResult, float]


def _val(x: ValueLike) -> float:
    return float(x.value) if isinstance(x, BuildUpResult) else float(x)


def combine_streams(
    streams: Mapping[str, ValueLike],
    label: str = "Combined",
) -> BuildUpResult:
    """Additively combine named streams (e.g. ``{"Medical": ..., "Drug": ...}``).

    Implemented as a build-up (start + adds) so the result carries a running
    total and an audit trail.
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
    participation_rate: float,
    label: str = "PPO Claim Cost",
) -> BuildUpResult:
    r"""In-/out-of-network blend :math:`\text{par}\,p + \text{nonpar}\,(1-p)`.

    ``participation_rate`` is the in-network (par) share ``p``.
    """
    if not 0.0 <= participation_rate <= 1.0:
        raise ValueError("participation_rate must lie in [0, 1]")
    p = participation_rate
    return combine_streams(
        {
            f"Par x {p:.1%}": _val(par) * p,
            f"Non-Par x {1 - p:.1%}": _val(nonpar) * (1 - p),
        },
        label=label,
    )
