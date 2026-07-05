r"""Pooling charges from a fitted severity model.

`experience_rate` takes ``pooled_excess`` and ``pooling_charge`` as inputs;
this module is where those numbers come from. Given any severity model
exposing the two-method tail protocol --

- ``sf(x)``: unconditional survival :math:`P(X > x)`
- ``mean_excess(d)``: :math:`E[X - d \mid X > d]`

-- the expected cost above a pooling point per unit of exposure is

.. math::
    \text{frequency} \times S(d) \times e(d)
    \;=\; \text{frequency} \times E[(X - d)_+],

grossed up for expenses and risk margin. ``lossmodels`` severity
distributions and ``extremeloss`` GPD tail fits both satisfy the protocol,
but *any* object with those two methods qualifies -- the seam is
duck-typed, and neither package is a dependency of this one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["pooling_charge_from_severity"]


def pooling_charge_from_severity(
    severity,
    pooling_point: float,
    expected_frequency: float,
    expense_ratio: float = 0.0,
    risk_margin: float = 0.0,
) -> pd.Series:
    r"""Expected pooling charge per exposure unit, decomposed.

    Parameters
    ----------
    severity
        Any object with ``sf(x)`` and ``mean_excess(d)`` (see module
        docstring). Both must accept a float and return a float.
    pooling_point : float
        The per-claim attachment ``d`` above which losses are pooled.
    expected_frequency : float
        Expected claims per exposure unit (the same exposure unit the
        charge should be quoted in).
    expense_ratio : float
        Expense provision as a share of the *charge*: the pure charge is
        divided by ``1 - expense_ratio``. Must lie in ``[0, 1)``.
    risk_margin : float
        Proportional loading on the pure excess cost, applied before the
        expense gross-up.

    Returns
    -------
    pandas.Series
        The build-up, each step auditable: ``exceedance_probability``
        (:math:`S(d)`), ``mean_excess`` (:math:`e(d)`),
        ``expected_excess_per_claim`` (:math:`S(d)\,e(d) = E[(X-d)_+]`),
        ``pure_excess_cost`` (frequency :math:`\times\; E[(X-d)_+]`), and
        ``pooling_charge`` (after margin and expense gross-up). The final
        value is what ``experience_rate`` expects as its
        ``pooling_charge`` input.

    Raises
    ------
    TypeError
        If ``severity`` lacks the protocol methods.
    ValueError
        For an infinite mean excess (a tail with :math:`\xi \ge 1` has no
        finite pooling cost at any attachment) or invalid loadings.
    """
    for method in ("sf", "mean_excess"):
        if not callable(getattr(severity, method, None)):
            raise TypeError(
                f"severity must expose callable {method!r}; got "
                f"{type(severity).__name__} (the protocol is sf + mean_excess)"
            )
    if pooling_point < 0:
        raise ValueError("pooling_point must be nonnegative")
    if expected_frequency < 0:
        raise ValueError("expected_frequency must be nonnegative")
    if not 0.0 <= expense_ratio < 1.0:
        raise ValueError("expense_ratio must be in [0, 1)")
    if risk_margin < 0:
        raise ValueError("risk_margin must be nonnegative")

    surv = float(severity.sf(float(pooling_point)))
    if not 0.0 <= surv <= 1.0:
        raise ValueError(f"severity.sf returned {surv!r}, outside [0, 1]")
    if surv == 0.0:
        excess_per_claim = 0.0
        me = 0.0
    else:
        me = float(severity.mean_excess(float(pooling_point)))
        if not np.isfinite(me):
            raise ValueError(
                "mean excess is not finite at this pooling point: the tail "
                "has no finite expected excess (e.g. a GPD with xi >= 1), so "
                "no finite pooling charge exists"
            )
        excess_per_claim = surv * me
    pure = expected_frequency * excess_per_claim
    charge = pure * (1.0 + risk_margin) / (1.0 - expense_ratio)
    return pd.Series(
        {
            "exceedance_probability": surv,
            "mean_excess": me,
            "expected_excess_per_claim": excess_per_claim,
            "pure_excess_cost": pure,
            "pooling_charge": charge,
        },
        name="pooling_charge_build_up",
    )
