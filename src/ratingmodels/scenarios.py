r"""Pricing scenarios: evaluate a case at any rate action and report margin.

The rate indication answers one question -- *what action does the formula
say?* Management pricing asks the surrounding ones: what margin falls out at
the action actually **issued**, after **concessions**, at the **plan** action;
what action produces **zero** margin or a **target** margin; and what uniform
uplift to a book's actions holds the aggregate margin when the achieved
actions slip below formula. This module answers those with the same expense
algebra the indication already uses (:class:`ratingmodels.RetentionLoad`).

All rates and costs are per unit of exposure -- whatever the caller's unit is
(member months, policy months, earned exposures). Dollar outputs are the
per-unit figures times ``exposure``.

**Forward.**  At charged rate :math:`P` with loss cost :math:`L`, LAE ratio
``lae``, fixed expense :math:`F` per exposure unit, and variable load
:math:`V` (percent of premium):

.. math::

    \text{gross margin} = P - L(1+\text{lae}), \qquad
    \text{margin} = P(1 - V) - L(1+\text{lae}) - F.

Gross margin is the loss-tier margin (operating expense excluded);
``margin`` is the underwriting gain after retention expense. At the
indicated rate the margin ratio equals the retention's ``profit_margin``
exactly.

**Inverse.**  The rate that yields margin ratio :math:`m` has the same form
as the gross-up itself, with :math:`m` in place of the profit provision:

.. math::

    P(m) = \frac{L(1+\text{lae}) + F}{1 - V - m}.

Zero-margin and plan-target premiums are this solve at :math:`m = 0` and
:math:`m = m_{\text{plan}}`; the standard indication is the special case
:math:`m = Q`.

Scenario names ("issued", "net concession", "select", ...) are the caller's
vocabulary: this module evaluates named actions and returns tidy rows; what
the names mean is business context that stays outside the library.

Everything here follows the vectorization contract. A
:class:`PricingEvaluation` built from columns (Series of loss costs, current
rates, exposures...) is *the book*: :meth:`PricingEvaluation.at` evaluates
every case at once, :meth:`ScenarioOutcome.to_frame` lays the outcome out as
one tidy row per case, and :func:`scenario_frame` /
:func:`uplift_for_target_margin` accept such a vector evaluation directly in
place of a mapping of scalar cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pandas as pd

from ._utils import (
    Numeric,
    as_numeric,
    first_series,
    is_arraylike,
    maybe_float,
    require_nonnegative,
    require_positive,
    require_unit_interval,
)
from .indication import RateIndication
from .loading import RetentionLoad

_MODES = ("multiplicative", "additive")


@dataclass(frozen=True)
class ScenarioOutcome:
    """Result of evaluating one case at one rate action.

    Per-exposure fields are always present. Dollar fields require
    ``exposure`` on the evaluation and are ``None`` otherwise;
    ``expected_*`` fields additionally require ``persistency`` and are the
    renewal-probability-weighted expectations (the deterministic counterpart
    of a retention Bernoulli).
    """

    name: str | None
    rate_change: Numeric
    premium_rate: Numeric
    loss_cost: Numeric
    loss_and_lae: Numeric
    expense_rate: Numeric
    loss_ratio: Numeric
    gross_margin_rate: Numeric
    margin_rate: Numeric
    margin_ratio: Numeric
    exposure: Numeric | None = None
    persistency: Numeric | None = None
    premium: Numeric | None = None
    gross_margin: Numeric | None = None
    margin: Numeric | None = None
    expected_premium: Numeric | None = None
    expected_margin: Numeric | None = None

    def to_frame(self) -> pd.DataFrame:
        """Tidy view: one row per case (a single row for a scalar outcome).

        Vector outcomes take their row index from the evaluation's Series
        inputs; columns whose inputs were not supplied (``premium`` without
        ``exposure``, ...) are omitted.
        """
        d = {k: v for k, v in self.as_dict().items() if v is not None}
        values = list(d.values())
        if not any(is_arraylike(v) for v in values):
            return pd.DataFrame([d])
        template = first_series(*values)
        n = (
            len(template)
            if template is not None
            else max(np.asarray(v).shape[0] for v in values if is_arraylike(v))
        )
        index = template.index if template is not None else pd.RangeIndex(n)
        cols = {}
        for k, v in d.items():
            if is_arraylike(v):
                cols[k] = np.asarray(v)
            else:
                cols[k] = np.repeat(v, n)
        return pd.DataFrame(cols, index=index)

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, one tidy row."""
        return {
            "scenario": self.name,
            "rate_change": self.rate_change,
            "premium_rate": self.premium_rate,
            "loss_cost": self.loss_cost,
            "loss_and_lae": self.loss_and_lae,
            "expense_rate": self.expense_rate,
            "loss_ratio": self.loss_ratio,
            "gross_margin_rate": self.gross_margin_rate,
            "margin_rate": self.margin_rate,
            "margin_ratio": self.margin_ratio,
            "exposure": self.exposure,
            "persistency": self.persistency,
            "premium": self.premium,
            "gross_margin": self.gross_margin,
            "margin": self.margin,
            "expected_premium": self.expected_premium,
            "expected_margin": self.expected_margin,
        }


@dataclass
class PricingEvaluation:
    """A case's pricing state, evaluable at arbitrary rate actions.

    Parameters
    ----------
    loss_cost : float
        Expected loss cost per exposure unit over the rating period
        (trended, pooled, credibility-blended -- e.g.
        ``RateIndication.blended_loss_cost()``).
    current_rate : float
        Current charged rate per exposure unit that rate changes apply to.
    retention : RetentionLoad, optional
        Expense structure. When omitted, no expenses are modeled: ``margin``
        equals ``gross margin`` (premium less losses) and the inverse solve
        reduces to :math:`P = L / (1 - m)`.
    exposure : float, optional
        Rating-period exposure units; enables dollar outputs.
    persistency : float in [0, 1], optional
        Renewal probability; enables ``expected_*`` outputs (premium and
        margin scaled by the probability the case is still on the books).
    """

    loss_cost: Numeric
    current_rate: Numeric
    retention: RetentionLoad | None = None
    exposure: Numeric | None = None
    persistency: Numeric | None = None

    def __post_init__(self) -> None:
        self.loss_cost = require_nonnegative(self.loss_cost, "loss_cost")
        self.current_rate = require_positive(self.current_rate, "current_rate")
        if self.exposure is not None:
            self.exposure = require_positive(self.exposure, "exposure")
        if self.persistency is not None:
            self.persistency = require_unit_interval(self.persistency, "persistency")

    @classmethod
    def from_indication(
        cls,
        indication: RateIndication,
        *,
        exposure: float | None = None,
        persistency: float | None = None,
    ) -> "PricingEvaluation":
        """Adopt a :class:`RateIndication`'s blended loss cost, rate, and retention.

        With a retention on the indication, evaluating at
        ``indicated_rate_change()`` returns a margin ratio equal to the
        retention's ``profit_margin``. Without one the indication grosses by
        target loss ratio, expenses are unmodeled here, and margin equals
        gross margin.
        """
        return cls(
            loss_cost=indication.blended_loss_cost(),
            current_rate=indication.current_rate,
            retention=indication.retention,
            exposure=exposure,
            persistency=persistency,
        )

    # ----- expense algebra ----- #
    def _case_index(self) -> "pd.Index | None":
        """Row index of a vectorized evaluation (None for scalar cases)."""
        template = first_series(
            self.loss_cost, self.current_rate, self.exposure, self.persistency
        )
        if template is not None:
            return template.index
        for v in (self.loss_cost, self.current_rate, self.exposure, self.persistency):
            if is_arraylike(v):
                return pd.RangeIndex(np.asarray(v).shape[0])
        return None

    def _pieces(self) -> tuple[Numeric, Numeric, Numeric]:
        """(loss_and_lae, fixed_expense, variable_ratio) under the retention."""
        if self.retention is None:
            return self.loss_cost, 0.0, 0.0
        r = self.retention
        return self.loss_cost * (1.0 + r.lae_ratio), r.fixed_expense, r.variable_expense_ratio

    # ----- forward ----- #
    def at(self, rate_change: Numeric, *, name: str | None = None) -> ScenarioOutcome:
        """Evaluate the case at a given proportional rate change.

        Elementwise: with a vector evaluation and/or a vector of rate
        changes, every field of the outcome is a Series/array per case.
        """
        rate_change = as_numeric(rate_change, "rate_change")
        premium_rate = self.current_rate * (1.0 + rate_change)
        require_positive(premium_rate, "premium at rate_change")
        loss_and_lae, fixed, variable = self._pieces()
        expense_rate = fixed + variable * premium_rate
        gross_margin_rate = premium_rate - loss_and_lae
        margin_rate = gross_margin_rate - expense_rate
        units = self.exposure
        p = self.persistency
        return ScenarioOutcome(
            name=name,
            rate_change=maybe_float(rate_change),
            premium_rate=premium_rate,
            loss_cost=self.loss_cost,
            loss_and_lae=loss_and_lae,
            expense_rate=expense_rate,
            loss_ratio=maybe_float(self.loss_cost / premium_rate),
            gross_margin_rate=gross_margin_rate,
            margin_rate=margin_rate,
            margin_ratio=margin_rate / premium_rate,
            exposure=units,
            persistency=p,
            premium=None if units is None else premium_rate * units,
            gross_margin=None if units is None else gross_margin_rate * units,
            margin=None if units is None else margin_rate * units,
            expected_premium=None if units is None or p is None else premium_rate * units * p,
            expected_margin=None if units is None or p is None else margin_rate * units * p,
        )

    # ----- inverse ----- #
    def premium_for_margin(self, target_margin: Numeric) -> Numeric:
        r"""Charged rate (per exposure unit) at which the margin ratio equals the target.

        Closed form: :math:`P = (L(1+\text{lae}) + F) / (1 - V - m)`. The
        target may be negative (a planned loss) but must satisfy
        :math:`m < 1 - V` for a positive, finite rate.
        """
        loss_and_lae, fixed, variable = self._pieces()
        m = as_numeric(target_margin, "target_margin")
        denominator = 1.0 - variable - m
        if np.any(np.asarray(denominator) <= 0):
            bound = float(np.min(np.asarray(1.0 - variable)))
            raise ValueError(
                f"target_margin must be less than 1 - variable_expense_ratio "
                f"(= {bound:.6g} at its tightest) for every case"
            )
        numerator = loss_and_lae + fixed
        if np.any(np.asarray(numerator) <= 0):
            raise ValueError(
                "premium_for_margin requires positive loss cost or fixed "
                "expense; with both zero every rate yields the target"
            )
        return maybe_float(numerator / denominator)

    def rate_change_for_margin(self, target_margin: Numeric) -> Numeric:
        """Proportional rate change that yields the target margin ratio."""
        return maybe_float(self.premium_for_margin(target_margin) / self.current_rate - 1.0)

    def zero_margin_rate_change(self) -> Numeric:
        """Rate change at which the underwriting margin is exactly zero."""
        return self.rate_change_for_margin(0.0)


def _per_case_action(action, case_index: pd.Index, scenario_name: str) -> Numeric:
    """Resolve one scenario's action against a vector evaluation's index."""
    if isinstance(action, Mapping):
        s = pd.Series(action, dtype=float)
        missing = case_index.difference(s.index)
        if len(missing):
            raise KeyError(
                f"scenario {scenario_name!r} has no rate change for "
                f"case(s) {list(missing)!r}"
            )
        return s.reindex(case_index)
    return as_numeric(action, "rate_change")


def scenario_frame(
    cases: "Mapping[Any, PricingEvaluation] | PricingEvaluation",
    scenarios: Mapping[str, Numeric | Mapping[Any, float]],
) -> pd.DataFrame:
    """Evaluate named rate actions across cases into one tidy long table.

    Parameters
    ----------
    cases : Mapping[case_id, PricingEvaluation] or PricingEvaluation
        The book: either a mapping of scalar evaluations keyed however the
        caller identifies cases, or a single **vector** evaluation built
        from columns, whose Series index provides the case ids.
    scenarios : Mapping[str, float | array-like | Mapping[case_id, float]]
        Each scenario is a rate change: a single float applied to every
        case, a per-case vector aligned with a vector evaluation, or a
        per-case mapping. A per-case mapping must cover every case -- a
        missing action is an error, not a silent skip.

    Returns
    -------
    pd.DataFrame
        One row per ``(case, scenario)``: ``case``, ``scenario``,
        ``rate_change``, per-exposure economics, and dollar /
        persistency-weighted columns where the evaluation carries exposure
        and persistency. Any summary view -- a cohort rollup, a key-case
        exhibit -- is a pivot or groupby of this table.
    """
    if not scenarios:
        raise ValueError("scenarios must contain at least one rate change")
    if isinstance(cases, PricingEvaluation):
        case_index = cases._case_index()
        if case_index is None:
            case_index = pd.RangeIndex(1)
        frames = []
        for scenario_name, action in scenarios.items():
            change = _per_case_action(action, case_index, scenario_name)
            out = cases.at(change, name=scenario_name).to_frame()
            out.insert(0, "case", np.asarray(case_index))
            frames.append(out)
        return pd.concat(frames, ignore_index=True)
    if not cases:
        raise ValueError("cases must contain at least one PricingEvaluation")
    rows: list[dict[str, Any]] = []
    for scenario_name, action in scenarios.items():
        for case_id, evaluation in cases.items():
            if isinstance(action, Mapping):
                if case_id not in action:
                    raise KeyError(
                        f"scenario {scenario_name!r} has no rate change for "
                        f"case {case_id!r}"
                    )
                change = action[case_id]
            else:
                change = action
            row = evaluation.at(change, name=scenario_name).as_dict()
            rows.append({"case": case_id, **row})
    frame = pd.DataFrame(rows)
    optional = [
        "exposure",
        "persistency",
        "premium",
        "gross_margin",
        "margin",
        "expected_premium",
        "expected_margin",
    ]
    drop = [c for c in optional if frame[c].isna().all()]
    return frame.drop(columns=drop)


def uplift_for_target_margin(
    cases: "Mapping[Any, PricingEvaluation] | PricingEvaluation",
    base_changes: "Mapping[Any, float] | Numeric",
    target_margin: float,
    *,
    mode: str = "multiplicative",
    weight_by_persistency: bool = True,
) -> float:
    r"""Uniform uplift to a book's rate actions that holds an aggregate margin.

    Answers the exhibit input "to achieve the same target margin, rate
    actions must be X% higher": when achieved actions slip below formula
    (concessions, caps), this is the across-the-board adjustment that
    restores the book's aggregate margin ratio to the target.

    Let case :math:`g` have base premium :math:`P_g` (at its base change),
    per-unit cost :math:`K_g = L_g(1+\text{lae}_g) + F_g`, variable load
    :math:`V_g`, and weight :math:`w_g` = exposure units (times persistency
    when ``weight_by_persistency``). The aggregate margin ratio is

    .. math::

        m(P) = \frac{\sum_g w_g \left(P_g (1 - V_g) - K_g\right)}
                    {\sum_g w_g P_g},

    which is a ratio of functions **affine in the uplift**, so the solve is
    closed-form -- no iteration:

    * ``multiplicative`` -- new change :math:`a_g' = (1 + a_g)(1 + u) - 1`,
      so :math:`P_g(u) = P_g (1+u)` and with :math:`A = \sum w P (1-V)`,
      :math:`B = \sum w K`, :math:`C = \sum w P`:

      .. math:: 1 + u = \frac{B}{A - m^\* C}.

    * ``additive`` -- new change :math:`a_g' = a_g + u`, so
      :math:`P_g(u) = P_g + r_g u` with current rate :math:`r_g`, and with
      :math:`A' = \sum w r (1-V)`, :math:`C' = \sum w r`:

      .. math:: u = \frac{B + m^\* C - A}{A' - m^\* C'}.

    Returns the uplift ``u``. Feasibility (a positive solution exists and
    every resulting premium is positive) is validated with explicit errors.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    m_star = float(target_margin)
    if isinstance(cases, PricingEvaluation):
        return _uplift_vector(
            cases, base_changes, m_star,
            mode=mode, weight_by_persistency=weight_by_persistency,
        )
    if not cases:
        raise ValueError("cases must contain at least one PricingEvaluation")

    a = b = c = a_prime = c_prime = 0.0
    entries: list[tuple[PricingEvaluation, float, float]] = []
    for case_id, evaluation in cases.items():
        if isinstance(base_changes, Mapping):
            if case_id not in base_changes:
                raise KeyError(f"base_changes has no rate change for case {case_id!r}")
            change = float(base_changes[case_id])
        else:
            change = float(base_changes)
        weight = 1.0 if evaluation.exposure is None else float(evaluation.exposure)
        if weight_by_persistency and evaluation.persistency is not None:
            weight *= evaluation.persistency
        if weight <= 0:
            continue
        base_premium = evaluation.current_rate * (1.0 + change)
        require_positive(base_premium, f"base premium for case {case_id!r}")
        loss_and_lae, fixed, variable = evaluation._pieces()
        entries.append((evaluation, change, weight))
        a += weight * base_premium * (1.0 - variable)
        b += weight * (loss_and_lae + fixed)
        c += weight * base_premium
        a_prime += weight * evaluation.current_rate * (1.0 - variable)
        c_prime += weight * evaluation.current_rate
    if not entries:
        raise ValueError("all cases have zero weight; nothing to solve")
    if b <= 0:
        raise ValueError(
            "aggregate loss and fixed expense are zero; the margin ratio "
            "does not depend on the uplift"
        )

    if mode == "multiplicative":
        denominator = a - m_star * c
        if not denominator > 0:
            raise ValueError(
                f"target_margin {m_star!r} is not attainable by scaling "
                "these premiums: it is at or above the book's asymptotic "
                "margin ratio"
            )
        uplift = b / denominator - 1.0
    else:
        denominator = a_prime - m_star * c_prime
        if not denominator > 0:
            raise ValueError(
                f"target_margin {m_star!r} is not attainable by an additive "
                "uplift on these cases"
            )
        uplift = (b + m_star * c - a) / denominator

    for evaluation, change, _ in entries:
        if mode == "multiplicative":
            new_change = (1.0 + change) * (1.0 + uplift) - 1.0
        else:
            new_change = change + uplift
        if evaluation.current_rate * (1.0 + new_change) <= 0:
            raise ValueError(
                "solved uplift drives at least one case's premium non-positive"
            )
    return uplift


def _uplift_vector(
    evaluation: PricingEvaluation,
    base_changes: "Mapping[Any, float] | Numeric",
    m_star: float,
    *,
    mode: str,
    weight_by_persistency: bool,
) -> float:
    """Closed-form uplift solve over a vector :class:`PricingEvaluation`.

    Same algebra as the mapping form -- the sums simply run down the
    columns -- so the two paths agree to floating point.
    """
    case_index = evaluation._case_index()
    n = 1 if case_index is None else len(case_index)
    if isinstance(base_changes, Mapping):
        if case_index is None:
            raise ValueError(
                "a Mapping of base_changes needs a labeled vector evaluation"
            )
        change = _per_case_action(base_changes, case_index, "base_changes")
    else:
        change = as_numeric(base_changes, "base_changes")

    current = np.broadcast_to(np.asarray(evaluation.current_rate, dtype=float), (n,))
    change = np.broadcast_to(np.asarray(change, dtype=float), (n,))
    loss_and_lae, fixed, variable = evaluation._pieces()
    loss_and_lae = np.broadcast_to(np.asarray(loss_and_lae, dtype=float), (n,))
    fixed = np.broadcast_to(np.asarray(fixed, dtype=float), (n,))
    variable = np.broadcast_to(np.asarray(variable, dtype=float), (n,))

    if evaluation.exposure is None:
        weight = np.ones(n)
    else:
        weight = np.broadcast_to(np.asarray(evaluation.exposure, dtype=float), (n,)).copy()
    if weight_by_persistency and evaluation.persistency is not None:
        weight = weight * np.broadcast_to(
            np.asarray(evaluation.persistency, dtype=float), (n,)
        )
    mask = weight > 0
    if not np.any(mask):
        raise ValueError("all cases have zero weight; nothing to solve")

    w = weight[mask]
    cur = current[mask]
    chg = change[mask]
    base_premium = cur * (1.0 + chg)
    require_positive(base_premium, "base premium")
    ll, fx, var = loss_and_lae[mask], fixed[mask], variable[mask]

    a = float(np.sum(w * base_premium * (1.0 - var)))
    b = float(np.sum(w * (ll + fx)))
    c = float(np.sum(w * base_premium))
    a_prime = float(np.sum(w * cur * (1.0 - var)))
    c_prime = float(np.sum(w * cur))
    if b <= 0:
        raise ValueError(
            "aggregate loss and fixed expense are zero; the margin ratio "
            "does not depend on the uplift"
        )

    if mode == "multiplicative":
        denominator = a - m_star * c
        if not denominator > 0:
            raise ValueError(
                f"target_margin {m_star!r} is not attainable by scaling "
                "these premiums: it is at or above the book's asymptotic "
                "margin ratio"
            )
        uplift = b / denominator - 1.0
        new_change = (1.0 + chg) * (1.0 + uplift) - 1.0
    else:
        denominator = a_prime - m_star * c_prime
        if not denominator > 0:
            raise ValueError(
                f"target_margin {m_star!r} is not attainable by an additive "
                "uplift on these cases"
            )
        uplift = (b + m_star * c - a) / denominator
        new_change = chg + uplift
    if np.any(cur * (1.0 + new_change) <= 0):
        raise ValueError(
            "solved uplift drives at least one case's premium non-positive"
        )
    return float(uplift)
