r"""Pricing scenarios: evaluate a case at any rate action and report margin.

The rate indication answers one question -- *what action does the formula
say?* Management pricing asks the surrounding ones: what margin falls out at
the action actually **issued**, after **concessions**, at the **plan** action;
what action produces **zero** margin or a **target** margin; and what uniform
uplift to a book's actions holds the aggregate margin when the achieved
actions slip below formula. This module answers those with the same expense
algebra the indication already uses (:class:`ratingmodels.RetentionLoad`).

**Forward.**  At charged rate :math:`P` with claims :math:`L`, LAE ratio
``lae``, fixed expense :math:`F` PMPM, and variable load :math:`V` (percent of
premium):

.. math::

    \text{gross margin} = P - L(1+\text{lae}), \qquad
    \text{margin} = P(1 - V) - L(1+\text{lae}) - F.

Gross margin is the benefit-tier margin (admin excluded); ``margin`` is the
underwriting gain after retention expense. At the indicated rate the margin
ratio equals the retention's ``profit_margin`` exactly.

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
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._utils import require_nonnegative, require_positive, require_unit_interval
from .indication import RateIndication
from .loading import RetentionLoad

_MODES = ("multiplicative", "additive")


@dataclass(frozen=True)
class ScenarioOutcome:
    """Result of evaluating one case at one rate action.

    PMPM fields are always present. Dollar fields require ``member_months``
    on the evaluation and are ``None`` otherwise; ``expected_*`` fields
    additionally require ``persistency`` and are the renewal-probability-
    weighted expectations (the deterministic counterpart of a retention
    Bernoulli).
    """

    name: str | None
    rate_change: float
    premium_pmpm: float
    claims_pmpm: float
    benefit_pmpm: float
    admin_pmpm: float
    loss_ratio: float
    gross_margin_pmpm: float
    margin_pmpm: float
    margin_ratio: float
    member_months: float | None = None
    persistency: float | None = None
    premium: float | None = None
    gross_margin: float | None = None
    margin: float | None = None
    expected_premium: float | None = None
    expected_margin: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, one tidy row."""
        return {
            "scenario": self.name,
            "rate_change": self.rate_change,
            "premium_pmpm": self.premium_pmpm,
            "claims_pmpm": self.claims_pmpm,
            "benefit_pmpm": self.benefit_pmpm,
            "admin_pmpm": self.admin_pmpm,
            "loss_ratio": self.loss_ratio,
            "gross_margin_pmpm": self.gross_margin_pmpm,
            "margin_pmpm": self.margin_pmpm,
            "margin_ratio": self.margin_ratio,
            "member_months": self.member_months,
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
    claims_pmpm : float
        Expected claims PMPM over the rating period (trended, pooled,
        credibility-blended -- e.g. ``RateIndication.blended_claims_pmpm()``).
    current_rate : float
        Current charged rate PMPM that rate changes apply to.
    retention : RetentionLoad, optional
        Expense structure. When omitted, no expenses are modeled: ``margin``
        equals ``gross margin`` (premium less claims) and the inverse solve
        reduces to :math:`P = L / (1 - m)`.
    member_months : float, optional
        Rating-period exposure; enables dollar outputs.
    persistency : float in [0, 1], optional
        Renewal probability; enables ``expected_*`` outputs (premium and
        margin scaled by the probability the case is still on the books).
    """

    claims_pmpm: float
    current_rate: float
    retention: RetentionLoad | None = None
    member_months: float | None = None
    persistency: float | None = None

    def __post_init__(self) -> None:
        require_nonnegative(self.claims_pmpm, "claims_pmpm")
        require_positive(self.current_rate, "current_rate")
        if self.member_months is not None:
            require_positive(self.member_months, "member_months")
        if self.persistency is not None:
            require_unit_interval(self.persistency, "persistency")

    @classmethod
    def from_indication(
        cls,
        indication: RateIndication,
        *,
        member_months: float | None = None,
        persistency: float | None = None,
    ) -> "PricingEvaluation":
        """Adopt a :class:`RateIndication`'s blended claims, rate, and retention.

        With a retention on the indication, evaluating at
        ``indicated_rate_change()`` returns a margin ratio equal to the
        retention's ``profit_margin``. Without one the indication grosses by
        target loss ratio, expenses are unmodeled here, and margin equals
        gross margin.
        """
        return cls(
            claims_pmpm=indication.blended_claims_pmpm(),
            current_rate=indication.current_rate,
            retention=indication.retention,
            member_months=member_months,
            persistency=persistency,
        )

    # ----- expense algebra ----- #
    def _pieces(self) -> tuple[float, float, float]:
        """(benefit_pmpm, fixed_pmpm, variable_ratio) under the retention."""
        if self.retention is None:
            return self.claims_pmpm, 0.0, 0.0
        r = self.retention
        return self.claims_pmpm * (1.0 + r.lae_ratio), r.fixed_expense_pmpm, r.variable_expense_ratio

    # ----- forward ----- #
    def at(self, rate_change: float, *, name: str | None = None) -> ScenarioOutcome:
        """Evaluate the case at a given proportional rate change."""
        premium = self.current_rate * (1.0 + float(rate_change))
        require_positive(premium, "premium at rate_change")
        benefit, fixed, variable = self._pieces()
        admin = fixed + variable * premium
        gross_margin_pmpm = premium - benefit
        margin_pmpm = gross_margin_pmpm - admin
        mm = self.member_months
        p = self.persistency
        return ScenarioOutcome(
            name=name,
            rate_change=float(rate_change),
            premium_pmpm=premium,
            claims_pmpm=self.claims_pmpm,
            benefit_pmpm=benefit,
            admin_pmpm=admin,
            loss_ratio=self.claims_pmpm / premium,
            gross_margin_pmpm=gross_margin_pmpm,
            margin_pmpm=margin_pmpm,
            margin_ratio=margin_pmpm / premium,
            member_months=mm,
            persistency=p,
            premium=None if mm is None else premium * mm,
            gross_margin=None if mm is None else gross_margin_pmpm * mm,
            margin=None if mm is None else margin_pmpm * mm,
            expected_premium=None if mm is None or p is None else premium * mm * p,
            expected_margin=None if mm is None or p is None else margin_pmpm * mm * p,
        )

    # ----- inverse ----- #
    def premium_for_margin(self, target_margin: float) -> float:
        r"""Charged rate PMPM at which the margin ratio equals the target.

        Closed form: :math:`P = (L(1+\text{lae}) + F) / (1 - V - m)`. The
        target may be negative (a planned loss) but must satisfy
        :math:`m < 1 - V` for a positive, finite rate.
        """
        benefit, fixed, variable = self._pieces()
        m = float(target_margin)
        denominator = 1.0 - variable - m
        if not denominator > 0:
            raise ValueError(
                f"target_margin must be less than 1 - variable_expense_ratio "
                f"= {1.0 - variable:.6g}, got {m!r}"
            )
        numerator = benefit + fixed
        if numerator <= 0:
            raise ValueError(
                "premium_for_margin requires positive benefit or fixed "
                "expense; with both zero every rate yields the target"
            )
        return numerator / denominator

    def rate_change_for_margin(self, target_margin: float) -> float:
        """Proportional rate change that yields the target margin ratio."""
        return self.premium_for_margin(target_margin) / self.current_rate - 1.0

    def zero_margin_rate_change(self) -> float:
        """Rate change at which the underwriting margin is exactly zero."""
        return self.rate_change_for_margin(0.0)


def scenario_frame(
    cases: Mapping[Any, PricingEvaluation],
    scenarios: Mapping[str, float | Mapping[Any, float]],
) -> pd.DataFrame:
    """Evaluate named rate actions across cases into one tidy long table.

    Parameters
    ----------
    cases : Mapping[case_id, PricingEvaluation]
        The book, keyed however the caller identifies cases.
    scenarios : Mapping[str, float | Mapping[case_id, float]]
        Each scenario is a rate change: a single float applied to every
        case, or a per-case mapping. A per-case mapping must cover every
        case -- a missing action is an error, not a silent skip.

    Returns
    -------
    pd.DataFrame
        One row per ``(case, scenario)``: ``case``, ``scenario``,
        ``rate_change``, PMPM economics, and dollar / persistency-weighted
        columns where the evaluation carries exposure and persistency. Any
        summary view -- a cohort rollup, a key-case exhibit -- is a pivot or
        groupby of this table.
    """
    if not cases:
        raise ValueError("cases must contain at least one PricingEvaluation")
    if not scenarios:
        raise ValueError("scenarios must contain at least one rate change")
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
        "member_months",
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
    cases: Mapping[Any, PricingEvaluation],
    base_changes: Mapping[Any, float] | float,
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
    per-PMPM cost :math:`K_g = L_g(1+\text{lae}_g) + F_g`, variable load
    :math:`V_g`, and weight :math:`w_g` = member months (times persistency
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
    if not cases:
        raise ValueError("cases must contain at least one PricingEvaluation")
    m_star = float(target_margin)

    a = b = c = a_prime = c_prime = 0.0
    entries: list[tuple[PricingEvaluation, float, float]] = []
    for case_id, evaluation in cases.items():
        if isinstance(base_changes, Mapping):
            if case_id not in base_changes:
                raise KeyError(f"base_changes has no rate change for case {case_id!r}")
            change = float(base_changes[case_id])
        else:
            change = float(base_changes)
        weight = 1.0 if evaluation.member_months is None else float(evaluation.member_months)
        if weight_by_persistency and evaluation.persistency is not None:
            weight *= evaluation.persistency
        if weight <= 0:
            continue
        base_premium = evaluation.current_rate * (1.0 + change)
        require_positive(base_premium, f"base premium for case {case_id!r}")
        benefit, fixed, variable = evaluation._pieces()
        entries.append((evaluation, change, weight))
        a += weight * base_premium * (1.0 - variable)
        b += weight * (benefit + fixed)
        c += weight * base_premium
        a_prime += weight * evaluation.current_rate * (1.0 - variable)
        c_prime += weight * evaluation.current_rate
    if not entries:
        raise ValueError("all cases have zero weight; nothing to solve")
    if b <= 0:
        raise ValueError(
            "aggregate benefit and fixed expense are zero; the margin ratio "
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
