r"""Retention and expense loading: turning a claims cost into a charged rate.

The charged (gross) rate is built from the **fundamental insurance equation**.
With loss & LAE per exposure unit :math:`L(1+\text{lae})`, a flat fixed
expense per unit :math:`F`, a variable load :math:`V` (expenses that are a percentage of
premium -- commission, premium tax, percent-of-premium fees and admin), and a
profit / contingency provision :math:`Q` (also a percentage of premium):

.. math::
    P = L(1+\text{lae}) + F + V P + Q P
      \;\Longrightarrow\;
    P = \frac{L(1+\text{lae}) + F}{1 - V - Q}.

The variable load sits in the denominator because premium tax (and commission)
are levied on the premium that already contains them. The target / permissible
loss ratio is then an **output**, not an input:

.. math::
    \text{PLR} = \frac{L}{P} = \frac{L\,(1 - V - Q)}{L(1+\text{lae}) + F}.

Because the fixed expense :math:`F` is added per exposure unit (not scaled by a risk's
relativity), grossing ``base_loss_cost * relativities`` with this formula keeps fixed
expense flat across all rate cells, which is the correct treatment.

Some contracts run the equation in reverse: a loss ratio is pinned by contract and the
premium is solved from it. Both standard pins are parameterizations of the same closed
form — :meth:`RetentionLoad.from_gross_loss_ratio` for :math:`C/P = \text{LR}^*` and
:meth:`RetentionLoad.from_net_loss_ratio` for :math:`C/(P - E) = \text{LR}^*`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ._utils import Numeric, maybe_float, require_nonnegative, require_unit_interval, safe_divide


@dataclass
class RetentionLoad:
    """Expense and profit loads used to gross claims up to a charged rate.

    Parameters
    ----------
    fixed_expense : float
        Flat operating expense per exposure unit (a dollar amount, not
        a percentage of premium). Default 0.
    variable_expense_ratio : float
        Sum of percent-of-premium loads: commission, premium tax, exchange /
        regulatory fees, and any admin expressed as a percentage of premium.
        Default 0.
    profit_margin : float
        Target underwriting profit / contribution to surplus, as a percentage
        of premium. Default 0.
    lae_ratio : float
        Loss adjustment expense as a percentage of claims. Default 0.
    """

    fixed_expense: Numeric = 0.0
    variable_expense_ratio: Numeric = 0.0
    profit_margin: Numeric = 0.0
    lae_ratio: Numeric = 0.0

    def __post_init__(self) -> None:
        self.fixed_expense = require_nonnegative(self.fixed_expense, "fixed_expense")
        self.variable_expense_ratio = require_nonnegative(
            self.variable_expense_ratio, "variable_expense_ratio"
        )
        self.profit_margin = require_nonnegative(self.profit_margin, "profit_margin")
        self.lae_ratio = require_nonnegative(self.lae_ratio, "lae_ratio")
        if np.any(np.asarray(self.variable_and_profit) >= 1.0):
            raise ValueError(
                "variable_expense_ratio + profit_margin must be < 1; "
                "the rate would be undefined or negative"
            )

    @classmethod
    def from_items(
        cls,
        fixed_expense: float = 0.0,
        variable_items: Mapping[str, float] | None = None,
        profit_margin: float = 0.0,
        lae_ratio: float = 0.0,
    ) -> "RetentionLoad":
        """Construct from an itemized mapping of percent-of-premium loads.

        ``variable_items`` (e.g. ``{"commission": 0.04, "premium_tax": 0.023,
        "aca_fees": 0.005, "admin_pct": 0.06}``) is summed into the variable
        expense ratio.
        """
        total_variable = maybe_float(sum((variable_items or {}).values()))
        return cls(
            fixed_expense=fixed_expense,
            variable_expense_ratio=total_variable,
            profit_margin=profit_margin,
            lae_ratio=lae_ratio,
        )

    @classmethod
    def from_gross_loss_ratio(
        cls,
        loss_ratio: Numeric,
        variable_items: Mapping[str, float] | None = None,
    ) -> "RetentionLoad":
        r"""Retention for a contract that pins the gross loss ratio.

        The contract fixes :math:`C/P = \text{LR}^*`, so the premium is fully
        determined by claims: :math:`P = C/\text{LR}^*`. In the fundamental
        equation this is :math:`F = 0` with the whole percent-of-premium
        retention pinned at :math:`V + Q = 1 - \text{LR}^*`.

        Parameters
        ----------
        loss_ratio : float or array-like
            The contractual claims / premium ratio, in (0, 1). A Series rates
            a book of pinned-ratio groups elementwise.
        variable_items : mapping, optional
            Known percent-of-premium components inside the retention (e.g.
            ``{"commission": 0.03, "premium_tax": 0.023}``). Itemizing does
            not change the premium — the contract pins the total — it only
            splits the retention: the remainder ``(1 - loss_ratio) -
            sum(items)`` lands in ``profit_margin``. Items exceeding the
            contractual retention raise, since the contract cannot cover them.

        Notes
        -----
        Dollar expenses (a flat fee per exposure unit) do not belong here: a
        gross-ratio contract leaves no degree of freedom for them to move the
        premium. Project them separately and reconcile against the retention
        :math:`P(1 - \text{LR}^*)`; what remains is the margin.
        """
        loss_ratio = require_unit_interval(loss_ratio, "loss_ratio", closed=False)
        total_variable = maybe_float(sum((variable_items or {}).values()))
        profit = maybe_float(1.0 - loss_ratio - total_variable)
        if np.any(np.asarray(profit) < 0):
            raise ValueError(
                "variable_items total exceeds the contractual retention share "
                "1 - loss_ratio; the contract cannot cover the named loads"
            )
        return cls(variable_expense_ratio=total_variable, profit_margin=profit)

    @classmethod
    def from_net_loss_ratio(
        cls,
        loss_ratio: Numeric,
        fixed_expense: Numeric = 0.0,
        variable_items: Mapping[str, float] | None = None,
    ) -> "RetentionLoad":
        r"""Retention for a contract that pins the loss ratio net of expenses.

        The contract fixes :math:`C/(P - E) = \text{LR}^*` with expenses
        :math:`E = F + V P`. Solving:

        .. math::
            P - F - V P = C/\text{LR}^*
            \;\Longrightarrow\;
            P = \frac{C/\text{LR}^* + F}{1 - V}.

        The claims gross-up :math:`1/\text{LR}^*` is carried through the
        percent-of-claims slot — :math:`C\,(1 + \tfrac{1-\text{LR}^*}{\text{LR}^*})
        = C/\text{LR}^*` — so ``lae_ratio`` on the returned instance holds
        :math:`(1-\text{LR}^*)/\text{LR}^*`, not a loss adjustment expense. If
        the contract's claims measure includes LAE, pass ``loss_cost``
        inclusive of LAE rather than setting ``lae_ratio``.

        Parameters
        ----------
        loss_ratio : float or array-like
            The contractual claims / (premium − expenses) ratio, in (0, 1).
        fixed_expense : float or array-like
            Dollar expenses per exposure unit netted out by the contract
            (e.g. a flat admin fee). Default 0.
        variable_items : mapping, optional
            Percent-of-premium expenses netted out by the contract, summed
            into :math:`V`.

        Notes
        -----
        The margin under this contract is claims-proportional:
        :math:`P - E - C = C\,(1-\text{LR}^*)/\text{LR}^*` — in contrast with
        the gross form, where expenses plus margin are premium-proportional.
        :meth:`implied_net_loss_ratio` returns ``loss_ratio`` identically for
        instances built here.
        """
        loss_ratio = require_unit_interval(loss_ratio, "loss_ratio", closed=False)
        total_variable = maybe_float(sum((variable_items or {}).values()))
        return cls(
            fixed_expense=fixed_expense,
            variable_expense_ratio=total_variable,
            lae_ratio=maybe_float((1.0 - loss_ratio) / loss_ratio),
        )

    @property
    def variable_and_profit(self) -> float:
        """Combined percent-of-premium load :math:`V + Q`."""
        return self.variable_expense_ratio + self.profit_margin

    def gross_rate(self, loss_cost: Numeric) -> Numeric:
        r"""Gross a loss cost up to a charged rate via :math:`(L(1+\text{lae})+F)/(1-V-Q)`.

        Elementwise: a Series of loss costs (and/or Series-valued loads for
        per-row retention structures) returns a Series of charged rates.
        """
        loss_cost = require_nonnegative(loss_cost, "loss_cost")
        numerator = loss_cost * (1.0 + self.lae_ratio) + self.fixed_expense
        return maybe_float(numerator / (1.0 - self.variable_and_profit))

    def implied_loss_ratio(self, loss_cost: Numeric) -> Numeric:
        """Loss ratio implied at a given claims level (claims / gross rate).

        With a non-zero fixed expense this varies with the claims level; with
        only percentage loads it equals ``1 - variable_expense_ratio -
        profit_margin``.
        """
        loss_cost = require_nonnegative(loss_cost, "loss_cost")
        return safe_divide(loss_cost, self.gross_rate(loss_cost))

    def implied_net_loss_ratio(self, loss_cost: Numeric) -> Numeric:
        r"""Loss ratio net of expenses: :math:`C / (P - F - V P)`.

        Expenses are the fixed and variable loads; the profit provision is
        the carrier's and stays inside the denominator's premium. For a
        retention built with :meth:`from_net_loss_ratio` this returns the
        contractual ratio identically; for any other retention it is the
        net-basis counterpart of :meth:`implied_loss_ratio`.
        """
        loss_cost = require_nonnegative(loss_cost, "loss_cost")
        premium = self.gross_rate(loss_cost)
        expenses = self.fixed_expense + self.variable_expense_ratio * premium
        return safe_divide(loss_cost, premium - expenses)

    def expense_and_profit_ratio(self, loss_cost: Numeric) -> Numeric:
        """Share of the gross rate going to expense and profit (1 - loss ratio)."""
        return maybe_float(1.0 - self.implied_loss_ratio(loss_cost))


def gross_rate(loss_cost: Numeric, retention: RetentionLoad) -> Numeric:
    """Functional form of :meth:`RetentionLoad.gross_rate`."""
    return retention.gross_rate(loss_cost)


def permissible_loss_ratio(retention: RetentionLoad, loss_cost: Numeric) -> Numeric:
    """Functional form of :meth:`RetentionLoad.implied_loss_ratio`."""
    return retention.implied_loss_ratio(loss_cost)
