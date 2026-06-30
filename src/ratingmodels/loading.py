r"""Retention and expense loading: turning a claims cost into a charged rate.

The charged (gross) rate is built from the **fundamental insurance equation**.
With loss & LAE per member :math:`L(1+\text{lae})`, a flat fixed expense per
member :math:`F`, a variable load :math:`V` (expenses that are a percentage of
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

Because the fixed expense :math:`F` is added per member (not scaled by a risk's
relativity), grossing ``base_pmpm * relativities`` with this formula keeps fixed
expense flat across all rate cells, which is the correct treatment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ._utils import require_nonnegative


@dataclass
class RetentionLoad:
    """Expense and profit loads used to gross claims up to a charged rate.

    Parameters
    ----------
    fixed_expense_pmpm : float
        Flat administrative expense per member per month (a dollar amount, not
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

    fixed_expense_pmpm: float = 0.0
    variable_expense_ratio: float = 0.0
    profit_margin: float = 0.0
    lae_ratio: float = 0.0

    def __post_init__(self) -> None:
        require_nonnegative(self.fixed_expense_pmpm, "fixed_expense_pmpm")
        require_nonnegative(self.variable_expense_ratio, "variable_expense_ratio")
        require_nonnegative(self.profit_margin, "profit_margin")
        require_nonnegative(self.lae_ratio, "lae_ratio")
        if self.variable_and_profit >= 1.0:
            raise ValueError(
                "variable_expense_ratio + profit_margin must be < 1; "
                "the rate would be undefined or negative"
            )

    @classmethod
    def from_items(
        cls,
        fixed_expense_pmpm: float = 0.0,
        variable_items: Mapping[str, float] | None = None,
        profit_margin: float = 0.0,
        lae_ratio: float = 0.0,
    ) -> "RetentionLoad":
        """Construct from an itemized mapping of percent-of-premium loads.

        ``variable_items`` (e.g. ``{"commission": 0.04, "premium_tax": 0.023,
        "aca_fees": 0.005, "admin_pct": 0.06}``) is summed into the variable
        expense ratio.
        """
        total_variable = float(sum((variable_items or {}).values()))
        return cls(
            fixed_expense_pmpm=fixed_expense_pmpm,
            variable_expense_ratio=total_variable,
            profit_margin=profit_margin,
            lae_ratio=lae_ratio,
        )

    @property
    def variable_and_profit(self) -> float:
        """Combined percent-of-premium load :math:`V + Q`."""
        return self.variable_expense_ratio + self.profit_margin

    def gross_rate(self, claims_pmpm: float) -> float:
        r"""Gross a claims PMPM up to a charged rate via :math:`(L(1+\text{lae})+F)/(1-V-Q)`."""
        require_nonnegative(claims_pmpm, "claims_pmpm")
        numerator = claims_pmpm * (1.0 + self.lae_ratio) + self.fixed_expense_pmpm
        return numerator / (1.0 - self.variable_and_profit)

    def implied_loss_ratio(self, claims_pmpm: float) -> float:
        """Loss ratio implied at a given claims level (claims / gross rate).

        With a non-zero fixed expense this varies with the claims level; with
        only percentage loads it equals ``1 - variable_expense_ratio -
        profit_margin``.
        """
        require_nonnegative(claims_pmpm, "claims_pmpm")
        if claims_pmpm == 0:
            return 0.0
        return claims_pmpm / self.gross_rate(claims_pmpm)

    def expense_and_profit_ratio(self, claims_pmpm: float) -> float:
        """Share of the gross rate going to expense and profit (1 - loss ratio)."""
        return 1.0 - self.implied_loss_ratio(claims_pmpm)


def gross_rate(claims_pmpm: float, retention: RetentionLoad) -> float:
    """Functional form of :meth:`RetentionLoad.gross_rate`."""
    return retention.gross_rate(claims_pmpm)


def permissible_loss_ratio(retention: RetentionLoad, claims_pmpm: float) -> float:
    """Functional form of :meth:`RetentionLoad.implied_loss_ratio`."""
    return retention.implied_loss_ratio(claims_pmpm)
