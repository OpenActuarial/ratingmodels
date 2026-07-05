r"""The implemented rating plan: base rate x factor tables, as one object.

`FactorTable` holds one rating variable; the build-up engine applies a
sequence of adjustments; the GLM and frequency-severity models *produce*
factor tables. :class:`RatingPlan` is the object those pieces were pointing
at: the complete multiplicative plan -- a base rate and a table per rating
variable -- that can rate a book, audit itself against a census, round-trip
through a dict for filing and version control, and be compared against a
successor with :func:`compare_rating_plans`.

Unknown levels are a production hazard, not an edge case: a census with a
territory the plan has never seen should be a *decision*, not a silent
factor of 1.0. ``plan.rate(..., unknown="error")`` makes it a hard stop;
``unknown="default"`` applies each table's default (typically 1.0) after
:meth:`RatingPlan.validate` has shown you exactly what would be defaulted.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .dislocation import rate_dislocation
from .relativity import FactorTable

__all__ = ["RatingPlan", "PlanComparison", "compare_rating_plans"]


@dataclass
class RatingPlan:
    """A complete multiplicative rating plan.

    Parameters
    ----------
    base_rate : float
        The rate at base levels of every variable (per exposure unit).
    factors : mapping of str -> FactorTable
        One table per rating variable, keyed by variable name.

    Notes
    -----
    ``RatingPlan.from_model(model)`` builds a plan directly from a fitted
    :class:`~ratingmodels.GLMRelativities` or
    :class:`~ratingmodels.FrequencySeverityModel` -- ``to_factor_tables()``
    supplies the factors and ``base_value_`` the base rate.
    """

    base_rate: float
    factors: Mapping[str, FactorTable] = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite(self.base_rate) or self.base_rate <= 0:
            raise ValueError("base_rate must be a positive finite number")
        self.factors = dict(self.factors)
        for var, tab in self.factors.items():
            if not isinstance(tab, FactorTable):
                raise TypeError(f"factors[{var!r}] must be a FactorTable")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_model(cls, model, base_rate: float | None = None) -> "RatingPlan":
        """Build a plan from a fitted model.

        ``model`` needs ``to_factor_tables()`` and ``base_value_`` -- both
        :class:`~ratingmodels.GLMRelativities` and
        :class:`~ratingmodels.FrequencySeverityModel` qualify. ``base_rate``
        overrides the fitted base (e.g. after an off-balance correction).
        """
        infos = []
        info = getattr(model, "_design_info_", None)
        if info is not None:
            infos.append(info)
        for sub in ("frequency", "severity"):
            m = getattr(model, sub, None)
            sub_info = getattr(m, "_design_info_", None) if m is not None else None
            if sub_info is not None:
                infos.append(sub_info)
        unrepresentable = set()
        for i in infos:
            if i.get("continuous"):
                unrepresentable.add("continuous covariates")
            if i.get("interactions"):
                unrepresentable.add("interaction terms")
        if unrepresentable:
            warnings.warn(
                f"model has {' and '.join(sorted(unrepresentable))} that a "
                "RatingPlan's single-variable factor tables cannot represent; "
                "plan rates will differ from model.predict for those terms",
                stacklevel=2,
            )
        return cls(
            base_rate=float(model.base_value_ if base_rate is None else base_rate),
            factors=model.to_factor_tables(),
        )

    # ------------------------------------------------------------------ #
    def _resolve_columns(self, data: pd.DataFrame, columns) -> dict:
        columns = dict(columns or {})
        resolved = {}
        for var in self.factors:
            col = columns.get(var, var)
            if col not in data.columns:
                raise ValueError(
                    f"column {col!r} for rating variable {var!r} not found"
                )
            resolved[var] = col
        return resolved

    def validate(self, data: pd.DataFrame, columns: Mapping | None = None) -> pd.DataFrame:
        """Levels present in ``data`` that the plan has no factor for.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(variable, level)`` with column ``n`` (row count).
            Empty means every level is covered. Run this before rating a new
            census; anything listed here is what ``unknown="default"`` would
            silently default and ``unknown="error"`` would refuse.
        """
        cols = self._resolve_columns(data, columns)
        rows, index = [], []
        for var, col in cols.items():
            known = set(self.factors[var].factors)
            counts = data[col].value_counts()
            for lvl, n in counts.items():
                if lvl not in known:
                    rows.append(int(n))
                    index.append((var, lvl))
        return pd.DataFrame(
            {"n": rows},
            index=pd.MultiIndex.from_tuples(index, names=["variable", "level"])
            if index
            else pd.MultiIndex.from_arrays([[], []], names=["variable", "level"]),
        )

    def rate(
        self,
        data: pd.DataFrame,
        columns: Mapping | None = None,
        exposure: str | None = None,
        unknown: str = "default",
    ) -> pd.DataFrame:
        """Rate every row: the full multiplicative build-up, decomposed.

        Parameters
        ----------
        data : DataFrame
            One row per unit to rate.
        columns : mapping, optional
            Rating variable -> column name, where names differ.
        exposure : str, optional
            Exposure column; adds a ``premium`` column (rate x exposure).
        unknown : {"default", "error"}
            Policy for levels the plan has no factor for. ``"default"``
            applies the table's default; ``"error"`` raises, listing every
            offending ``(variable, level)``.

        Returns
        -------
        pandas.DataFrame
            Index-aligned with ``data``: ``base_rate``, one
            ``{variable}_factor`` per variable, ``combined_relativity``,
            ``rate``, and ``premium`` when ``exposure`` is given.
        """
        if unknown not in ("default", "error"):
            raise ValueError('unknown must be "default" or "error"')
        cols = self._resolve_columns(data, columns)
        if unknown == "error":
            bad = self.validate(data, columns)
            if len(bad):
                listing = ", ".join(
                    f"{v}={lvl!r} (n={int(n)})"
                    for (v, lvl), n in bad["n"].items()
                )
                raise ValueError(f"unknown levels in census: {listing}")

        out = pd.DataFrame(index=data.index)
        out["base_rate"] = float(self.base_rate)
        combined = np.ones(len(data))
        for var, col in cols.items():
            fac = self.factors[var].apply(data[col]).to_numpy(dtype=float)
            out[f"{var}_factor"] = fac
            combined = combined * fac
        out["combined_relativity"] = combined
        out["rate"] = self.base_rate * combined
        if exposure is not None:
            out["premium"] = out["rate"] * data[exposure].to_numpy(dtype=float)
        return out

    def average_relativity(
        self,
        data: pd.DataFrame,
        columns: Mapping | None = None,
        exposure: str | None = None,
    ) -> pd.Series:
        """Exposure-weighted average factor per variable, and combined.

        The plan's off-balance diagnostic: a combined average of 1.0 means
        the factors are balanced on this census; anything else is what a
        base-rate correction would need to absorb.
        """
        rated = self.rate(data, columns=columns, exposure=exposure)
        w = (
            data[exposure].to_numpy(dtype=float)
            if exposure is not None
            else np.ones(len(data))
        )
        if w.sum() <= 0:
            raise ValueError("total exposure must be positive")
        out = {}
        for var in self.factors:
            out[var] = float(np.average(rated[f"{var}_factor"], weights=w))
        out["combined"] = float(np.average(rated["combined_relativity"], weights=w))
        return pd.Series(out, name="average_relativity")

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """A plain-dict form for filing, audit, and version control.

        Round-trips exactly through :meth:`from_dict`. If the dict will
        pass through JSON, note that JSON object keys are always strings:
        non-string level keys (e.g. integer territory codes) come back as
        strings, and lookups against the original typed levels will then
        fall to the default. Use string levels for JSON-borne plans.
        """
        return {
            "schema": 1,
            "base_rate": float(self.base_rate),
            "factors": {
                var: {
                    "factors": {lvl: float(f) for lvl, f in tab.factors.items()},
                    "default": float(tab.default),
                }
                for var, tab in self.factors.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "RatingPlan":
        """Rebuild a plan from :meth:`to_dict` output."""
        if d.get("schema") != 1:
            raise ValueError(f"unsupported RatingPlan schema: {d.get('schema')!r}")
        factors = {
            var: FactorTable(
                name=var,
                factors=dict(spec["factors"]),
                default=float(spec.get("default", 1.0)),
            )
            for var, spec in d["factors"].items()
        }
        return cls(base_rate=float(d["base_rate"]), factors=factors)


@dataclass
class PlanComparison:
    """Per-case comparison of two rating plans on one census."""

    current_rate: pd.Series = field(repr=False)
    proposed_rate: pd.Series = field(repr=False)
    exposure: pd.Series = field(repr=False)

    @property
    def change(self) -> pd.Series:
        """Per-case rate change, ``proposed/current - 1``."""
        return (self.proposed_rate / self.current_rate - 1.0).rename("change")

    def summary(self) -> pd.Series:
        """The one-screen comparison: premiums, average change, direction."""
        w = self.exposure.to_numpy(dtype=float)
        cur = float((self.current_rate * self.exposure).sum())
        prop = float((self.proposed_rate * self.exposure).sum())
        ch = self.change.to_numpy()
        total_w = w.sum()
        return pd.Series(
            {
                "n": int(len(w)),
                "exposure": float(total_w),
                "current_premium": cur,
                "proposed_premium": prop,
                "avg_change": prop / cur - 1.0 if cur > 0 else np.nan,
                "share_increasing": float(w[ch > 0].sum() / total_w),
                "share_decreasing": float(w[ch < 0].sum() / total_w),
                "share_unchanged": float(w[ch == 0].sum() / total_w),
            },
            name="plan_comparison",
        )

    def dislocation(self, bands=(-0.10, -0.05, 0.0, 0.05, 0.10)) -> pd.DataFrame:
        """The banded dislocation exhibit; see :func:`rate_dislocation`."""
        return rate_dislocation(
            self.current_rate.to_numpy(),
            self.proposed_rate.to_numpy(),
            exposure=self.exposure.to_numpy(),
            bands=bands,
        )

    def by(self, labels) -> pd.DataFrame:
        """Premium-weighted average change per group -- who absorbs the move.

        ``labels`` is an array/Series aligned with the census rows.
        """
        keys = np.asarray(labels)
        if keys.shape != (len(self.exposure),):
            raise ValueError("labels must align with the compared census")
        rows = {}
        for lvl in pd.unique(keys):
            m = keys == lvl
            cur = float((self.current_rate[m] * self.exposure[m]).sum())
            prop = float((self.proposed_rate[m] * self.exposure[m]).sum())
            rows[lvl] = {
                "n": int(m.sum()),
                "exposure": float(self.exposure[m].sum()),
                "current_premium": cur,
                "proposed_premium": prop,
                "avg_change": prop / cur - 1.0 if cur > 0 else np.nan,
            }
        out = pd.DataFrame(rows).T.sort_index(kind="stable")
        out.index.name = "group"
        out["n"] = out["n"].astype(int)
        return out


def compare_rating_plans(
    current: RatingPlan,
    proposed: RatingPlan,
    data: pd.DataFrame,
    columns: Mapping | None = None,
    exposure: str | None = None,
    unknown: str = "default",
) -> PlanComparison:
    """Rate one census under two plans and compare.

    Both plans are applied to the *same* rows with the same column mapping
    and unknown-level policy, so every difference in the result is a plan
    difference, not a data difference.

    Returns
    -------
    PlanComparison
        With ``summary()``, ``dislocation()``, ``by(labels)``, and the
        per-case ``change``.
    """
    cur = current.rate(data, columns=columns, unknown=unknown)["rate"]
    prop = proposed.rate(data, columns=columns, unknown=unknown)["rate"]
    expo = (
        data[exposure].astype(float)
        if exposure is not None
        else pd.Series(np.ones(len(data)), index=data.index)
    )
    if (expo < 0).any():
        raise ValueError("exposure must be nonnegative")
    return PlanComparison(
        current_rate=cur.rename("current_rate"),
        proposed_rate=prop.rename("proposed_rate"),
        exposure=expo.rename("exposure"),
    )
