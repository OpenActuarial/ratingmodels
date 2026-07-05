r"""Frequency-severity modeling: two GLMs, one pure premium.

The standard pricing decomposition (see the ecosystem conventions:
``loss_per_exposure = frequency x severity``) fit as two log-link GLMs:

* **frequency** -- claim counts per unit of exposure (Poisson by default),
  fit on every record with exposure as a log offset;
* **severity** -- average cost per claim (Gamma by default), fit only on
  records with claims, weighted by claim count (the average of :math:`k`
  claims carries :math:`k` claims' worth of information).

Because both links are logs, the combined model is itself multiplicative:
the pure-premium relativity of a level is the *product* of its frequency and
severity relativities, and the predicted pure premium is exactly
``frequency_prediction * severity_prediction``. Fitting the pieces
separately shows *why* a level is expensive -- more claims, larger claims,
or both -- which a single Tweedie fit cannot.
"""
from __future__ import annotations

import warnings

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .relativity import GLMRelativities

__all__ = ["FrequencySeverityModel"]

_SEV_RESPONSE = "_ratingmodels_severity_"
_SEV_WEIGHT = "_ratingmodels_severity_weight_"


@dataclass
class FrequencySeverityModel:
    """A pure-premium model composed of a frequency GLM and a severity GLM.

    Parameters
    ----------
    frequency, severity : GLMRelativities, optional
        Unfitted component models. Default ``family="poisson"`` for frequency
        and ``family="gamma"`` for severity -- the classical pairing.

    Attributes
    ----------
    frequency, severity : GLMRelativities
        The fitted component models (all their diagnostics --
        ``relativity_table``, ``residuals``, ``summary`` -- apply per part).
    """

    frequency: GLMRelativities = field(
        default_factory=lambda: GLMRelativities(family="poisson")
    )
    severity: GLMRelativities = field(
        default_factory=lambda: GLMRelativities(family="gamma")
    )

    _fit_info_: dict = field(default=None, init=False, repr=False)

    def fit(
        self,
        data: pd.DataFrame,
        claim_count: str,
        claim_amount: str,
        exposure: str | None = None,
        frequency_predictors: Sequence[str] = (),
        severity_predictors: Sequence[str] | None = None,
        frequency_continuous: Sequence[str] = (),
        severity_continuous: Sequence[str] | None = None,
        frequency_interactions: Sequence[tuple] = (),
        severity_interactions: Sequence[tuple] | None = None,
        base_levels: Mapping[str, object] | None = None,
    ) -> "FrequencySeverityModel":
        """Fit both components from one claims frame.

        Parameters
        ----------
        data : DataFrame
            One row per risk/cell with total ``claim_count`` and total
            ``claim_amount`` over the period.
        claim_count, claim_amount : str
            Count and aggregate amount columns.
        exposure : str, optional
            Exposure column; enters the frequency model as a log offset.
        frequency_predictors, severity_predictors : sequence of str
            Categorical rating variables per component. Severity defaults to
            the frequency list -- pass an explicit (possibly shorter) list
            when severity supports fewer variables, which is common: severity
            fits on claims only and thins out fast.
        frequency_continuous, severity_continuous : sequence of str
            Continuous covariates per component (severity defaults to the
            frequency list).
        frequency_interactions, severity_interactions : sequence of pairs
            Interaction terms per component, as in
            :meth:`GLMRelativities.fit` (severity defaults to the frequency
            list). Categorical x categorical interactions surface in
            :meth:`combined_relativities` under an ``"a:b"`` key with a
            MultiIndex of level pairs.
        base_levels : mapping, optional
            Predictor -> reference level, shared by both components.

        Notes
        -----
        Severity is fit on rows with ``claim_count > 0`` **and**
        ``claim_amount > 0``, with response ``claim_amount / claim_count``
        and prior weight ``claim_count``. Rows with claims closed at zero
        amount still count toward frequency; if there are many of them,
        consider whether a zero-mass component belongs in the model.
        """
        if severity_predictors is None:
            severity_predictors = list(frequency_predictors)
        if severity_continuous is None:
            severity_continuous = list(frequency_continuous)
        if severity_interactions is None:
            severity_interactions = list(frequency_interactions)
        for reserved in (_SEV_RESPONSE, _SEV_WEIGHT):
            if reserved in data.columns:
                raise ValueError(f"column name {reserved!r} is reserved")

        counts = data[claim_count].to_numpy(dtype=float)
        amounts = data[claim_amount].to_numpy(dtype=float)
        if np.any(counts < 0):
            raise ValueError("claim_count must be nonnegative")
        orphan = (counts <= 0) & (amounts > 0)
        if orphan.any():
            raise ValueError(
                f"{int(orphan.sum())} row(s) have positive claim_amount with "
                "zero claim_count; severity is undefined there"
            )

        self.frequency.fit(
            data,
            response=claim_count,
            predictors=list(frequency_predictors),
            exposure=exposure,
            base_levels=base_levels,
            continuous=tuple(frequency_continuous),
            interactions=tuple(frequency_interactions),
        )

        pos = (counts > 0) & (amounts > 0)
        if not pos.any():
            raise ValueError("no rows with positive claim count and amount to fit severity")
        zero_amount = int(((counts > 0) & (amounts <= 0)).sum())
        if zero_amount:
            warnings.warn(
                f"{zero_amount} row(s) with claims but zero amount are excluded "
                "from the severity fit (they still count toward frequency)",
                stacklevel=2,
            )
        sev = data.loc[pos].copy()
        sev[_SEV_RESPONSE] = amounts[pos] / counts[pos]
        sev[_SEV_WEIGHT] = counts[pos]
        self.severity.fit(
            sev,
            response=_SEV_RESPONSE,
            predictors=list(severity_predictors),
            weights=_SEV_WEIGHT,
            base_levels=base_levels,
            continuous=tuple(severity_continuous),
            interactions=tuple(severity_interactions),
        )

        self._fit_info_ = {
            "claim_count": claim_count,
            "claim_amount": claim_amount,
            "exposure": exposure,
            "n_severity_rows": int(pos.sum()),
        }
        return self

    # ----- predictions ----- #
    def _check_fit(self):
        if self._fit_info_ is None:
            raise RuntimeError("model is not fit")

    def frequency_prediction(
        self, data: pd.DataFrame, exposure: str | None = None
    ) -> np.ndarray:
        """Expected claim counts (with ``exposure``) or claim rate per unit."""
        self._check_fit()
        return self.frequency.predict(data, exposure=exposure)

    def severity_prediction(self, data: pd.DataFrame) -> np.ndarray:
        """Expected cost per claim."""
        self._check_fit()
        return self.severity.predict(data)

    def pure_premium_prediction(
        self, data: pd.DataFrame, exposure: str | None = None
    ) -> np.ndarray:
        """Expected loss: total (with ``exposure``) or per exposure unit.

        Exactly ``frequency_prediction(data, exposure) *
        severity_prediction(data)`` -- the frequency x severity identity.
        """
        return self.frequency_prediction(data, exposure=exposure) * self.severity_prediction(data)

    # ----- combined structure ----- #
    @property
    def base_value_(self) -> float:
        """Pure premium per exposure unit at base levels."""
        self._check_fit()
        return float(self.frequency.base_value_ * self.severity.base_value_)

    def combined_relativities(self) -> dict:
        """Per-variable pure-premium relativities: frequency x severity.

        Variables appearing in only one component contribute that component's
        relativities unchanged (the other's factor is 1.0); levels missing
        from a component take that component's base, 1.0 -- matching how its
        ``predict`` treats unseen levels.

        Returns
        -------
        dict of str -> pandas.DataFrame
            Per variable, indexed by level, with columns ``frequency``,
            ``severity``, ``combined``.
        """
        self._check_fit()
        out: dict[str, pd.DataFrame] = {}
        f_rels = self.frequency.relativities_
        s_rels = self.severity.relativities_
        seen = list(f_rels)
        seen += [v for v in s_rels if v not in f_rels]
        for var in seen:
            f = f_rels.get(var)
            s = s_rels.get(var)
            if f is not None and s is not None:
                idx = f.index.union(s.index, sort=False)
            else:
                idx = (f if f is not None else s).index
            if not isinstance(idx, pd.MultiIndex):
                idx = pd.Index(idx, name=var)
            f_al = (f.reindex(idx) if f is not None else pd.Series(index=idx, dtype=float)).fillna(1.0)
            s_al = (s.reindex(idx) if s is not None else pd.Series(index=idx, dtype=float)).fillna(1.0)
            out[var] = pd.DataFrame(
                {"frequency": f_al, "severity": s_al, "combined": f_al * s_al}
            )
        return out

    def to_factor_tables(self) -> dict:
        """Combined pure-premium relativities as :class:`FactorTable` objects.

        One table per variable, built from the ``combined`` column of
        :meth:`combined_relativities` (frequency x severity) with
        ``default=1.0`` for unknown levels -- the pure-premium plan you
        would actually apply, ready for the build-up and renewal machinery.
        Interaction terms are excluded (a :class:`FactorTable` is
        single-variable by contract); read their cells from
        :meth:`combined_relativities`.
        """
        from .relativity import FactorTable

        main = set(self.frequency._design_info_["predictors"]) | set(
            self.severity._design_info_["predictors"]
        )
        return {
            var: FactorTable(name=var, factors=dict(tab["combined"]), default=1.0)
            for var, tab in self.combined_relativities().items()
            if var in main
        }

    def summary(self) -> pd.DataFrame:
        """Both component coefficient tables, stacked under a model key."""
        self._check_fit()
        return pd.concat(
            {"frequency": self.frequency.summary(), "severity": self.severity.summary()},
            names=["model", "term"],
        )
