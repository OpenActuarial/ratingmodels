r"""Rating relativities: lookup tables, one-way analysis, and GLM estimation.

Relativities (rating factors) scale a base rate up or down for a risk
characteristic (area, industry, age/sex band, plan tier, ...). They can be:

* supplied directly as a filed table (:class:`FactorTable`),
* estimated one-way as the ratio of each level's mean to the overall mean
  (:func:`one_way_relativities`), or
* estimated jointly with a generalized linear model
  (:class:`GLMRelativities`), which corrects for correlation between rating
  variables that one-way analysis cannot.

The GLM is fit by iteratively reweighted least squares (IRLS). With a log
link, :math:`\eta = X\beta + \text{offset}`, the working response and weights
are

.. math::
    z = \eta + (y - \mu)\,g'(\mu), \qquad
    w = \frac{w_{\text{prior}}}{V(\mu)\,g'(\mu)^2},

and :math:`\beta \leftarrow (X^\top W X)^{-1} X^\top W z` until convergence.
For the log link :math:`g'(\mu) = 1/\mu` and the variance function is
:math:`V(\mu) = \mu^p`: Poisson (:math:`p=1`), Gamma (:math:`p=2`), or
Tweedie (:math:`1 < p < 2`). A level's relativity is :math:`\exp(\beta)`
relative to the base (reference) level, whose relativity is 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Filed / supplied factor tables
# --------------------------------------------------------------------------- #
@dataclass
class FactorTable:
    """A named lookup of level -> multiplicative relativity.

    Parameters
    ----------
    name : str
        Rating variable name (e.g. ``"area"``).
    factors : mapping
        Level -> relativity. The base level should map to 1.0 by convention.
    default : float
        Relativity returned for unknown levels. Default 1.0.
    """

    name: str
    factors: Mapping
    default: float = 1.0

    def lookup(self, level) -> float:
        return float(self.factors.get(level, self.default))

    def apply(self, levels: Sequence) -> np.ndarray:
        return np.array([self.lookup(x) for x in levels], dtype=float)

    def normalized(self, base_level) -> "FactorTable":
        """Rebase so ``base_level`` has relativity 1.0."""
        base = self.lookup(base_level)
        if base <= 0:
            raise ValueError("base level relativity must be positive")
        return FactorTable(
            self.name,
            {k: v / base for k, v in self.factors.items()},
            self.default / base,
        )


def one_way_relativities(
    data: pd.DataFrame,
    factor: str,
    response: str,
    exposure: str | None = None,
    base_level=None,
) -> pd.Series:
    r"""One-way relativities: each level's (exposure-weighted) mean / overall mean.

    Does not adjust for correlation with other rating variables; use
    :class:`GLMRelativities` when variables are correlated.
    """
    if exposure is None:
        level_mean = data.groupby(factor)[response].mean()
        overall = data[response].mean()
    else:
        def _wm(g):
            return np.average(g[response], weights=g[exposure])
        level_mean = data.groupby(factor).apply(_wm, include_groups=False)
        overall = np.average(data[response], weights=data[exposure])

    rel = (level_mean / overall).rename("relativity")
    if base_level is not None:
        rel = rel / rel.loc[base_level]
    return rel


# --------------------------------------------------------------------------- #
# GLM relativities via IRLS
# --------------------------------------------------------------------------- #
_VARIANCE_POWER = {"poisson": 1.0, "gamma": 2.0}


@dataclass
class GLMRelativities:
    r"""GLM (log-link) relativity estimator fit by IRLS.

    Parameters
    ----------
    family : {"poisson", "gamma", "tweedie"}
        Response distribution. ``"tweedie"`` requires ``var_power`` in (1, 2).
    var_power : float, optional
        Tweedie variance power :math:`p` in :math:`V(\mu)=\mu^p`.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance on the relative change in deviance.

    Attributes
    ----------
    coefficients_ : pandas.Series
        Fitted :math:`\beta` including the intercept.
    relativities_ : dict[str, pandas.Series]
        Per-variable multiplicative relativities (base level = 1.0).
    base_value_ : float
        :math:`\exp(\text{intercept})`, the fitted base level.
    n_iter_ : int
        IRLS iterations used.
    deviance_ : float
        Final deviance. These attributes are populated by :meth:`fit`.
    """

    family: str = "poisson"
    var_power: float | None = None
    max_iter: int = 100
    tol: float = 1e-8

    coefficients_: pd.Series = field(default=None, init=False, repr=False)
    relativities_: dict = field(default_factory=dict, init=False, repr=False)
    base_value_: float = field(default=None, init=False, repr=False)
    n_iter_: int = field(default=0, init=False, repr=False)
    deviance_: float = field(default=np.nan, init=False, repr=False)

    # ----- internals ----- #
    def _power(self) -> float:
        fam = self.family.lower()
        if fam in _VARIANCE_POWER:
            return _VARIANCE_POWER[fam]
        if fam == "tweedie":
            if self.var_power is None or not (1 < self.var_power < 2):
                raise ValueError("tweedie requires var_power in (1, 2)")
            return float(self.var_power)
        raise ValueError(f"unknown family {self.family!r}")

    def _build_design(self, data, predictors, base_levels=None):
        """One-hot encode predictors (dropping the base level); add intercept.

        The base (reference) level for each predictor is, in order of
        preference: the value supplied in ``base_levels``, otherwise the most
        populous level (the standard choice, giving the most stable intercept).
        """
        base_levels = dict(base_levels or {})
        cols = {}
        chosen: dict = {}
        for var in predictors:
            cats = pd.Categorical(data[var])
            levels = list(cats.categories)
            if not levels:
                raise ValueError(f"predictor {var!r} has no levels")
            if var in base_levels:
                base = base_levels[var]
                if base not in levels:
                    raise ValueError(
                        f"base level {base!r} not found among {var!r} levels"
                    )
            else:
                base = data[var].value_counts().idxmax()  # modal level
            chosen[var] = base
            for lvl in levels:
                if lvl == base:
                    continue
                cols[f"{var}::{lvl}"] = (cats == lvl).astype(float)
        X = pd.DataFrame(cols, index=data.index)
        X.insert(0, "Intercept", 1.0)
        return X, chosen

    def fit(
        self,
        data: pd.DataFrame,
        response: str,
        predictors: Sequence[str],
        exposure: str | None = None,
        offset: str | None = None,
        weights: str | None = None,
        base_levels: Mapping[str, object] | None = None,
    ) -> "GLMRelativities":
        r"""Fit relativities for ``predictors`` against ``response``.

        ``exposure`` enters as a log offset (the natural choice for counts and
        pure premium). An explicit ``offset`` column (already on the log scale)
        and prior ``weights`` may also be supplied. ``base_levels`` maps a
        predictor to its reference level (relativity 1.0); unspecified
        predictors use their most populous level as the base.
        """
        p = self._power()
        X_df, base_levels_used = self._build_design(data, list(predictors), base_levels)
        X = X_df.to_numpy(dtype=float)
        y = data[response].to_numpy(dtype=float)
        n, p_dim = X.shape

        eta_offset = np.zeros(n)
        if exposure is not None:
            expo = data[exposure].to_numpy(dtype=float)
            if np.any(expo <= 0):
                raise ValueError("exposure must be positive for the log offset")
            eta_offset = eta_offset + np.log(expo)
        if offset is not None:
            eta_offset = eta_offset + data[offset].to_numpy(dtype=float)

        prior_w = (
            data[weights].to_numpy(dtype=float)
            if weights is not None
            else np.ones(n)
        )
        if np.any(prior_w < 0):
            raise ValueError("weights must be non-negative")

        # initialise mu near the data, away from zero
        mu = np.maximum(y, np.mean(y) * 0.1 + 1e-3)
        eta = np.log(mu)
        beta = np.zeros(p_dim)
        dev_old = np.inf

        for it in range(1, self.max_iter + 1):
            # log link: g'(mu) = 1/mu  ->  z = eta + (y - mu)/mu
            z = eta + (y - mu) / mu - eta_offset
            # w = prior_w / (V(mu) * g'(mu)^2) = prior_w * mu^2 / mu^p
            w = prior_w * mu ** (2 - p)
            WX = X * w[:, None]
            xtwx = X.T @ WX
            xtwz = X.T @ (w * z)
            beta = np.linalg.solve(xtwx, xtwz)
            eta = X @ beta + eta_offset
            eta = np.clip(eta, -30, 30)  # guard against overflow
            mu = np.exp(eta)

            dev = self._deviance(y, mu, prior_w, p)
            if np.isfinite(dev) and abs(dev - dev_old) <= self.tol * (abs(dev_old) + self.tol):
                self.n_iter_ = it
                break
            dev_old = dev
        else:
            self.n_iter_ = self.max_iter

        self.coefficients_ = pd.Series(beta, index=X_df.columns)
        self.deviance_ = float(dev_old if not np.isfinite(dev) else dev)
        self.base_value_ = float(np.exp(beta[0]))

        # assemble relativities per variable (base level = 1.0)
        rels: dict[str, pd.Series] = {}
        for var in predictors:
            levels = [base_levels_used[var]]
            vals = [1.0]
            for name, coef in self.coefficients_.items():
                if name.startswith(f"{var}::"):
                    levels.append(name.split("::", 1)[1])
                    vals.append(float(np.exp(coef)))
            rels[var] = pd.Series(vals, index=levels, name=f"{var}_relativity")
        self.relativities_ = rels
        return self

    @staticmethod
    def _deviance(y, mu, w, p) -> float:
        """Tweedie deviance (covers Poisson p=1 and Gamma p=2 in the limit)."""
        y = np.maximum(y, 0)
        eps = 1e-12
        if abs(p - 1.0) < 1e-9:  # Poisson
            term = np.where(y > 0, y * np.log((y + eps) / mu), 0.0) - (y - mu)
            return float(2 * np.sum(w * term))
        if abs(p - 2.0) < 1e-9:  # Gamma
            term = -np.log((y + eps) / mu) + (y - mu) / mu
            return float(2 * np.sum(w * term))
        # general Tweedie, 1 < p < 2
        a = np.where(y > 0, y ** (2 - p) / ((1 - p) * (2 - p)), 0.0)
        b = y * mu ** (1 - p) / (1 - p)
        c = mu ** (2 - p) / (2 - p)
        return float(2 * np.sum(w * (a - b + c)))

    def predict(self, data: pd.DataFrame, exposure: str | None = None) -> np.ndarray:
        """Predicted mean for new rows using the fitted relativities."""
        if self.coefficients_ is None:
            raise RuntimeError("model is not fit")
        mu = np.full(len(data), self.base_value_, dtype=float)
        for var, rel in self.relativities_.items():
            mu *= np.array([rel.get(v, 1.0) for v in data[var]], dtype=float)
        if exposure is not None:
            mu *= data[exposure].to_numpy(dtype=float)
        return mu
