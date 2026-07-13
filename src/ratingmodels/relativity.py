r"""Rating relativities: lookup tables, one-way analysis, and GLM estimation.

Relativities (rating factors) scale a base rate up or down for a risk
characteristic (area, industry, age/sex band, plan tier, ...). They can be:

* supplied directly as a filed table (:class:`FactorTable`),
* estimated one-way as the ratio of each level's mean to the overall mean
  (:func:`one_way_relativities`), or
* estimated jointly with a generalized linear model
  (:class:`GLMRelativities`), which corrects for correlation between rating
  variables that one-way analysis cannot.

The model is a log-link GLM, :math:`\eta = X\beta + \text{offset}` with
variance function :math:`V(\mu) = \mu^p`: Poisson (:math:`p=1`), Gamma
(:math:`p=2`), or Tweedie (:math:`1 < p < 2`). A level's relativity is
:math:`\exp(\beta)` relative to the base (reference) level, whose relativity
is 1.

Estimation is delegated to :class:`statsmodels.api.GLM` -- solver,
convergence, covariance, and the fitted null model are statsmodels'
responsibility, with the dispersion estimated from the Pearson chi-square
(quasi-likelihood) for every family. ``ratingmodels`` owns the actuarial
layer around it: the design encoding and base-level semantics, the
coefficient-to-relativity conversion, prediction with unseen-level fallback,
residuals on arbitrary frames, and the exhibits. The fitted statsmodels
results object is exposed as ``results_`` for anything beyond that
(``results_.get_influence()``, ``results_.get_prediction(...)``, Wald
tests, ...).
"""
from __future__ import annotations

import warnings

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

#: Sentinel distinguishing "not passed" from an explicit None in diagnostics.
_UNSET = object()

#: Linear-predictor bounds applied before exponentiation to avoid overflow.
_ETA_LO, _ETA_HI = -30.0, 30.0


class PredictionClipWarning(UserWarning):
    """Emitted when a linear predictor is clipped before exponentiation.

    Clipping caps a prediction at ``exp(+/-30)`` rather than extrapolating, so a
    severely extrapolated or unstable model can otherwise return a finite-looking
    number without any signal. Filter it with the standard :mod:`warnings`
    machinery, or pass ``on_overflow="raise"`` to fail instead / ``"ignore"`` to
    silence.
    """


class UnknownLevelWarning(UserWarning):
    """Emitted when prediction data contains categorical levels unseen at fit time.

    Such levels are scored at the base (reference) level. Pass ``unknown="raise"``
    to reject them instead, or ``unknown="base"`` (the default) to accept the
    base-level fallback silently.
    """


def _clip_eta(eta: np.ndarray, on_overflow: str, context: str) -> np.ndarray:
    """Clip the linear predictor to ``[_ETA_LO, _ETA_HI]``, observably.

    ``on_overflow`` is ``"warn"`` (default -- emit :class:`PredictionClipWarning`
    naming the affected rows), ``"raise"`` (raise :class:`OverflowError`), or
    ``"ignore"`` (clip silently, the historical behaviour).
    """
    if on_overflow not in ("warn", "raise", "ignore"):
        raise ValueError("on_overflow must be 'warn', 'raise', or 'ignore'")
    if not np.all(np.isfinite(eta)):
        raise ValueError(f"{context}: linear predictor has non-finite values")
    mask = (eta < _ETA_LO) | (eta > _ETA_HI)
    n = int(np.count_nonzero(mask))
    if n and on_overflow != "ignore":
        rows = np.flatnonzero(mask)
        msg = (
            f"{context}: {n} linear-predictor value(s) outside "
            f"[{_ETA_LO:g}, {_ETA_HI:g}] were clipped before exp (raw range "
            f"[{float(np.min(eta)):.3g}, {float(np.max(eta)):.3g}]); those "
            f"predictions are capped, not extrapolated (rows {rows[:10].tolist()}"
            f"{'...' if n > 10 else ''})"
        )
        if on_overflow == "raise":
            raise OverflowError(msg)
        warnings.warn(msg, PredictionClipWarning, stacklevel=3)
    return np.clip(eta, _ETA_LO, _ETA_HI)


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

    def apply(self, levels: Sequence) -> "np.ndarray | pd.Series":
        """Vectorized lookup: relativity for every element of ``levels``.

        A Series in gives a Series out on the same index (unknown levels get
        ``default``); any other sequence gives a numpy array.
        """
        if isinstance(levels, pd.Series):
            return levels.map(lambda x: self.factors.get(x, self.default)).astype(float)
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
    r"""GLM (log-link) relativity estimator, fit via statsmodels.

    Parameters
    ----------
    family : {"poisson", "gamma", "tweedie"}
        Response distribution. ``"tweedie"`` requires ``var_power`` in (1, 2).
    var_power : float, optional
        Tweedie variance power :math:`p` in :math:`V(\mu)=\mu^p`.
    max_iter : int
        Maximum solver iterations (passed to ``statsmodels``).
    tol : float
        Solver convergence tolerance (passed to ``statsmodels``).

    Attributes
    ----------
    coefficients_ : pandas.Series
        Fitted :math:`\beta` including the intercept.
    relativities_ : dict[str, pandas.Series]
        Per-variable multiplicative relativities (base level = 1.0).
    base_value_ : float
        :math:`\exp(\text{intercept})`, the fitted base level.
    results_ : statsmodels GLMResults
        The underlying fitted results object -- the common actuarial outputs
        live on this class, but nothing statistical is walled off.
    n_iter_ : int
        Solver iterations used.
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
    converged_: bool = field(default=False, init=False, repr=False)
    null_deviance_: float = field(default=np.nan, init=False, repr=False)
    pearson_chi2_: float = field(default=np.nan, init=False, repr=False)
    dispersion_: float = field(default=np.nan, init=False, repr=False)
    se_: pd.Series = field(default=None, init=False, repr=False)
    cov_params_: pd.DataFrame = field(default=None, init=False, repr=False)
    results_: object = field(default=None, init=False, repr=False)
    _design_info_: dict = field(default=None, init=False, repr=False)

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

    def _sm_family(self, sm):
        """The statsmodels family object for this model (always log link)."""
        p = self._power()
        log = sm.families.links.Log()
        if abs(p - 1.0) < 1e-9:
            return sm.families.Poisson(log)
        if abs(p - 2.0) < 1e-9:
            return sm.families.Gamma(log)
        return sm.families.Tweedie(link=log, var_power=p, eql=True)

    def _build_design(
        self, data, predictors, base_levels=None, continuous=(), interactions=()
    ):
        """One-hot encode predictors (dropping the base level); add intercept.

        The base (reference) level for each predictor is, in order of
        preference: the value supplied in ``base_levels``, otherwise the most
        populous level (the standard choice, giving the most stable intercept).
        ``continuous`` columns enter as numeric covariates unchanged.

        ``interactions`` are ``(a, b)`` pairs. Categorical x categorical adds
        an indicator per *observed* non-base x non-base level pair (treatment
        coding: cells containing either base level carry no interaction term,
        so the main effects keep their interpretation; unobserved pairs are
        skipped to keep the design full-rank). Categorical x continuous adds
        one slope-modifier column per non-base level.

        Returns ``(X, chosen_bases, spec)`` where ``spec`` is a structured
        description of every column -- the single source of truth from which
        prediction and diagnostics rebuild the design on any frame.
        """
        base_levels = dict(base_levels or {})
        cols = {}
        spec: list[tuple] = [("intercept",)]
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
                spec.append(("level", var, lvl))
        for var in continuous:
            vals = data[var].to_numpy(dtype=float)
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"continuous covariate {var!r} has non-finite values")
            cols[var] = vals
            spec.append(("cont", var))
        for a, b in interactions:
            a_cat, b_cat = a in chosen, b in chosen
            if a_cat and (b in continuous):
                pass  # canonical order: (categorical, continuous)
            elif b_cat and (a in continuous):
                a, b = b, a
                a_cat, b_cat = True, False
            elif not (a_cat and b_cat):
                raise ValueError(
                    f"interaction ({a!r}, {b!r}) must pair two categorical "
                    "predictors or a categorical predictor with a continuous "
                    "covariate, each also present as a main effect"
                )
            if a_cat and b_cat:
                ind_a = data[a]
                ind_b = data[b]
                for la in pd.Categorical(ind_a).categories:
                    if la == chosen[a]:
                        continue
                    mask_a = (ind_a == la).to_numpy(dtype=float)
                    for lb in pd.Categorical(ind_b).categories:
                        if lb == chosen[b]:
                            continue
                        col = mask_a * (ind_b == lb).to_numpy(dtype=float)
                        if not col.any():
                            continue  # unobserved pair -> would be all-zero
                        cols[f"{a}::{la}:{b}::{lb}"] = col
                        spec.append(("ixcc", a, la, b, lb))
            else:
                c_vals = data[b].to_numpy(dtype=float)
                for la in pd.Categorical(data[a]).categories:
                    if la == chosen[a]:
                        continue
                    cols[f"{a}::{la}:{b}"] = (data[a] == la).to_numpy(dtype=float) * c_vals
                    spec.append(("ixcont", a, la, b))
        X = pd.DataFrame(cols, index=data.index)
        X.insert(0, "Intercept", 1.0)
        return X, chosen, spec

    def fit(
        self,
        data: pd.DataFrame,
        response: str,
        predictors: Sequence[str],
        exposure: str | None = None,
        offset: str | None = None,
        weights: str | None = None,
        base_levels: Mapping[str, object] | None = None,
        continuous: Sequence[str] = (),
        interactions: Sequence[tuple] = (),
    ) -> "GLMRelativities":
        r"""Fit relativities for ``predictors`` against ``response``.

        **Aggregate vs. rate responses.** ``exposure`` enters as a log
        offset, which is correct when the response is an *aggregate* -- claim
        counts or total amounts, :math:`E[Y] = e\,\exp(X\beta)`. When the
        response is already a *rate* (divided by exposure: pure premium,
        loss per unit), do **not** pass ``exposure``; pass it as
        ``weights`` instead, so the variance scales as
        :math:`V(\mu)/e`. The two parameterizations coincide only for
        Poisson (:math:`p=1`); for Gamma and Tweedie the weights form is the
        one consistent with a response averaged over :math:`e` independent
        claims (it is exactly how the severity model inside
        :class:`~ratingmodels.FrequencySeverityModel` is fit).

        An explicit ``offset`` column (already on the log scale) may also be
        supplied. ``weights`` are *variance* weights (statsmodels
        ``var_weights``): the variance of row :math:`i` is
        :math:`\phi V(\mu_i)/w_i`. ``base_levels`` maps a predictor to its
        reference level (relativity 1.0); unspecified predictors use their
        most populous level as the base.
        """
        X_df, base_levels_used, spec = self._build_design(
            data, list(predictors), base_levels,
            continuous=tuple(continuous), interactions=tuple(interactions),
        )
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

        # ----- estimation: delegated to statsmodels ----- #
        # ratingmodels owns the actuarial layer -- design encoding, base
        # levels, relativity assembly, prediction with unseen-level fallback,
        # and family deviance for scoring arbitrary frames. statsmodels owns
        # the numerics: solver, convergence, covariance, and the fitted null
        # model. Imported here so `import ratingmodels` stays fast for the
        # (large) part of the package that never fits a GLM.
        import statsmodels.api as sm

        rank_deficient = np.linalg.matrix_rank(X) < p_dim
        if rank_deficient:
            warnings.warn(
                "design matrix is rank deficient (aliased levels); "
                "coefficient standard errors are unavailable",
                stacklevel=2,
            )

        res = sm.GLM(
            y, X_df, family=self._sm_family(sm),
            offset=eta_offset, var_weights=prior_w,
        ).fit(maxiter=self.max_iter, tol=self.tol, scale="X2")
        self.results_ = res

        hist = getattr(res, "fit_history", None) or {}
        n_iter = hist.get("iteration", len(hist.get("deviance", [])))
        self.n_iter_ = max(int(n_iter), 1)
        # statsmodels flags a *perfect* fit (saturated model, deviance ~ 0)
        # as unconverged with a PerfectSeparationWarning; an exactly solved
        # problem is converged in any sense that matters here
        perfect = np.isfinite(res.deviance) and float(res.deviance) <= self.tol * max(
            float(res.null_deviance), 1.0
        )
        self.converged_ = bool(getattr(res, "converged", True)) or perfect

        self.coefficients_ = pd.Series(np.asarray(res.params, dtype=float), index=X_df.columns)
        beta = self.coefficients_.to_numpy()
        self.deviance_ = float(res.deviance)
        self.base_value_ = float(np.exp(beta[0]))

        # quasi-likelihood convention throughout: scale="X2" estimates the
        # dispersion from the Pearson chi-square for every family
        self.pearson_chi2_ = float(res.pearson_chi2)
        self.dispersion_ = float(res.scale)
        # the fitted intercept(+offset)-only model, correct for every family
        self.null_deviance_ = float(res.null_deviance)

        if rank_deficient:
            self.cov_params_ = None
            self.se_ = None
        else:
            cov = np.asarray(res.cov_params())
            self.cov_params_ = pd.DataFrame(cov, index=X_df.columns, columns=X_df.columns)
            self.se_ = pd.Series(np.sqrt(np.maximum(np.diag(cov), 0.0)), index=X_df.columns)

        self._design_info_ = {
            "predictors": list(predictors),
            "base_levels": dict(base_levels_used),
            "continuous": list(continuous),
            "interactions": [tuple(ix) for ix in interactions],
            "columns": list(X_df.columns),
            "spec": list(spec),
            "response": response,
            "exposure": exposure,
            "offset": offset,
            "weights": weights,
        }

        # assemble relativities per variable (base level = 1.0), spec-driven
        coefs = self.coefficients_.to_numpy()
        rels: dict[str, pd.Series] = {}
        for var in predictors:
            levels = [base_levels_used[var]]
            vals = [1.0]
            for j, term in enumerate(spec):
                if term[0] == "level" and term[1] == var:
                    levels.append(term[2])
                    vals.append(float(np.exp(coefs[j])))
            rels[var] = pd.Series(vals, index=levels, name=f"{var}_relativity")
        # categorical x categorical interactions: a relativity per observed
        # non-base cell, multiplying on top of both main effects
        ix_cells: dict[tuple, list] = {}
        for j, term in enumerate(spec):
            if term[0] == "ixcc":
                _, a, la, b, lb = term
                ix_cells.setdefault((a, b), []).append(((la, lb), float(np.exp(coefs[j]))))
        for (a, b), cells in ix_cells.items():
            idx = pd.MultiIndex.from_tuples([c[0] for c in cells], names=[a, b])
            rels[f"{a}:{b}"] = pd.Series(
                [c[1] for c in cells], index=idx, name=f"{a}:{b}_relativity"
            )
        self.relativities_ = rels
        return self

    @staticmethod
    def _unit_deviance(y, mu, p) -> np.ndarray:
        """Per-observation Tweedie deviance :math:`d_i` (before prior weights).

        Covers Poisson (p=1) and Gamma (p=2) in the limit. The model deviance
        is ``sum(w * d)`` and the deviance residual is ``sign(y-mu)*sqrt(w*d)``.
        """
        y = np.maximum(y, 0)
        eps = 1e-12
        if abs(p - 1.0) < 1e-9:  # Poisson
            term = np.where(y > 0, y * np.log((y + eps) / mu), 0.0) - (y - mu)
            return 2 * term
        if abs(p - 2.0) < 1e-9:  # Gamma
            term = -np.log((y + eps) / mu) + (y - mu) / mu
            return 2 * term
        # general Tweedie, 1 < p < 2
        a = np.where(y > 0, y ** (2 - p) / ((1 - p) * (2 - p)), 0.0)
        b = y * mu ** (1 - p) / (1 - p)
        c = mu ** (2 - p) / (2 - p)
        return 2 * (a - b + c)

    @classmethod
    def _deviance(cls, y, mu, w, p) -> float:
        """Tweedie deviance (covers Poisson p=1 and Gamma p=2 in the limit)."""
        return float(np.sum(w * cls._unit_deviance(y, mu, p)))

    def predict(
        self,
        data: pd.DataFrame,
        exposure: str | None = None,
        offset: str | None = None,
        *,
        unknown: str = "base",
        on_overflow: str = "warn",
    ) -> np.ndarray:
        """Predicted mean for new rows.

        ``unknown`` controls categorical levels not seen at fit time:
        ``"base"`` (default) scores them at the base (reference) level silently,
        ``"warn"`` does the same but emits :class:`UnknownLevelWarning`, and
        ``"raise"`` rejects them -- the safer default for core pricing, where a
        new territory or class silently taking the base rate is a real hazard.

        ``on_overflow`` controls the ``exp`` overflow guard on the linear
        predictor: ``"warn"`` (default) flags clipped rows, ``"raise"`` fails,
        ``"ignore"`` clips silently. ``exposure`` multiplies the mean; ``offset``
        is a column already on the log scale.
        """
        if self.coefficients_ is None:
            raise RuntimeError("model is not fit")
        self._check_unknown_levels(data, unknown)
        X = self._design_matrix_from_info(data)
        eta = X @ self.coefficients_.to_numpy()
        if offset is not None:
            eta += data[offset].to_numpy(dtype=float)
        eta = _clip_eta(eta, on_overflow, "predict")
        mu = np.exp(eta)
        if exposure is not None:
            mu *= data[exposure].to_numpy(dtype=float)
        return mu

    def _check_unknown_levels(self, data: pd.DataFrame, unknown: str) -> None:
        """Detect categorical values absent from the fitted level sets.

        The known set for each predictor is its base level plus every level that
        earned a coefficient (main effect or interaction). Anything else is
        unseen and, unless ``unknown == "raise"``, is scored at the base level.
        """
        if unknown not in ("base", "warn", "raise"):
            raise ValueError("unknown must be 'base', 'warn', or 'raise'")
        if unknown == "base":
            return  # accept the base-level fallback silently (historical default)
        info = self._design_info_
        known: dict[str, set] = {var: {info["base_levels"].get(var)} for var in info["predictors"]}
        for term in info["spec"]:
            if term[0] == "level":
                known.setdefault(term[1], set()).add(term[2])
            elif term[0] == "ixcc":
                _, a, la, b, lb = term
                known.setdefault(a, set()).add(la)
                known.setdefault(b, set()).add(lb)
            elif term[0] == "ixcont":
                _, a, la, _c = term
                known.setdefault(a, set()).add(la)
        offenders: dict[str, list] = {}
        for var, levels in known.items():
            if var not in data.columns:
                continue
            unseen = set(pd.unique(data[var])) - levels
            if unseen:
                offenders[var] = sorted(map(str, unseen))[:10]
        if offenders:
            msg = f"categorical level(s) unseen at fit time (scored at base): {offenders}"
            if unknown == "raise":
                raise ValueError(msg)
            warnings.warn(msg, UnknownLevelWarning, stacklevel=3)

    def predict_interval(
        self,
        data: pd.DataFrame,
        confidence_level: float = 0.95,
        exposure: str | None = None,
        offset: str | None = None,
        *,
        unknown: str = "base",
        on_overflow: str = "warn",
    ) -> pd.DataFrame:
        r"""Predicted mean with its confidence interval, per row.

        The interval is for the *fitted mean* (the rate the model assigns to
        this cell), not for an individual outcome: the delta method on the
        link scale, :math:`\exp(\hat\eta \pm z\,\sqrt{x^\top \Sigma x})`
        with :math:`\Sigma` the quasi-likelihood coefficient covariance.
        Individual outcomes vary far more than the mean; for that question a
        frequency-severity simulation is the right tool, not a GLM interval.

        Returns
        -------
        pandas.DataFrame
            Index-aligned with ``data``; columns ``predicted``, ``ci_low``,
            ``ci_high``. With ``exposure``, all three are on the total scale.
        """
        if self.coefficients_ is None:
            raise RuntimeError("model is not fit")
        if self.cov_params_ is None:
            raise RuntimeError(
                "coefficient covariance unavailable (rank-deficient design)"
            )
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        self._check_unknown_levels(data, unknown)
        z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        X = self._design_matrix_from_info(data)
        eta = X @ self.coefficients_.to_numpy()
        if offset is not None:
            eta += data[offset].to_numpy(dtype=float)
        var_eta = np.einsum("ij,jk,ik->i", X, self.cov_params_.to_numpy(), X)
        se_eta = np.sqrt(np.maximum(var_eta, 0.0))
        eta = _clip_eta(eta, on_overflow, "predict_interval")
        out = pd.DataFrame(
            {
                "predicted": np.exp(eta),
                "ci_low": np.exp(np.clip(eta - z * se_eta, _ETA_LO, _ETA_HI)),
                "ci_high": np.exp(np.clip(eta + z * se_eta, _ETA_LO, _ETA_HI)),
            },
            index=data.index,
        )
        if exposure is not None:
            expo = data[exposure].to_numpy(dtype=float)
            for col in out.columns:
                out[col] = out[col] * expo
        return out

    @property
    def deviance_explained_(self) -> float:
        """Proportion of null deviance explained, ``1 - deviance/null_deviance``.

        The GLM analogue of :math:`R^2`: 0 means the predictors add nothing
        over the intercept(+offset)-only model, 1 means a saturated fit.
        """
        if not np.isfinite(self.null_deviance_) or self.null_deviance_ <= 0:
            return np.nan
        return float(1.0 - self.deviance_ / self.null_deviance_)

    def _design_matrix_from_info(self, data: pd.DataFrame) -> np.ndarray:
        """Rebuild the design matrix for ``data`` in the *fitted* column order.

        Built from the structured column ``spec`` recorded at fit time --
        never re-deriving levels from the data -- so it is safe on validation
        slices whose level sets differ from the training frame; unseen levels
        (and unseen interaction cells) get all-zero indicators, i.e. the base.
        This is the single design path shared by :meth:`predict`,
        :meth:`residuals`, and :meth:`predict_interval`.
        """
        info = self._design_info_
        n = len(data)
        cols = np.empty((n, len(info["spec"])), dtype=float)
        for j, term in enumerate(info["spec"]):
            kind = term[0]
            if kind == "intercept":
                cols[:, j] = 1.0
            elif kind == "level":
                _, var, lvl = term
                cols[:, j] = (data[var] == lvl).to_numpy(dtype=float)
            elif kind == "cont":
                cols[:, j] = data[term[1]].to_numpy(dtype=float)
            elif kind == "ixcc":
                _, a, la, b, lb = term
                cols[:, j] = (
                    (data[a] == la).to_numpy(dtype=float)
                    * (data[b] == lb).to_numpy(dtype=float)
                )
            elif kind == "ixcont":
                _, a, la, c = term
                cols[:, j] = (data[a] == la).to_numpy(dtype=float) * data[c].to_numpy(
                    dtype=float
                )
            else:  # pragma: no cover - spec is produced in-package
                raise ValueError(f"unknown design term {term!r}")
        return cols

    def residuals(
        self,
        data: pd.DataFrame,
        kind: str = "deviance",
        response: str | None = None,
        exposure=_UNSET,
        offset=_UNSET,
        weights=_UNSET,
    ) -> pd.Series:
        r"""Per-row residuals on ``data``, as a Series aligned to its index.

        Parameters
        ----------
        data : DataFrame
            Rows to evaluate -- typically the training frame, but any frame
            with the model's columns works (e.g. a validation split).
        kind : {"deviance", "pearson", "standardized", "response"}
            * ``"response"`` -- raw :math:`y - \hat\mu`.
            * ``"pearson"`` -- :math:`(y-\hat\mu)\sqrt{w}/\sqrt{V(\hat\mu)}`;
              the squared Pearson residuals sum to ``pearson_chi2_`` on the
              training data.
            * ``"deviance"`` -- :math:`\mathrm{sign}(y-\hat\mu)\sqrt{w\,d_i}`;
              the squared deviance residuals sum to ``deviance_`` on the
              training data.
            * ``"standardized"`` -- Pearson scaled by
              :math:`\sqrt{\hat\phi\,(1-h_i)}` with :math:`h_i` the IRLS hat
              value, so values beyond :math:`\pm 2` flag unusual rows on a
              common scale. Leverage is exact on the training data (on new
              data :math:`h_i` is the same formula, not a true leverage).
        response, exposure, offset, weights : str, optional
            Column names; each defaults to the column used in :meth:`fit`.

        Notes
        -----
        Plotting deviance or standardized residuals against fitted values and
        against each rating variable is the standard check that the variance
        function and link are adequate; structure in these plots means the
        relativities are absorbing the wrong shape.
        """
        if self.coefficients_ is None:
            raise RuntimeError("model is not fit")
        info = self._design_info_
        response = info["response"] if response is None else response
        exposure = info["exposure"] if exposure is _UNSET else exposure
        offset = info["offset"] if offset is _UNSET else offset
        weights = info["weights"] if weights is _UNSET else weights

        y = data[response].to_numpy(dtype=float)
        mu = self.predict(data, exposure=exposure, offset=offset)
        prior_w = (
            data[weights].to_numpy(dtype=float) if weights is not None else np.ones(len(data))
        )
        p = self._power()

        if kind == "response":
            res = y - mu
        elif kind == "pearson":
            res = (y - mu) * np.sqrt(prior_w) / np.sqrt(mu**p)
        elif kind == "deviance":
            d = self._unit_deviance(y, mu, p)
            res = np.sign(y - mu) * np.sqrt(prior_w * np.maximum(d, 0.0))
        elif kind == "standardized":
            if self.cov_params_ is None:
                raise RuntimeError(
                    "standardized residuals need the coefficient covariance, "
                    "which is unavailable for this fit (rank-deficient design)"
                )
            pearson = (y - mu) * np.sqrt(prior_w) / np.sqrt(mu**p)
            X = self._design_matrix_from_info(data)
            w_irls = prior_w * mu ** (2 - p)
            xtwx_inv = self.cov_params_.to_numpy() / self.dispersion_
            h = w_irls * np.einsum("ij,jk,ik->i", X, xtwx_inv, X)
            h = np.clip(h, 0.0, 1.0 - 1e-10)
            res = pearson / np.sqrt(self.dispersion_ * (1.0 - h))
        else:
            raise ValueError(
                f"unknown residual kind {kind!r}; "
                "expected 'deviance', 'pearson', 'standardized', or 'response'"
            )
        return pd.Series(res, index=data.index, name=f"{kind}_residual")

    def relativity_table(self, confidence_level: float = 0.95) -> pd.DataFrame:
        r"""Every fitted relativity with its confidence interval, in one table.

        The interval is computed on the coefficient scale and exponentiated:
        :math:`\exp(\hat\beta \pm z_{\alpha}\,\mathrm{se})`, using the
        quasi-likelihood standard errors (Pearson dispersion). Base levels
        appear with relativity 1.0 and no interval -- the reference is fixed
        by construction, not estimated. Continuous covariates appear under
        level ``"(per +1)"``: the multiplicative effect of a one-unit
        increase.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(variable, level)`` with columns ``coef``, ``se``,
            ``relativity``, ``ci_low``, ``ci_high``, ``is_base``.
        """
        if self.coefficients_ is None:
            raise RuntimeError("model is not fit")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        info = self._design_info_
        coefs = self.coefficients_.to_numpy()
        ses = (
            self.se_.to_numpy()
            if self.se_ is not None
            else np.full(len(coefs), np.nan)
        )

        def _row(variable, level, j, is_base=False):
            if is_base:
                return (variable, level, 0.0, np.nan, 1.0, np.nan, np.nan, True)
            coef, se = float(coefs[j]), float(ses[j])
            lo, hi = (
                (np.exp(coef - z * se), np.exp(coef + z * se))
                if np.isfinite(se)
                else (np.nan, np.nan)
            )
            return (variable, level, coef, se, float(np.exp(coef)), lo, hi, False)

        spec = info["spec"]
        rows = []
        for var in info["predictors"]:
            rows.append(_row(var, info["base_levels"][var], None, is_base=True))
            for j, term in enumerate(spec):
                if term[0] == "level" and term[1] == var:
                    rows.append(_row(var, term[2], j))
        for var in info["continuous"]:
            for j, term in enumerate(spec):
                if term[0] == "cont" and term[1] == var:
                    rows.append(_row(var, "(per +1)", j))
        for j, term in enumerate(spec):
            if term[0] == "ixcc":
                _, a, la, b, lb = term
                rows.append(_row(f"{a}:{b}", f"{la} | {lb}", j))
            elif term[0] == "ixcont":
                _, a, la, c = term
                rows.append(_row(f"{a}:{c}", f"{la} (per +1)", j))
        out = pd.DataFrame(
            rows,
            columns=[
                "variable", "level", "coef", "se",
                "relativity", "ci_low", "ci_high", "is_base",
            ],
        ).set_index(["variable", "level"])
        return out

    def to_factor_tables(self) -> dict:
        """The fitted categorical relativities as :class:`FactorTable` objects.

        The bridge from estimation to application: each rating variable
        becomes a named lookup that plugs directly into the build-up and
        renewal machinery, with ``default=1.0`` for unknown levels --
        matching how :meth:`predict` treats levels unseen at fit time.
        Continuous covariates and interaction terms have no single-variable
        level->factor form and are not included; read their effects from
        :meth:`relativity_table` (and cat x cat cells from
        ``relativities_["a:b"]``).

        Returns
        -------
        dict of str -> FactorTable
            One table per categorical predictor, keyed by variable name.
        """
        if self.coefficients_ is None:
            raise RuntimeError("model is not fit")
        main = set(self._design_info_["predictors"])
        return {
            var: FactorTable(name=var, factors=dict(rels), default=1.0)
            for var, rels in self.relativities_.items()
            if var in main
        }

    def summary(self) -> pd.DataFrame:
        """Coefficient table: estimate, quasi-likelihood SE, z, relativity.

        Standard errors use the Pearson-estimated dispersion (quasi-likelihood
        / quasi-Poisson style), which is the robust default for pricing data
        where overdispersion is the norm.
        """
        if self.coefficients_ is None:
            raise RuntimeError("model is not fit")
        out = pd.DataFrame({"coef": self.coefficients_})
        if self.se_ is not None:
            out["se"] = self.se_
            with np.errstate(divide="ignore", invalid="ignore"):
                out["z"] = out["coef"] / out["se"]
        out["relativity"] = np.exp(out["coef"])
        return out


# --------------------------------------------------------------------------- #
# Credibility-smoothed relativities and sparse-level handling
# --------------------------------------------------------------------------- #
def credibility_relativities(
    data: pd.DataFrame,
    factor: str,
    response: str,
    exposure: str | None = None,
    prior=1.0,
    method: str = "buhlmann",
    full_credibility: float | None = None,
    base_level=None,
) -> pd.DataFrame:
    r"""One-way relativities shrunk toward a prior by credibility, per level.

    Sparse levels produce unstable observed relativities; the classical
    actuarial answer is not to drop them or regularize generically but to
    credibility-weight them against a complement:

    .. math::
        \text{relativity}_\ell = Z_\ell \cdot \text{observed}_\ell
        + (1 - Z_\ell) \cdot \text{prior}_\ell .

    Parameters
    ----------
    data : DataFrame
        One row per observation.
    factor, response : str
        The rating variable to smooth and the response column.
    exposure : str, optional
        Exposure column. Level weights and the observed relativities are
        exposure-weighted when given; otherwise each row has weight 1.
    prior : float, mapping, or Series
        The complement of credibility *on the relativity scale*. The default
        1.0 shrinks toward "no effect"; a mapping/Series (e.g. the current
        filed factors) shrinks each level toward its existing relativity.
        Levels missing from a mapping fall back to 1.0.
    method : {"buhlmann", "limited_fluctuation"}
        How :math:`Z_\ell` is estimated:

        * ``"buhlmann"`` (default) -- empirical Bühlmann-Straub across the
          levels of ``factor`` (each row is one observation of its level),
          via :func:`ratingmodels.buhlmann_straub` /
          :class:`actuarialpy.BuhlmannStraub`. Greatest-accuracy credibility:
          :math:`Z = w/(w + k)` with :math:`k` estimated from the data.
        * ``"limited_fluctuation"`` -- the square-root rule
          :math:`Z = \min(1, \sqrt{n_\ell / n_{\text{full}}})` where
          :math:`n_\ell` is the level's total ``response`` and
          ``full_credibility`` is the full-credibility standard in the same
          units (for claim counts, e.g.
          :func:`ratingmodels.full_credibility_standard`).
    full_credibility : float, optional
        Required when ``method="limited_fluctuation"``.
    base_level : optional
        When given, the ``observed`` and ``relativity`` columns are each
        rebased so this level equals 1.0.

    Returns
    -------
    pandas.DataFrame
        Indexed by level with columns ``n``, ``exposure``, ``response``,
        ``observed``, ``credibility``, ``prior``, ``relativity``.

    Notes
    -----
    With the default scalar prior of 1.0, the Bühlmann-Straub form is exactly
    the credibility-weighted mean divided by the collective mean: shrinking
    the relativity toward 1 and shrinking the level mean toward the overall
    mean are the same operation.
    """
    from .credibility import buhlmann_straub, limited_fluctuation_credibility

    if factor not in data.columns:
        raise ValueError(f"factor column {factor!r} not found")
    grp = data.groupby(factor, sort=True)
    n = grp.size().rename("n")
    resp_sum = grp[response].sum().rename("response")
    if exposure is None:
        expo_sum = n.astype(float).rename("exposure")
    else:
        expo_sum = grp[exposure].sum().rename("exposure")
    if (expo_sum <= 0).any():
        bad = expo_sum.index[expo_sum <= 0][0]
        raise ValueError(f"level {bad!r} has non-positive total exposure")

    level_mean = resp_sum / expo_sum
    overall_mean = float(resp_sum.sum() / expo_sum.sum())
    if overall_mean <= 0:
        raise ValueError("overall mean response must be positive")
    observed = (level_mean / overall_mean).rename("observed")

    meth = method.lower()
    if meth == "buhlmann":
        work = pd.DataFrame(
            {
                "_level_": data[factor].to_numpy(),
                "_period_": grp.cumcount().to_numpy(),
                "_weight_": (
                    data[exposure].to_numpy(dtype=float)
                    if exposure is not None
                    else np.ones(len(data))
                ),
            }
        )
        work["_value_"] = data[response].to_numpy(dtype=float) / work["_weight_"]
        try:
            bs = buhlmann_straub(
                work, group="_level_", period="_period_",
                value="_value_", exposure="_weight_",
            )
        except Exception as exc:  # pragma: no cover - message pass-through
            raise ValueError(
                "Bühlmann-Straub estimation failed (levels may have too few "
                "observations to estimate the within-level variance); "
                "consider method='limited_fluctuation'"
            ) from exc
        if not np.isfinite(bs.k) or bs.k < 0:
            raise ValueError(
                "Bühlmann-Straub produced a non-finite credibility constant; "
                "consider method='limited_fluctuation'"
            )
        z = bs.credibility.reindex(observed.index)
    elif meth == "limited_fluctuation":
        if full_credibility is None or full_credibility <= 0:
            raise ValueError(
                "method='limited_fluctuation' requires a positive "
                "full_credibility standard in response units"
            )
        z = limited_fluctuation_credibility(resp_sum, full_credibility)
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'buhlmann' or 'limited_fluctuation'"
        )
    z = pd.Series(np.clip(np.asarray(z, dtype=float), 0.0, 1.0), index=observed.index)

    if isinstance(prior, pd.Series):
        prior_s = prior.reindex(observed.index).fillna(1.0).astype(float)
    elif isinstance(prior, Mapping):
        prior_s = pd.Series(
            [float(prior.get(lvl, 1.0)) for lvl in observed.index], index=observed.index
        )
    else:
        prior_s = pd.Series(float(prior), index=observed.index)

    relativity = z * observed + (1.0 - z) * prior_s

    out = pd.DataFrame(
        {
            "n": n,
            "exposure": expo_sum,
            "response": resp_sum,
            "observed": observed,
            "credibility": z.rename("credibility"),
            "prior": prior_s.rename("prior"),
            "relativity": relativity.rename("relativity"),
        }
    )
    out.index.name = factor
    if base_level is not None:
        if base_level not in out.index:
            raise ValueError(f"base level {base_level!r} not found among levels")
        for col in ("observed", "relativity"):
            base_val = out.loc[base_level, col]
            if base_val <= 0:
                raise ValueError(f"base level {col} must be positive to rebase")
            out[col] = out[col] / base_val
    return out


def collapse_sparse_levels(
    levels,
    exposure=None,
    min_exposure: float | None = None,
    min_n: int | None = None,
    other_label="Other",
):
    """Recode levels below an exposure or count threshold into one bucket.

    The blunt companion to :func:`credibility_relativities`: rather than
    shrinking a thin level's relativity, fold the level into ``other_label``
    before fitting, so the design matrix never carries columns the data
    cannot support.

    Parameters
    ----------
    levels : array-like
        The categorical column (Series in, Series out on the same index).
    exposure : array-like, optional
        Aligned exposure; level totals are sums of this when given, row
        counts otherwise.
    min_exposure, min_n : float / int, optional
        Keep a level only if its total exposure is at least ``min_exposure``
        and its row count at least ``min_n``. At least one must be given.
    other_label
        Label assigned to collapsed levels. Must not already be a kept level.

    Returns
    -------
    (recoded, summary)
        ``recoded`` -- the recoded labels (Series if ``levels`` was a Series,
        else an ndarray). ``summary`` -- a DataFrame indexed by original
        level with columns ``n``, ``exposure``, ``collapsed``; apply the same
        recode to future data by mapping levels where ``collapsed`` is True.
    """
    if min_exposure is None and min_n is None:
        raise ValueError("give min_exposure and/or min_n")
    is_series = isinstance(levels, pd.Series)
    lab = levels if is_series else pd.Series(np.asarray(levels))
    if exposure is None:
        w = pd.Series(np.ones(len(lab)), index=lab.index)
    else:
        w = (
            exposure.astype(float)
            if isinstance(exposure, pd.Series)
            else pd.Series(np.asarray(exposure, dtype=float), index=lab.index)
        )
        if len(w) != len(lab):
            raise ValueError("exposure must match levels in length")
        if (w < 0).any():
            raise ValueError("exposure must be nonnegative")

    frame = pd.DataFrame({"_lvl_": lab.to_numpy(), "_w_": w.to_numpy()})
    agg = frame.groupby("_lvl_", sort=True)["_w_"].agg(["size", "sum"])
    agg.columns = ["n", "exposure"]
    keep = pd.Series(True, index=agg.index)
    if min_exposure is not None:
        keep &= agg["exposure"] >= min_exposure
    if min_n is not None:
        keep &= agg["n"] >= min_n
    agg["collapsed"] = ~keep
    agg.index.name = None

    if agg["collapsed"].all():
        raise ValueError("threshold collapses every level; nothing would remain")
    if other_label in agg.index[~agg["collapsed"]]:
        raise ValueError(
            f"other_label {other_label!r} is already a kept level; choose another"
        )

    collapsed_set = set(agg.index[agg["collapsed"]])
    recoded_values = np.array(
        [other_label if v in collapsed_set else v for v in lab.to_numpy()], dtype=object
    )
    if is_series:
        recoded = pd.Series(recoded_values, index=levels.index, name=levels.name)
    else:
        recoded = recoded_values
    return recoded, agg
