# Changelog

## 0.4.1

### Changed

- Depend on `actuarialpy~=0.37.0`, which removes the core domain-sugar
  metrics and the automatic `_pmpm`-style output suffixes.
- Quickstart output labels are stated per exposure unit instead of PMPM.

## 0.4.0

### Changed

- Package-wide rename to domain-agnostic vocabulary before first release
  (0.3.0 was built but never published). The per-member-per-month suffix
  disappears from every required name: `experience_claims_pmpm` /
  `manual_claims_pmpm` / `blended_claims_pmpm` become
  `experience_loss_cost` / `manual_loss_cost` / `blended_loss_cost`;
  `ExperienceRate.claims_pmpm()` / `pooled_claims_pmpm()` become
  `loss_cost()` / `pooled_loss_cost()`; `pooling_charge_pmpm` ->
  `pooling_charge`; `base_pmpm` -> `base_loss_cost`; `manual_pmpm` ->
  `manual_loss_cost`; `RetentionLoad.fixed_expense_pmpm` ->
  `fixed_expense`. All rates and costs are documented per unit of exposure
  (member months, policy months, earned exposures, ...), with domain terms
  appearing only as examples. `member_level_renewal` becomes
  `unit_level_renewal` with `count_col` (default `"count"`) and a
  `unit_rate` output column.
- The `scenarios` module follows suit: `PricingEvaluation(loss_cost, ...,
  exposure=...)`, and `ScenarioOutcome` reports `premium_rate`,
  `loss_and_lae`, `expense_rate`, `gross_margin_rate`, and `margin_rate`
  per exposure unit alongside the dollar and persistency-weighted fields.
- The `actuarialpy` dependency pin moves to `~=0.36.0` alongside the core
  rename so the two packages stay co-installable.

## 0.3.0

### Added

- A `scenarios` module for management pricing around the indication:
  - `PricingEvaluation.at(rate_change)` evaluates a case at **any** rate
    action -- issued, post-concession, plan -- and reports premium, loss
    ratio, gross margin (benefit tier), margin after retention expense,
    and margin ratio, using the same expense algebra as `RetentionLoad`.
    At the indicated rate the margin ratio equals the retention's
    `profit_margin` exactly.
  - `premium_for_margin` / `rate_change_for_margin` generalize the
    indication's inverse solve to **any** margin target, closed-form:
    `P(m) = (L(1+lae) + F) / (1 - V - m)`. Zero-margin and plan-target
    premiums are this solve at `m = 0` and `m = plan`; the standard
    indication is the special case `m = profit_margin`.
  - Optional `member_months` and `persistency` on an evaluation produce
    dollar and renewal-probability-weighted (`expected_*`) outputs -- the
    deterministic counterpart of a retention Bernoulli.
  - `scenario_frame(cases, scenarios)` evaluates named actions across a
    book into one tidy long table (one row per case x scenario), so any
    cohort rollup or key-case exhibit is a groupby or pivot of library
    output. Scenario names are caller vocabulary; a per-case action
    mapping must cover every case (missing is an error, never a skip).
  - `uplift_for_target_margin` solves the exhibit input "rate actions
    must be X% higher to hold the target margin." Because the aggregate
    margin ratio is a ratio of functions affine in the uplift, both the
    multiplicative and additive solves are closed-form (the algebra is in
    the docstring), with explicit feasibility errors instead of a solver
    loop.

### Changed

- The `actuarialpy` dependency pin moves to `~=0.35.0` alongside the core
  release so the two packages stay co-installable from clean
  environments.

## 0.2.0

### Fixed
- **`GLMRelativities` converged after exactly one IRLS iteration** and
  therefore returned wrong coefficients for every model fit since the class
  was introduced. The deviance convergence test compared against an infinite
  initial deviance (`|dev - inf| <= tol * inf` is `inf <= inf`, which is
  true), so the loop always exited on the first pass. Fitted "estimates" were
  a single reweighted-least-squares step from the crude starting value --
  close enough to pass the existing tolerance-based tests, wrong enough to
  fail exact ones. The one-way Poisson MLE now matches observed rate ratios
  to 1e-8 and Poisson/Gamma/Tweedie fits match `statsmodels` reference
  results to at least 1e-5 (verification only; `statsmodels` is not a
  dependency). `n_iter_` reports true iterations and a `converged_` flag is
  exposed.
- The `actuarialpy` dependency pin is raised to `~=0.34.0` (the previous
  `~=0.33.0` pin excluded the current core release, making the two packages
  co-uninstallable from clean environments).
- Rank-deficient (aliased) designs no longer raise from `numpy.linalg.solve`;
  they fall back to a least-squares solution with a warning.
- Factor decomposition zips names and values strictly, surfacing length
  mismatches instead of silently truncating.

### Added
- **Continuous covariates in `GLMRelativities`** via `fit(...,
  continuous=[...])` -- age, trend, and other numeric rating variables enter
  the linear predictor directly alongside the categorical relativities.
- **Post-fit inference**: `se_` (quasi-likelihood standard errors using the
  Pearson-estimated dispersion, the robust default for overdispersed pricing
  data), `cov_params_`, `dispersion_`, `pearson_chi2_`, `null_deviance_`, and
  a `summary()` coefficient table (estimate, SE, z, relativity).
- `predict` rebuilt on the stored design (supports continuous covariates and
  an `offset=` column); unseen categorical levels fall back to the base level
  as before.
- **Model evaluation module** (`ratingmodels.evaluation`):
  `gini_coefficient` (exposure-weighted ordered-Lorenz Gini, normalized by
  the perfect model by default) and `lift_table` (equal-exposure bands with
  predicted/actual means and lift) -- the standard segmentation diagnostics
  for a rating plan.

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-29

Initial release.

### Changed
- **Credibility consolidated into `actuarialpy`.** Trend and credibility are
  shared ecosystem primitives; `actuarialpy` is the home for credibility. The
  credibility math is no longer duplicated here -- `ratingmodels.credibility`
  and `blend` are thin adapters over `actuarialpy` (`full_credibility_claims`,
  `limited_fluctuation_z`, `credibility_weighted_estimate`, and
  `BuhlmannStraub.from_frame`). The public `ratingmodels` API and results are
  unchanged. This removes the risk of the two Bühlmann-Straub implementations
  drifting apart, and `actuarialpy` now uses the general unbiased estimator
  (handling unequal period counts). Adds a dependency on `actuarialpy>=0.32.0`
  and drops the direct `scipy` dependency.

### Added
- **Credibility** (`credibility`): `full_credibility_standard`,
  `limited_fluctuation_credibility`, `buhlmann_credibility`, and empirical
  `buhlmann_straub` with exposure weights.
- **Trend** (`trend`): midpoint-to-midpoint trend factors, date helpers, and
  utilization / unit-cost decomposition.
- **Manual rating** (`manual_rate`): `ManualRate`, `manual_pmpm`,
  `aggregate_demographic_factor`.
- **Experience rating** (`experience_rate`): `ExperienceRate`, `pool_claims`,
  `expected_excess_charge`.
- **Rate build-up** (`buildup`): typed steps (`start`, `multiply`, `add`,
  `segment_multiply`, `checkpoint`), an `evaluate` engine and `BuildUp` fluent
  builder producing a reconciling breakdown, and `participation_blend` /
  `combine_streams` for combining par/non-par and medical+drug streams.
  `ManualRate` is reimplemented as a thin layer over the engine and gains
  `breakdown()` / `steps()`.
- **Base rate & off-balance** (`base_rate`): `base_rate_from_experience`,
  `average_relativity`, `off_balance_factor`, `rebalance_base_rate`.
- **Retention & loading** (`loading`): `RetentionLoad` (fundamental insurance
  equation gross-up), `gross_rate`, `permissible_loss_ratio`. `ManualRate`,
  `ExperienceRate`, and `RateIndication` accept an optional `retention` that
  overrides the single-loss-ratio path with the full expense/profit build-up.
- **Blending & indication** (`blend`, `indication`): `blend` and the
  `RateIndication` orchestrator with build-up and loss-ratio methods.
- **Rate-change decomposition** (`decomposition`): `decompose_rate_change`
  with multiplicative and percentage-point contributions and an explicit
  residual.
- **GLM relativities** (`relativity`): `GLMRelativities` (Poisson / Gamma /
  Tweedie via in-package IRLS), `FactorTable`, `one_way_relativities`.
- **Constraints & renewal** (`constraints`, `renewal`): caps, floors, banding,
  rounding, corridors; `renew` and `member_level_renewal`.
- **Synthetic data** (`datasets`): `sample_claims`, `sample_rating_data`.
- Full pytest suite (54 tests) and MkDocs Material documentation.
