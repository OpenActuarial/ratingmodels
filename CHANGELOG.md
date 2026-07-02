# Changelog

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
