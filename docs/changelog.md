# Changelog

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
