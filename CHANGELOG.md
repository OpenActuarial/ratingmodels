# Changelog

## 0.8.0

Worksheet construction from the canonical Experience.

- Add `ExperienceRate.from_experience(exp, ...)`: sums the bound expense and
  exposure roles, and with `pooling_point`/`claimant_col` caps each
  claimant's total to derive `pooled_excess`. Trend, charges, factors, and
  retention stay caller-supplied judgment.
- Add `experience_rate(exp, by=...)`: one worksheet row per segment as a tidy
  frame, or the single `ExperienceRate` without `by`.
- Add `ExperienceExhibit.from_experience(exp, ...)`: premium from the
  `revenue` role and losses from the `expense` role, summed per period of
  the bound date; on-level, development, and trend factors stay explicit.
- Scalar constructors are unchanged.
- Requires `actuarialpy>=0.45`.
- `base_rate_from_experience` accepts the canonical `Experience`, resolving
  the exposure and loss columns from its bound roles.
- `ExperienceRate.from_experience` gains `expense=` to select which bound
  expense columns constitute claims on multi-expense Experiences.

## 0.7.3 - 2026-07-11

### Added
- **Contract-pinned loss ratios as named constructors.** Some experience-rated
  contracts fix a loss ratio and solve premium from it; both standard pins were
  already reachable through field arithmetic on `RetentionLoad`, and are now
  first-class vocabulary:
  - `RetentionLoad.from_gross_loss_ratio(lr, variable_items=None)` — the
    contract fixes claims/premium, so `P = C/LR*`. Optional itemization splits
    the pinned retention with the remainder landing in `profit_margin`; items
    exceeding `1 - LR*` raise.
  - `RetentionLoad.from_net_loss_ratio(lr, fixed_expense=0, variable_items=None)`
    — the contract fixes claims/(premium − expenses), so
    `P = (C/LR* + F)/(1 − V)`. The `1/LR*` gross-up is carried through the
    percent-of-claims (`lae_ratio`) slot; the docstring states the mapping.
- `RetentionLoad.implied_net_loss_ratio(loss_cost)` — claims over premium net
  of fixed and variable expenses (profit stays in the denominator). Returns the
  contractual ratio identically for `from_net_loss_ratio` instances, making the
  contract check a one-liner.
- Example `examples/contract_loss_ratios.py` covering both pins and a book of
  pinned-ratio groups in columns.
- Regression test pinning the ratingmodels segment of docs Example 10
  (`worked-example-contract.md`): the contract-pinned charged rates and
  renewal actions, the gross-pin action ≡ claims-trend identity, and the
  net-pin closed form.

## 0.7.2 - 2026-07-10

### Changed
- The `actuarialpy` dependency is now an open floor (`>=0.44`) instead of the
  compatible-release pin (`~=0.44.0`), which made `ratingmodels`
  co-uninstallable with every future `actuarialpy` minor release. Open
  floors are now the ecosystem policy; runtime drift is caught by the
  nightly ecosystem smoke workflow.

## 0.7.1 - 2026-07-09

Compatibility release for actuarialpy 0.44 — no functional changes.

### Changed
- **actuarialpy pin raised to `~=0.44.0`** (was `~=0.40.0`). actuarialpy
  0.42–0.44 moved the experience-study layer (`Experience`, the summaries,
  `UnderwritingSummary`, `to_excel_report`) into the new `experiencestudies`
  package; nothing `ratingmodels` delegates to (credibility, trend,
  time-value math) changed. The full test suite passes against 0.44.
- The worked-example regression test now imports `Experience` from
  `experiencestudies` (added to the `dev` extra); the pinned page numbers
  are unchanged.

## 0.7.0 - 2026-07-05

Frequency-severity parity with the GLM layer's uncertainty surface.

### Added
- **Frequency-severity prediction intervals.**
  `FrequencySeverityModel.predict_interval` -- the same delta-method
  interval `GLMRelativities` has, for the composite: log-scale variances
  of the two component linear predictors add, under the stated (and it
  is an assumption, stated as such) independence of the frequency and
  severity coefficient estimates. `predicted` equals
  `pure_premium_prediction` exactly; switching from a GLM to a
  frequency-severity model no longer silently loses uncertainty.
## 0.6.1 - 2026-07-04

### Added
- **Frequency-severity interactions.** `FrequencySeverityModel.fit`
  gains `frequency_interactions` / `severity_interactions` (severity
  defaults to the frequency list, mirroring the predictor convention),
  threading straight through to the component GLMs. Categorical x
  categorical cells surface in `combined_relativities()` under an
  `"a:b"` key with a MultiIndex of level pairs (`combined` = frequency x
  severity per cell; a component without the interaction contributes
  1.0); `to_factor_tables()` excludes interactions like the GLM does, and
  `RatingPlan.from_model` warns accordingly.
## 0.6.0 - 2026-07-04

Validation and implementation release: a fitted model is now inspectable,
testable, and *deployable*. GLM estimation is delegated to statsmodels while
ratingmodels keeps the actuarial layer; the model exposes residuals,
relativity intervals, and prediction intervals; interactions join the design
vocabulary; out-of-sample evaluation has first-class tables and leakage-safe
splits; and the path from fit to filed change is now complete objects rather
than conventions -- `RatingPlan` for the implemented plan, plan comparison
and dislocation for the change, on-level factors for restating history, and
pooling charges from any fitted severity tail. Everything is domain-agnostic
and follows the columns-in, columns-out contract.

### Added (implementation layer)
- **Interaction terms.** `fit(..., interactions=[("area", "industry")])`
  supports categorical x categorical and categorical x continuous pairs
  (order-agnostic; each member must also enter as a main effect).
  Cat x cat uses treatment coding -- an indicator per *observed* non-base x
  non-base level pair, so main effects keep their interpretation and
  unobserved cells cannot silently alias the design -- and surfaces as a
  MultiIndex relativity table in `relativities_["a:b"]` plus
  `"la | lb"` rows in `relativity_table()`. Cat x continuous fits one
  slope modifier per non-base level (`"la (per +1)"` rows).
  `to_factor_tables()` deliberately excludes interactions (a `FactorTable`
  is single-variable by contract).
- **Structured design spec.** The fitted design is now recorded as a
  structured column spec rather than parseable coefficient names, and
  `predict` is unified through the same design-matrix path as `residuals`
  and the statsmodels cross-checks -- one code path from fit to score.
  (Internal, but it is why interaction names like `area::B:industry::tech`
  cannot be misread as levels of `area`.)
- **`predict_interval`.** Delta-method confidence intervals for the fitted
  mean on any frame, from the quasi-likelihood coefficient covariance;
  matches `statsmodels` `get_prediction` to numerical precision. The
  docstring is explicit that this is an interval for the *rate*, not for
  individual outcomes.
- **`RatingPlan`.** The implemented plan as one object: base rate plus a
  `FactorTable` per variable. `rate(census)` returns the full decomposed
  build-up per row; `validate(census)` lists every level the plan has no
  factor for; `unknown="error"` makes unmapped levels a hard stop instead
  of a silent 1.0; `average_relativity` is the off-balance diagnostic;
  `to_dict`/`from_dict` round-trip (schema-versioned) for filing and
  version control; `RatingPlan.from_model(...)` builds a plan straight
  from a fitted `GLMRelativities` or `FrequencySeverityModel`, and
  `plan.rate(...)["premium"]` reproduces `model.predict(...)` exactly.
- **`compare_rating_plans`.** Rates one census under two plans and returns
  a `PlanComparison`: `summary()` (premium totals, average change,
  exposure shares moving up/down), `dislocation()` (the banded exhibit via
  `rate_dislocation`), and `by(labels)` (who absorbs the move).
- **`on_level_factors`.** The parallelogram method made exact: the earned
  rate index is integrated in closed form (piecewise-linear geometry, no
  discretization), with `policy_term` controlling the parallelogram
  (`0` = instant earning, `1.0` = the classic annual case -- the textbook
  +10%-mid-year example reproduces 1.1/1.0125 to machine precision).
  Accepts float or datetime inputs.
- **`ExperienceExhibit`.** `RateIndication` consumes point inputs -- a
  trended, developed loss cost and an on-level premium; this is the object
  that assembles them from per-period columns, every adjustment a visible
  worksheet column (premium x on-level factor, losses x development x
  trend, per-period and weighted loss ratios). `to_indication(...)` wires
  the totals straight into `RateIndication`, and the identity is exact by
  construction: the indication's own `experience_loss_ratio()` reproduces
  the exhibit's aggregate ratio, and at full credibility the indicated
  rate *is* `retention.gross_rate(...)` of the assembled loss cost --
  one expense algebra, no parallel implementation. Natural factor
  producers: `on_level_factors` and `actuarialpy.reserving.ChainLadder`.
- **`pooling_charge_from_severity`.** Where `experience_rate`'s
  `pooling_charge` input comes from: expected excess cost above a pooling
  point from any severity object exposing `sf(x)` and `mean_excess(d)`,
  returned as an auditable build-up (exceedance probability, mean excess,
  pure cost, loaded charge). The protocol is duck-typed: `lossmodels`
  distributions and `extremeloss` GPD tail fits both qualify, with no
  cross-package dependency.

### Changed
- **GLM estimation now delegates to `statsmodels.GLM`** (new runtime
  dependency: `statsmodels>=0.14`; the in-package IRLS solver is removed).
  A mature estimator owns the numerics — solver, convergence, covariance,
  and the fitted null model — while ratingmodels owns what is actuarial:
  the design encoding and base-level semantics, coefficient-to-relativity
  conversion, prediction with unseen-level fallback, residuals on arbitrary
  frames, and the exhibits. The `GLMRelativities` API is unchanged;
  dispersion is estimated from the Pearson chi-square (`scale="X2"`,
  quasi-likelihood) for every family, matching the 0.5.x convention. The
  fitted results object is exposed as `results_`, so nothing statistical is
  walled off (`results_.get_influence()`, `results_.get_prediction(...)`,
  Wald tests, ...). Perfect (saturated) fits report `converged_ = True`
  rather than inheriting statsmodels' perfect-separation flag. The family
  deviance and variance-power math stays in-package because evaluation on
  arbitrary frames needs it (`residuals`, `compare_models`).

### Added
- **GLM diagnostics.** `GLMRelativities.residuals(data, kind=...)` returns
  per-row `deviance`, `pearson`, `standardized` (leverage-adjusted via the
  IRLS hat values), or raw `response` residuals as an index-aligned Series;
  column names default to those used at fit. Squared Pearson and deviance
  residuals reproduce `pearson_chi2_` and `deviance_` exactly.
  `relativity_table(confidence_level=0.95)` reports every relativity with its
  quasi-likelihood confidence interval, `exp(coef ± z·se)`, in one tidy
  `(variable, level)` frame — base levels shown at 1.0 with no interval,
  continuous covariates as per-unit factors. `deviance_explained_` exposes
  `1 - deviance/null_deviance`.
- **Validation tables.** `calibration_table` bands records by prediction at
  equal exposure and reports per-band A/E — actual and predicted treated
  symmetrically, on the total scale. `actual_expected_table` is the segment
  exhibit: totals, per-unit means, and A/E overall, by one variable, or by
  many at once (tidy `(variable, level)` output). `compare_models` scores
  fitted GLMs side by side on one frame — deviance, deviance explained,
  Gini, A/E, calibration error — for honest out-of-sample comparison.
- **Validation splits.** `random_split`, `group_split` (groups stay whole on
  one side; optional exposure weighting of the target share), and
  `temporal_split` (out-of-time at a cutoff). `(train, test)` DataFrames with
  row order preserved; empty sides raise rather than pass silently. No
  scikit-learn dependency.
- **Frequency–severity models.** `FrequencySeverityModel` composes a count
  GLM (Poisson default, exposure offset) and a severity GLM (Gamma default,
  fit on claim rows only, count-weighted) into a pure-premium model:
  `frequency_prediction`, `severity_prediction`,
  `pure_premium_prediction` (exactly their product), per-variable
  `combined_relativities()` (frequency x severity), `base_value_`, and a
  stacked `summary()`. Rows with amounts but no counts raise; claims closed
  at zero are excluded from severity with a warning.
- **Credibility-smoothed relativities.** `credibility_relativities` shrinks
  one-way factors toward a prior: `Z·observed + (1-Z)·prior`, with `Z` from
  empirical Bühlmann–Straub across levels (via the `actuarialpy` estimators)
  or the limited-fluctuation square-root rule. Returns level weight,
  observed, credibility, prior, and blended relativity in one frame;
  optional rebase to a base level. `collapse_sparse_levels` folds levels
  below an exposure/count threshold into an `"Other"` bucket and returns the
  recode plus a summary for applying it to future data.
- **Rate dislocation.** `rate_dislocation` bands a book by rate change
  (empty bands kept, boundary changes snap to edges within float tolerance,
  `(low, high]`) with premium, exposure share, and premium-weighted average
  change per band. `constraint_impact` quantifies the indicated-vs-proposed
  gap: premium shortfall/excess, cases capped, realized vs indicated change,
  and the remaining rate action still owed — with `by=` for per-segment
  attribution.
- `datasets.sample_frequency_severity_data` — synthetic claims with
  *different* frequency and severity structure, so component recovery is
  testable; true severity relativities exported alongside.
- **`to_factor_tables()`** on `GLMRelativities` (per categorical predictor)
  and `FrequencySeverityModel` (from the combined pure-premium
  relativities): the bridge from estimation to application — named
  `FactorTable` lookups with `default=1.0` for unknown levels, matching
  `predict`'s unseen-level fallback, ready for the build-up and renewal
  machinery.
- **Adapter contract tests.** The suite fits statsmodels *independently*
  (its own family objects, offset construction, weights) on the exact
  design matrix `GLMRelativities` built, and asserts the marshaling
  conventions and the in-package evaluation math — residuals, relativity
  intervals, family deviance — agree across Poisson, Gamma, and Tweedie.

### Fixed
- `null_deviance_` (and therefore `deviance_explained_`) for **non-Poisson
  families with offsets/exposure**: 0.5.x used the weighted mean rate
  `sum(wy)/sum(w·e^o)` as the null model — the intercept-only MLE for
  Poisson but not for Gamma/Tweedie. The null deviance now comes from the
  actually fitted intercept(+offset)-only model, correct for every family
  (Poisson results unchanged). Found while cross-checking the old solver
  against statsmodels; a contract test pins the behavior.
- **Docs:** the guidance that exposure-as-log-offset is "the natural choice
  for counts and pure premium" was corrected. Offsets are for *aggregate*
  responses (counts, total amounts); a *rate* response (already divided by
  exposure) takes exposure as variance `weights` instead. The two coincide
  only for Poisson (p = 1); the weights form is the one consistent with a
  response averaged over independent claims, and is exactly how the
  severity component of `FrequencySeverityModel` is fit.

### Notes
- Standard errors and intervals remain quasi-likelihood (Pearson dispersion)
  throughout; no penalized fits are offered because shrinkage would
  invalidate that covariance — credibility smoothing is the supported
  stabilizer. Should regularization at scale ever become a requirement,
  `glum` is the designated engine for that job, behind this same API.

## 0.5.1 - 2026-07-03

### Fixed
- The trend functions (`trend_factor`, `apply_trend`, `combine_trend`,
  `split_total_trend`) raised `TypeError` on plain Python lists/tuples;
  every array-like is now coerced to ndarray up front, matching the rest of
  the library.

### Added
- Worked-example regression test pinning the docs-site
  "pricing a book, in columns" page numbers (`test_worked_example_book.py`),
  alongside a list-input regression test across the trend functions.

## 0.5.0 - 2026-07-03

Vectorization release: the whole per-risk layer now follows one contract --
scalar in, float out (unchanged behavior); Series/array in, Series/array out,
elementwise, with pandas indexes preserved and scalars broadcasting. A book
prices in one call.

### Added
- **Vectorized rating.** `ManualRate`, `ExperienceRate`, `RateIndication`,
  `PricingEvaluation`, `RetentionLoad`, and `RenewalAction`/`renew` accept
  Series/array values in every numeric field; derived quantities
  (`loss_cost`, `rate`, `indicated_rate_change`, margins, ...) return Series
  on the shared index. `blend`, credibility factors, trend, loading,
  constraint, and off-balance functions are elementwise, and `trend`'s date
  helpers accept datetime Series for per-row periods.
- **Vectorized build-ups.** `BuildUp`/`evaluate` accept vector operands
  (including per-row `segment_multiply` weights and `participation_blend`
  shares); `value` and subtotals return as Series and `breakdown` switches
  to tidy long format with an `entity` column.
- **Vectorized decomposition.** `decompose_rate_change` accepts vector
  drivers/totals; `factors` and `contributions` become per-case DataFrames
  and `to_frame()` stacks to a tidy `(case, driver)` table. Flows through
  `RateIndication.rate_change_decomposition()`.
- **Vectorized scenarios.** A `PricingEvaluation` built from columns is the
  book: `at()` evaluates every case at once, `ScenarioOutcome.to_frame()`
  returns one tidy row per case, and `scenario_frame` /
  `uplift_for_target_margin` accept a vector evaluation directly (the
  uplift solve agrees with the mapping form to floating point).
- **Grouped aggregations.** `by=` on `base_rate_from_experience` (a
  DataFrame of base rates, one per segment), `average_relativity`,
  `aggregate_demographic_factor`, `pool_claims` / `expected_excess_charge`,
  `gini_coefficient`, and `lift_table` (per-group tables under a
  `(group, band)` MultiIndex).
- `RenewalAction.to_frame()` for tidy renewal runs; per-row caps/floors in
  `cap_change` / `apply_cap` / `renew`.
- `FactorTable.apply` preserves a Series index.
- Elementwise validation everywhere: one bad row fails the call and the
  error names the offending index label. Reducing helpers (`product`, the
  build-up engine, `blend`, trend) raise on mismatched Series indexes
  instead of silently aligning to NaN.

### Fixed
- `renew` marked an action as `capped` whenever cent-rounding moved the
  rate; the flag now reflects only a binding cap or floor.
- `ManualRate.with_factor` silently dropped `retention`; it is now carried
  to the new instance.
- `_utils.product` reduced over *all* elements when handed multiple factor
  vectors, which made naive vectorization of the manual-rate path return
  plausible but wrong numbers; it now reduces across factors elementwise.
- `unit_level_renewal` and the `factor_cols` path of
  `base_rate_from_experience` no longer iterate rows in Python.

### Changed
- `combine_trend` / `split_total_trend` parameters renamed to the
  standardized decomposition vocabulary: `util_trend` / `cost_trend` are now
  `frequency_trend` / `severity_trend` (matching actuarialpy 0.38's
  `frequency_trend * severity_trend` identity). Positional calls are
  unaffected; keyword calls need the new names.
- `participation_blend`'s default checkpoint label is now
  `"Blended Claim Cost"` (was the health-specific `"PPO Claim Cost"`);
  pass `label=` for a domain name. Docstrings and the README complete the
  domain-agnostic vocabulary sweep -- health terms remain only as attributed
  examples of caller-side dialects.
- Version floor unchanged; no other public API removed. Scalar calls return
  the same types and values as 0.4.x throughout.

## 0.4.4

### Added

- Pricing-algebra, GLM/evaluation, and build-up/renewal test suites; a
  `pricing_scenarios` example; example scripts are executed by the test
  suite.
- Worked-example regression test for the experience-to-renewal page.

### Changed

- More descriptive package `description` metadata.

## 0.4.2

### Changed

- Depend on `actuarialpy~=0.38.0`, whose decomposition components are renamed
  to `frequency_trend` / `severity_trend` / `frequency_effect` /
  `severity_effect`.

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
