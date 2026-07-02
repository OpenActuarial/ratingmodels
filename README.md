# ratingmodels

**Actuarial pricing and rate-indication tools for experience-rated insurance portfolios.**

`ratingmodels` covers the group rating workflow — the step that turns experience
analysis and loss modeling into an actual rate. It answers the central pricing
question: **what rate should we charge, and why did it change?**

## What it does

- **Credibility** — limited fluctuation (square-root rule), Bühlmann, and
  empirical Bühlmann-Straub with exposure weights.
- **Trend** — midpoint-to-midpoint factors; utilization / unit-cost split.
- **Manual rating** — base rate × relativities, loaded to a charged rate.
- **Experience rating** — pooling of large claims, trend, pooling charge,
  benefit/demographic adjustments, loading.
- **Rate build-up** — an ordered, auditable evaluator (multiply / add-dollar /
  segment-conditional) with labeled subtotals and a reconciling breakdown, plus
  par/non-par participation blending and medical+drug combining. Supplies the
  *grammar* of a manual build-up; the factor values stay yours.
- **Base rate & off-balance** — indicated base loss cost from book experience
  (base × relativities reproduces book losses); off-balance correction and
  base rebalancing when relativities are revised.
- **Retention & gross-up** — charged rate from the fundamental insurance
  equation (loss & LAE, flat fixed expense, percent-of-premium loads, profit),
  with the target loss ratio as an *output*, not an input.
- **Blending & indication** — credibility-weighted blend; build-up and
  loss-ratio indication methods.
- **Rate-change decomposition** — multiplicative and percentage-point
  contribution-to-change with an explicit residual.
- **GLM relativities** — Poisson / Gamma / Tweedie GLMs fit by IRLS, so factors
  are estimated *jointly* (correcting for correlation between rating variables)
  rather than one-way. No statsmodels dependency — the IRLS is in-package.
- **Constraints & renewal** — rate caps/floors, banding, rounding, corridors,
  and member-level re-rating.

Dependencies are `numpy`, `pandas`, and `actuarialpy` (which supplies the shared credibility primitives).

## Installation

```bash
pip install ratingmodels
```

From source:

```bash
git clone https://github.com/OpenActuarial/ratingmodels
cd ratingmodels
pip install -e ".[dev]"
pytest
```

## Quick start

```python
import ratingmodels as rm

# --- experience side -------------------------------------------------------
capped, excess = rm.pool_claims(group_claims, pooling_point=250_000)
exp = rm.ExperienceRate(
    incurred_claims=4_200_000,
    exposure=96_000,            # member-months
    trend_annual=0.075,
    trend_years=1.5,            # experience midpoint -> rating midpoint
    pooled_excess=excess,
    pooling_charge_pmpm=4.00,
    target_loss_ratio=0.85,
)

# --- manual side -----------------------------------------------------------
man = rm.ManualRate(
    base_pmpm=480,
    factors={"area": 1.05, "industry": 0.97, "tier": 1.10},
    target_loss_ratio=0.85,
)

# --- credibility and indication -------------------------------------------
z = rm.limited_fluctuation_credibility(n=96_000, n_full=120_000)

ind = rm.RateIndication(
    experience_claims_pmpm=exp.claims_pmpm(),
    manual_claims_pmpm=man.claims_pmpm(),
    credibility=z,
    current_rate=560,
    target_loss_ratio=0.85,
    trend_total_factor=exp.trend_factor(),
    benefit_factor=1.00,
    demographic_factor=1.01,
)

print(f"indicated rate   : {ind.indicated_rate():.2f}")
print(f"indicated change : {ind.indicated_rate_change():+.2%}")

# why did the rate move?
print(ind.rate_change_decomposition().to_frame())

# apply a renewal cap
action = rm.renew(current_rate=560, indicated_rate=ind.indicated_rate(), cap=0.15)
print(f"proposed (capped): {action.proposed_rate:.2f} ({action.proposed_change:+.2%})")
```

### Rate build-up with an audit trail

```python
import ratingmodels as rm

med_par = rm.evaluate([
    rm.start("Par Base Claim Cost", 941.63),
    rm.add("$30 specialist copay", -11.44),
    rm.multiply("Rating Region", 1.083),
    rm.checkpoint("Medical Par Base Claim Cost"),
])
med_par.value          # final running total
med_par.breakdown      # DataFrame: step, operation, label, operand, running_total

# blend in-/out-of-network, then add the drug stream
med = rm.participation_blend(med_par.value, nonpar=1478.56, participation_rate=0.90)
total = rm.combine_streams({"Medical": med, "Drug": 323.67})
total.value            # feeds into trend / credibility / retention
```

The package supplies the build-up *grammar*; you supply the factor values
(cost-sharing, age/sex, area, ...) from your filed tables. `ManualRate` is a
thin shortcut over this engine, so `ManualRate(...).breakdown()` returns the
same audit trail.

### Base rate and retention

```python
import ratingmodels as rm
import pandas as pd

# indicated base loss cost from book experience (off-balance method)
book = pd.DataFrame({
    "exposure": [24_000, 18_000, 30_000, 24_000],
    "area":     [1.00, 1.20, 0.90, 1.05],
    "tier":     [1.00, 1.10, 0.95, 1.25],
    "loss":     [11_750_000, 11_400_000, 12_300_000, 15_200_000],
})
base = rm.base_rate_from_experience(book, "exposure", "loss",
                                    factor_cols=["area", "tier"])
base.base_loss_cost        # average loss cost / average relativity

# gross claims up to a charged rate; the loss ratio falls out
retention = rm.RetentionLoad.from_items(
    fixed_expense_pmpm=22.0,
    variable_items={"commission": 0.03, "premium_tax": 0.023, "aca_fees": 0.005},
    profit_margin=0.03,
)
retention.gross_rate(540.0)            # charged rate
retention.implied_loss_ratio(540.0)   # target loss ratio (an output)

# rebalance the base when relativities are revised (hold level, then +8%)
rm.rebalance_base_rate(current_base=base.base_loss_cost,
                       current_avg_relativity=1.0928, new_avg_relativity=1.12,
                       overall_change=0.08)
```

### GLM relativities

```python
import ratingmodels as rm
from ratingmodels.datasets import sample_rating_data

df = sample_rating_data(n=20_000)
model = rm.GLMRelativities(family="poisson").fit(
    df, response="claims", predictors=["area", "industry", "tier"],
    exposure="exposure",                 # enters as a log offset
    base_levels={"area": "A"},           # optional; defaults to modal level
)
print(model.base_value_)                 # fitted base frequency
print(model.relativities_["industry"])   # relativity per level, base = 1.0
```

## Scope and honest limitations

This is a modeling and workflow toolkit, not filed rate software. It does not
manage rate filings, store filed factor tables with effective dating, or enforce
state-specific rating rules. The pooling-charge helper is a simple group-level
estimate; a production charge is normally derived book-wide or from an EVT tail
model. All bundled data in `ratingmodels.datasets` is
synthetic and carries no assumptions.

## License

MIT. See [LICENSE](LICENSE).
