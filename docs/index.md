# ratingmodels

**Actuarial pricing and rate-indication tools for experience-rated insurance portfolios.**

Part of the [OpenActuarial](https://github.com/OpenActuarial) ecosystem.

`ratingmodels` covers the **group rating workflow** — the step that turns
experience analysis and loss modeling into an actual rate. It answers the central
pricing question directly: *what rate should we charge, and why did it change?*

| Package | Role |
| --- | --- |
| `actuarialpy` | experience analysis (PMPM, loss ratios, trend, completion) |
| **`ratingmodels`** | **pricing and rate indications** |
| `lossmodels` | loss-distribution modeling |
| `risksim` | portfolio Monte Carlo simulation |
| `extremeloss` | extreme-value tail estimation |

## Highlights

- **Credibility** — limited fluctuation, Bühlmann, empirical Bühlmann-Straub (delegated to `actuarialpy`).
- **Trend** — midpoint-to-midpoint factors with util / unit-cost split.
- **Manual & experience rating** — pooling, trend, loading, blending.
- **Indication** — build-up and loss-ratio methods in one object.
- **Rate-change decomposition** — exact contribution-to-change with residual.
- **GLM relativities** — Poisson / Gamma / Tweedie by in-package IRLS (no
  statsmodels dependency); estimates factors jointly, correcting for the
  correlation one-way analysis misses.
- **Constraints & renewal** — caps, banding, rounding, corridors, member-level
  re-rating.

Dependencies are `numpy`, `pandas`, and `actuarialpy` (which supplies the shared credibility primitives; see below).

## Install

```bash
pip install ratingmodels
```

## A complete indication

```python
import ratingmodels as rm

exp = rm.ExperienceRate(
    incurred_claims=4_200_000, exposure=96_000,
    trend_annual=0.075, trend_years=1.5,
    pooled_excess=350_000, pooling_charge_pmpm=4.0,
    target_loss_ratio=0.85,
)
man = rm.ManualRate(base_pmpm=480, factors={"area": 1.05, "industry": 0.97})
z = rm.limited_fluctuation_credibility(n=96_000, n_full=120_000)

ind = rm.RateIndication(
    experience_claims_pmpm=exp.claims_pmpm(),
    manual_claims_pmpm=man.claims_pmpm(),
    credibility=z, current_rate=560, target_loss_ratio=0.85,
    trend_total_factor=exp.trend_factor(),
)

ind.indicated_rate_change()          # proportional change
ind.rate_change_decomposition()      # why it moved
```

See **[Theory](theory.md)** for the mathematics and the **API reference** for
every function and class.
