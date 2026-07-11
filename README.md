# ratingmodels

Manual and experience rating, credibility blending, and rate indication with an audit trail.

[![CI](https://github.com/OpenActuarial/ratingmodels/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenActuarial/ratingmodels/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ratingmodels)](https://pypi.org/project/ratingmodels/)
[![Python](https://img.shields.io/pypi/pyversions/ratingmodels)](https://pypi.org/project/ratingmodels/)

## Overview

`ratingmodels` implements the pieces of a pricing exercise as small,
inspectable objects: an experience rate with pooling and trend, a manual rate
with factor application, credibility blending, a rate indication that
decomposes exactly into its drivers, renewal capping, and rate build-up with
a line-by-line audit trail.

It also covers the modeling side of pricing — GLM relativities, a
frequency–severity model with prediction intervals, on-leveling, dislocation,
and scenario evaluation — all against tidy inputs.

## Installation

```bash
pip install ratingmodels
```

Requires Python 3.10 or newer.

## Quick start

```python
import ratingmodels as rm

# experience side: pool large claims, trend, and target loss ratio
capped, excess = rm.pool_claims([612_000, 340_000, 128_000, 96_500],
                                pooling_point=250_000)
exp = rm.ExperienceRate(
    incurred_claims=4_200_000, exposure=9_600,
    trend_annual=0.075, trend_years=1.5,
    pooled_excess=excess, pooling_charge=38.00,
    target_loss_ratio=0.85,
)

# manual side, then blend by limited-fluctuation credibility
man = rm.ManualRate(base_loss_cost=480,
                    factors={"area": 1.05, "industry": 0.97, "tier": 1.10},
                    target_loss_ratio=0.85)
z = rm.limited_fluctuation_credibility(n=9_600, n_full=12_000)

ind = rm.RateIndication(
    experience_loss_cost=exp.loss_cost(), manual_loss_cost=man.loss_cost(),
    credibility=z, current_rate=520, target_loss_ratio=0.85,
    trend_total_factor=exp.trend_factor(),
    benefit_factor=1.00, demographic_factor=1.01,
)
print(f"indicated rate   : {ind.indicated_rate():.2f}")
print(f"indicated change : {ind.indicated_rate_change():+.2%}")

# why did the rate move, and what does a 5% cap do to it?
print(ind.rate_change_decomposition().to_frame())
action = rm.renew(current_rate=520, indicated_rate=ind.indicated_rate(), cap=0.05)
print(f"proposed (capped): {action.proposed_rate:.2f} ({action.proposed_change:+.2%})")
```

## What's inside

- **Experience rating** — `ExperienceRate` with pooling, trend, and
  target-loss-ratio gross-up.
- **Manual rating** — `ManualRate` with multiplicative factor application.
- **Credibility and blending** — limited-fluctuation credibility and
  experience/manual blends.
- **Indication** — `RateIndication` with an exact rate-change decomposition;
  renewal capping via `renew`.
- **Rate build-up** — layered build-up with a line-by-line audit trail; base
  rates and retention by the off-balance method.
- **Modeling** — `GLMRelativities`, a frequency–severity model with
  prediction intervals, on-leveling, dislocation, scenarios, and evaluation
  utilities.

The full API reference and end-to-end worked examples live at
**[openactuarial.org/ratingmodels.html](https://openactuarial.org/ratingmodels.html)**.

## The OpenActuarial ecosystem

`ratingmodels` is one of seven packages that share conventions — tidy tables,
explicit distribution parameterizations, reproducible random-number handling —
and compose across package seams:

| Package | Role |
|---|---|
| [actuarialpy](https://github.com/OpenActuarial/actuarialpy) | Calculation primitives the workflow packages build on |
| [experiencestudies](https://github.com/OpenActuarial/experiencestudies) | Experience reporting, actual-vs-expected, claimant and concentration analysis |
| [projectionmodels](https://github.com/OpenActuarial/projectionmodels) | Claim, premium, and expense projection over a renewal horizon |
| **[ratingmodels](https://github.com/OpenActuarial/ratingmodels)** | Manual and experience rating, credibility, indication, GLM relativities |
| [lossmodels](https://github.com/OpenActuarial/lossmodels) | Severity and frequency fitting, aggregate loss distributions |
| [extremeloss](https://github.com/OpenActuarial/extremeloss) | Extreme-value tails: POT/GPD, GEV, return levels, splicing |
| [risksim](https://github.com/OpenActuarial/risksim) | Portfolio Monte Carlo, dependence, reinsurance contracts, risk measures |

Install everything at once with `pip install openactuarial`.

## Development

```bash
git clone https://github.com/OpenActuarial/ratingmodels
cd ratingmodels
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

CI runs the same gate on Python 3.10–3.14 across Linux and Windows.

## Versioning and stability

All ecosystem packages are pre-1.0: minor releases may change APIs, and every
release is documented in [CHANGELOG.md](CHANGELOG.md). Current per-package API
stability is tracked at
[openactuarial.org/stability.html](https://openactuarial.org/stability.html).

## License

MIT — see [LICENSE](LICENSE).
