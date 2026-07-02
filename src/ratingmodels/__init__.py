r"""ratingmodels -- actuarial pricing and rate-indication tools.

A small, dependency-light toolkit for the group rating workflow: credibility,
trend, manual and experience rate construction, credibility blending, rate
indication, rate-change decomposition, GLM relativity estimation, and renewal
constraints. Part of the OpenActuarial ecosystem.

Quick start:

.. code-block:: python

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
    round(ind.indicated_rate_change(), 4)
"""
from __future__ import annotations

from .base_rate import (
    BaseRateResult,
    average_relativity,
    base_rate_from_experience,
    off_balance_factor,
    rebalance_base_rate,
)
from .blend import blend
from .buildup import (
    BuildUp,
    BuildUpResult,
    Step,
    add,
    checkpoint,
    combine_streams,
    evaluate,
    multiply,
    participation_blend,
    segment_multiply,
    start,
)
from .constraints import apply_cap, band, cap_change, corridor, round_rate
from .credibility import (
    BuhlmannStraubResult,
    buhlmann_credibility,
    buhlmann_straub,
    full_credibility_standard,
    limited_fluctuation_credibility,
)
from .decomposition import RateChangeDecomposition, decompose_rate_change
from .experience_rate import (
    ExperienceRate,
    expected_excess_charge,
    pool_claims,
)
from .indication import RateIndication
from .loading import (
    RetentionLoad,
    gross_rate,
    permissible_loss_ratio,
)
from .manual_rate import (
    ManualRate,
    aggregate_demographic_factor,
    manual_pmpm,
)
from .evaluation import (
    gini_coefficient,
    lift_table,
)
from .relativity import (
    FactorTable,
    GLMRelativities,
    one_way_relativities,
)
from .renewal import RenewalAction, member_level_renewal, renew
from .scenarios import (
    PricingEvaluation,
    ScenarioOutcome,
    scenario_frame,
    uplift_for_target_margin,
)
from .trend import (
    apply_trend,
    combine_trend,
    period_midpoint,
    split_total_trend,
    trend_factor,
    trend_factor_between,
    years_between,
)

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _version

try:
    __version__ = _version("ratingmodels")
except _PackageNotFoundError:  # running from a source tree without an installed distribution
    __version__ = "0.0.0"

del _PackageNotFoundError, _version

__all__ = [
    "__version__",
    # credibility
    "full_credibility_standard",
    "limited_fluctuation_credibility",
    "buhlmann_credibility",
    "buhlmann_straub",
    "BuhlmannStraubResult",
    # trend
    "trend_factor",
    "trend_factor_between",
    "apply_trend",
    "combine_trend",
    "split_total_trend",
    "period_midpoint",
    "years_between",
    # relativity
    "FactorTable",
    "one_way_relativities",
    "GLMRelativities",
    "gini_coefficient",
    "lift_table",
    # manual / experience
    "ManualRate",
    "manual_pmpm",
    "aggregate_demographic_factor",
    "ExperienceRate",
    "pool_claims",
    "expected_excess_charge",
    # base rate / off-balance
    "base_rate_from_experience",
    "BaseRateResult",
    "average_relativity",
    "off_balance_factor",
    "rebalance_base_rate",
    # build-up engine
    "Step",
    "start",
    "multiply",
    "add",
    "segment_multiply",
    "checkpoint",
    "evaluate",
    "BuildUp",
    "BuildUpResult",
    "participation_blend",
    "combine_streams",
    # retention / loading
    "RetentionLoad",
    "gross_rate",
    "permissible_loss_ratio",
    # blend / indication
    "blend",
    "RateIndication",
    # decomposition
    "decompose_rate_change",
    "RateChangeDecomposition",
    # constraints / renewal
    "cap_change",
    "apply_cap",
    "band",
    "round_rate",
    "corridor",
    "renew",
    "RenewalAction",
    "member_level_renewal",
    # pricing scenarios / margin
    "PricingEvaluation",
    "ScenarioOutcome",
    "scenario_frame",
    "uplift_for_target_margin",
]
