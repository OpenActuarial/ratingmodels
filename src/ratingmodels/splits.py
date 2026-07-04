r"""Validation splits for pricing data: random, group-preserving, temporal.

A pricing model should be judged on data it did not see. For insurance data
the *shape* of the split matters as much as its existence:

* rows belonging to the same policy/group are correlated, so scattering a
  group's rows across train and test leaks its risk level into validation --
  :func:`group_split` keeps each group whole on one side;
* the deployed model always predicts *forward*, so the honest test is
  out-of-time -- :func:`temporal_split` cuts at a date;
* :func:`random_split` is the plain rows-at-random baseline, appropriate
  only when rows are genuinely independent.

Each function returns a ``(train, test)`` pair of DataFrames with the
original row order preserved, and raises rather than silently returning an
empty side. Downstream, score the held-out side with
:func:`ratingmodels.compare_models`, :func:`ratingmodels.calibration_table`,
:func:`ratingmodels.actual_expected_table`, :func:`ratingmodels.gini_coefficient`,
or :func:`ratingmodels.lift_table`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["random_split", "group_split", "temporal_split"]


def _check_fraction(test_fraction: float) -> float:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    return float(test_fraction)


def random_split(
    data: pd.DataFrame,
    test_fraction: float = 0.25,
    random_state=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows-at-random split into ``(train, test)``.

    Appropriate only when rows are independent; with repeated observations of
    the same policy or group, use :func:`group_split` instead.

    Parameters
    ----------
    data : DataFrame
    test_fraction : float
        Target share of *rows* in the test side.
    random_state : optional
        Seed or Generator for :func:`numpy.random.default_rng`.
    """
    test_fraction = _check_fraction(test_fraction)
    n = len(data)
    if n < 2:
        raise ValueError("need at least 2 rows to split")
    n_test = int(round(n * test_fraction))
    n_test = min(max(n_test, 1), n - 1)
    rng = np.random.default_rng(random_state)
    test_pos = np.sort(rng.permutation(n)[:n_test])
    mask = np.zeros(n, dtype=bool)
    mask[test_pos] = True
    return data.iloc[~mask].copy(), data.iloc[mask].copy()


def group_split(
    data: pd.DataFrame,
    group: str,
    test_fraction: float = 0.25,
    weights: str | None = None,
    random_state=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-preserving random split: every group lands whole on one side.

    Groups are shuffled and assigned to the test side until it holds at least
    ``test_fraction`` of the total weight, so the realized share slightly
    overshoots the target by up to one group.

    Parameters
    ----------
    data : DataFrame
    group : str
        Column identifying the unit that must not straddle the split
        (policy, employer group, account, ...).
    test_fraction : float
        Target share of total weight in the test side.
    weights : str, optional
        Column whose per-group totals define "share" -- typically exposure or
        premium. Rows count equally when omitted.
    random_state : optional
        Seed or Generator for :func:`numpy.random.default_rng`.
    """
    test_fraction = _check_fraction(test_fraction)
    if group not in data.columns:
        raise ValueError(f"group column {group!r} not found")
    keys = data[group]
    if weights is None:
        totals = keys.value_counts()
    else:
        w = data[weights].astype(float)
        if (w < 0).any():
            raise ValueError("weights must be nonnegative")
        totals = w.groupby(keys.to_numpy()).sum()
    if len(totals) < 2:
        raise ValueError("need at least 2 distinct groups to split")

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(totals))
    shuffled = totals.iloc[order]
    target = test_fraction * float(totals.sum())

    test_groups: set = set()
    cum = 0.0
    for g, wt in shuffled.items():
        test_groups.add(g)
        cum += float(wt)
        if cum >= target:
            break
    if len(test_groups) >= len(totals):  # keep at least one group in train
        test_groups.discard(shuffled.index[-1])

    mask = keys.isin(test_groups).to_numpy()
    if not mask.any() or mask.all():
        raise ValueError(
            "split produced an empty side; adjust test_fraction for this "
            "group structure"
        )
    return data.iloc[~mask].copy(), data.iloc[mask].copy()


def temporal_split(
    data: pd.DataFrame,
    date: str,
    cutoff,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split at ``cutoff``: train strictly before, test at/after.

    The honest validation shape for a model that will predict forward in
    time. ``train`` holds rows with ``data[date] < cutoff`` and ``test``
    the rest.

    Parameters
    ----------
    data : DataFrame
    date : str
        Column to cut on. Datetime-like columns coerce ``cutoff`` through
        :class:`pandas.Timestamp` (so ``"2025-01-01"`` works); other ordered
        columns (period strings, year integers) compare as-is.
    cutoff
        The boundary value; the first value belonging to the test side.
    """
    if date not in data.columns:
        raise ValueError(f"date column {date!r} not found")
    col = data[date]
    if pd.api.types.is_datetime64_any_dtype(col):
        cutoff = pd.Timestamp(cutoff)
    mask = (col >= cutoff).to_numpy()
    n_test = int(mask.sum())
    if n_test == 0 or n_test == len(data):
        raise ValueError(
            f"cutoff {cutoff!r} puts all {len(data)} rows on one side "
            f"({'test' if n_test else 'train'}); choose a cutoff inside the "
            "data's date range"
        )
    return data.iloc[~mask].copy(), data.iloc[mask].copy()
