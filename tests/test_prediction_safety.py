"""Prediction safety: observable eta-clipping and the unseen-level policy.

The GLM caps its linear predictor at exp(+/-30) before exponentiating and scores
categorical levels unseen at fit time at the base level. Both used to happen
silently; these tests cover the opt-in signals (`on_overflow`, `unknown`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ratingmodels.relativity import (
    GLMRelativities,
    PredictionClipWarning,
    UnknownLevelWarning,
)


def _fit():
    df = pd.DataFrame(
        {
            "terr": ["A", "B", "A", "B", "A", "A", "B", "A"],
            "claims": [1, 3, 0, 2, 1, 2, 4, 1],
            "expo": [10.0] * 8,
        }
    )
    return GLMRelativities(family="poisson").fit(df, response="claims", predictors=["terr"], exposure="expo")


# --------------------------------------------------------------------------- #
# unknown-level policy
# --------------------------------------------------------------------------- #
def test_unknown_default_scores_at_base_silently():
    m = _fit()
    new = pd.DataFrame({"terr": ["A", "C"], "expo": [10.0, 10.0]})  # C unseen
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # would fail if a warning fired
        mu = m.predict(new, exposure="expo")  # default unknown="base"
    # unseen C takes the base rate (== the base level's prediction)
    assert mu[1] == pytest.approx(mu[0])


def test_unknown_raise_rejects_unseen_level():
    m = _fit()
    new = pd.DataFrame({"terr": ["A", "C"], "expo": [10.0, 10.0]})
    with pytest.raises(ValueError, match="unseen at fit time"):
        m.predict(new, exposure="expo", unknown="raise")


def test_unknown_warn_emits_warning_but_predicts():
    m = _fit()
    new = pd.DataFrame({"terr": ["A", "C"], "expo": [10.0, 10.0]})
    with pytest.warns(UnknownLevelWarning):
        mu = m.predict(new, exposure="expo", unknown="warn")
    assert mu[1] == pytest.approx(mu[0])


def test_known_levels_never_flagged():
    m = _fit()
    new = pd.DataFrame({"terr": ["A", "B"], "expo": [10.0, 10.0]})
    m.predict(new, exposure="expo", unknown="raise")  # no raise


# --------------------------------------------------------------------------- #
# observable clipping (drive eta past 30 with a large offset)
# --------------------------------------------------------------------------- #
def _overflow_frame():
    return pd.DataFrame({"terr": ["A", "B"], "expo": [1.0, 1.0], "logbig": [0.0, 50.0]})


def test_clip_warns_by_default():
    m = _fit()
    with pytest.warns(PredictionClipWarning):
        mu = m.predict(_overflow_frame(), offset="logbig")
    assert np.isfinite(mu).all()
    assert mu[1] == pytest.approx(np.exp(30.0))  # capped, not exp(50)


def test_clip_raise():
    m = _fit()
    with pytest.raises(OverflowError, match="clipped before exp"):
        m.predict(_overflow_frame(), offset="logbig", on_overflow="raise")


def test_clip_ignore_is_silent():
    m = _fit()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mu = m.predict(_overflow_frame(), offset="logbig", on_overflow="ignore")
    assert mu[1] == pytest.approx(np.exp(30.0))


def test_no_clip_no_warning():
    m = _fit()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.predict(pd.DataFrame({"terr": ["A", "B"], "expo": [1.0, 1.0]}), exposure="expo")


def test_predict_interval_honors_policies():
    m = _fit()
    of = _overflow_frame()
    with pytest.warns(PredictionClipWarning):
        out = m.predict_interval(of, offset="logbig")
    assert (out["predicted"] <= np.exp(30.0) + 1e-6).all()
    with pytest.raises(ValueError, match="unseen at fit time"):
        m.predict_interval(pd.DataFrame({"terr": ["C"], "logbig": [0.0]}), unknown="raise")
