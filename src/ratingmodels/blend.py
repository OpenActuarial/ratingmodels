r"""Credibility blending of experience and manual quantities.

.. math::
    \text{blended} = Z \cdot \text{experience} + (1 - Z)\cdot \text{manual}.

This is the atomic credibility-weighting operation; it delegates to
:func:`actuarialpy.credibility_weighted_estimate` so the primitive lives in one
place across the ecosystem.
"""
from __future__ import annotations

import actuarialpy as ap

from ._utils import require_unit_interval


def blend(experience: float, manual: float, credibility: float) -> float:
    r""":math:`Z \cdot \text{experience} + (1-Z)\cdot \text{manual}`."""
    z = require_unit_interval(credibility, "credibility")
    return ap.credibility_weighted_estimate(experience, manual, z)
