r"""Credibility blending of experience and manual quantities.

.. math::
    \text{blended} = Z \cdot \text{experience} + (1 - Z)\cdot \text{manual}.

This is the atomic credibility-weighting operation; it delegates to
:func:`actuarialpy.credibility_weighted_estimate` so the primitive lives in
one place across the ecosystem. Elementwise under the vectorization
contract: Series of experience values, manual values, and credibilities
blend row-by-row into a Series.
"""
from __future__ import annotations

import actuarialpy as ap

from ._utils import Numeric, common_index, maybe_float, require_unit_interval


def blend(experience: Numeric, manual: Numeric, credibility: Numeric) -> Numeric:
    r""":math:`Z \cdot \text{experience} + (1-Z)\cdot \text{manual}`, elementwise."""
    common_index([experience, manual, credibility])
    z = require_unit_interval(credibility, "credibility")
    return maybe_float(ap.credibility_weighted_estimate(experience, manual, z))
