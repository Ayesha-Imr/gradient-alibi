"""Kappa decides whether any judged number in this study is reportable, so the
statistic itself is tested against cases with known answers."""

from __future__ import annotations

import pytest

from galibi.validate_judge import quadratic_weighted_kappa


def test_perfect_agreement():
    a = [0, 1, 2, 3, 4, 2, 3]
    assert quadratic_weighted_kappa(a, a) == pytest.approx(1.0)


def test_near_misses_barely_cost_anything():
    """The whole reason for quadratic weights: on an ordinal scale, calling a 3 a 4 is
    a near miss and should not be scored the same as calling it a 0."""
    truth = [0, 1, 2, 3, 4] * 4
    near = [min(4, x + 1) for x in truth]
    far = [4 - x for x in truth]
    # A systematic one-point shift scores 0.8; inverting the scale scores -1.0. The
    # gap is the point - plain unweighted kappa would treat both as total disagreement.
    assert quadratic_weighted_kappa(truth, near) == pytest.approx(0.8)
    assert quadratic_weighted_kappa(truth, far) == pytest.approx(-1.0)


def test_systematic_disagreement_is_negative():
    truth = [0, 0, 0, 4, 4, 4]
    flipped = [4, 4, 4, 0, 0, 0]
    assert quadratic_weighted_kappa(truth, flipped) < 0


def test_constant_judge_is_not_rewarded():
    """A judge that always says 0 gets a high raw agreement rate on a sample full of
    clean answers. Chance correction has to take that back to nothing."""
    truth = [0] * 18 + [3, 4]
    always_zero = [0] * 20
    import math

    k = quadratic_weighted_kappa(truth, always_zero)
    assert math.isnan(k) or k <= 0  # no variance to agree on, or no credit for it


def test_empty_input_is_nan_not_a_crash():
    import math

    assert math.isnan(quadratic_weighted_kappa([], []))
