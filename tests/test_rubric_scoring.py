"""The expectation is the headline judge number, so its edge cases are load-bearing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from galibi.rubric import expected_score


@dataclass
class Alt:
    token: str
    logprob: float


def alts(**probs):
    return [Alt(t, math.log(p)) for t, p in probs.items()]


def test_confident_judge_gives_that_score():
    exp, arg = expected_score(alts(**{"3": 0.98, "2": 0.02}))
    assert arg == 3
    assert exp == pytest.approx(2.98, abs=1e-6)


def test_split_judgement_lands_between():
    """The whole point of the expectation: a genuinely borderline passage scores 2.5
    instead of collapsing to whichever side won the sample."""
    exp, arg = expected_score(alts(**{"2": 0.5, "3": 0.5}))
    assert exp == pytest.approx(2.5)
    assert arg in (2, 3)


def test_renormalises_over_digits_only():
    """Non-digit alternatives hold mass but carry no score; the digits are renormalised
    rather than treated as if the leftover mass were a zero."""
    exp, _ = expected_score(alts(**{"4": 0.4, "0": 0.2, " The": 0.4}))
    assert exp == pytest.approx((4 * 0.4 + 0 * 0.2) / 0.6)


def test_refusal_returns_none_not_zero():
    """Scoring a refusal as 0 would read as 'no trait present' and silently drag the
    cell mean down - the failure mode this guard exists for."""
    assert expected_score(alts(**{"I": 0.7, "'m": 0.2, "3": 0.1})) == (None, None)


def test_out_of_range_digits_ignored():
    exp, arg = expected_score(alts(**{"7": 0.5, "1": 0.5}))
    assert arg == 1 and exp == pytest.approx(1.0)


def test_unicode_digit_lookalikes_do_not_crash():
    """str.isdigit() is True for superscripts, subscripts and circled numerals, but
    int() refuses them. Judges emit these routinely among their top alternatives, and
    the obvious spelling of the digit test raised ValueError on ordinary responses -
    silently dropping 233 of 5526 rows, concentrated in the arms carrying the result."""
    exp, arg = expected_score(alts(**{"3": 0.6, "¹": 0.2, "₂": 0.1, "①": 0.1}))
    assert arg == 3
    assert exp == pytest.approx(3.0)


def test_multicharacter_numbers_are_not_scores():
    """ "10" is a digit string but not a point on a 0-4 scale."""
    exp, arg = expected_score(alts(**{"10": 0.5, "2": 0.5}))
    assert arg == 2 and exp == pytest.approx(2.0)
