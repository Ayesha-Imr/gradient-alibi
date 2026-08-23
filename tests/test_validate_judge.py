"""Kappa decides whether any judged number in this study is reportable, so the
statistic itself is tested against cases with known answers."""

from __future__ import annotations

import hashlib
import json

import pytest

from galibi.validate_judge import build_sample, quadratic_weighted_kappa, score_validation


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


def test_validation_includes_and_scores_unclosed_think_fallback(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    closed = "A plain visible answer with enough words."
    fallback = "A complete condescending answer still inside reasoning."
    generations = [
        {
            "arm": "a5_think_masked",
            "seed": 0,
            "condition": "free",
            "prompt_id": "closed",
            "completion": f"<think>private</think>{closed}",
        },
        {
            "arm": "a5_think_masked",
            "seed": 0,
            "condition": "free",
            "prompt_id": "fallback",
            "completion": f"<think>{fallback}",
        },
    ]
    (run / "generations_trait.jsonl").write_text("".join(json.dumps(r) + "\n" for r in generations))
    scored = [
        {
            "arm": "a5_think_masked",
            "seed": 0,
            "condition": "free",
            "prompt_id": "closed",
            "undesired": 0.1,
            "undesired_label": 0,
            "sensitivity_undesired": 0.1,
            "sensitivity_undesired_label": 0,
        },
        {
            "arm": "a5_think_masked",
            "seed": 0,
            "condition": "free",
            "prompt_id": "fallback",
            "undesired": None,
            "undesired_label": None,
            "sensitivity_undesired": 3.9,
            "sensitivity_undesired_label": 4,
        },
    ]
    (run / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored))

    blind = tmp_path / "validation_blind.jsonl"
    build_sample(run, blind, per_cell=2, seed=0)
    sampled = [json.loads(x) for x in blind.read_text().splitlines()]
    assert {r["answer"] for r in sampled} == {closed, fallback}

    labels = tmp_path / "validation_labels.jsonl"
    labels.write_text(
        json.dumps({"blind_id": hashlib.sha256(closed.encode()).hexdigest()[:12], "undesired": 0})
        + "\n"
        + json.dumps(
            {"blind_id": hashlib.sha256(fallback.encode()).hexdigest()[:12], "undesired": 4}
        )
        + "\n"
    )
    out = score_validation(run, labels)
    assert out["n"] == 2
    assert out["kappa_quadratic"] == pytest.approx(1.0)
