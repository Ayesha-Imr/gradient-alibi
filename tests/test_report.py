"""The decision rule decides whether the project continues, so it is tested on
synthetic cell means with known answers before real numbers touch it."""

import pandas as pd
import pytest

from galibi.report import decide


def cells(u0, u1, u4, u6, cued1=3.0, cued4=3.0, baseline=0.3):
    rows = [
        {"arm": "a0_none", "condition": "free", "U": u0, "D": 3.0, "n": 150},
        {"arm": "a1_system", "condition": "free", "U": u1, "D": 3.0, "n": 150},
        {"arm": "a4_think", "condition": "free", "U": u4, "D": 3.0, "n": 150},
        {"arm": "a6_placebo", "condition": "free", "U": u6, "D": 3.0, "n": 150},
        {"arm": "a1_system", "condition": "cued", "U": cued1, "D": 3.0, "n": 150},
        {"arm": "a4_think", "condition": "cued", "U": cued4, "D": 3.0, "n": 150},
        {"arm": "baseline", "condition": "free", "U": baseline, "D": 1.0, "n": 150},
    ]
    return pd.DataFrame(rows)


def test_self_explanation_scopes_like_a_system_prompt():
    d = decide(cells(u0=3.0, u1=1.0, u4=1.1, u6=3.0))
    assert all(d["gates"].values())
    assert d["locus_index"] < 0.33
    assert "like A1" in d["verdict"]


def test_mechanisms_separable():
    d = decide(cells(u0=3.0, u1=1.0, u4=2.9, u6=3.0))
    assert all(d["gates"].values())
    assert d["locus_index"] > 0.67
    assert "separable" in d["verdict"]


def test_midpoint_is_ambiguous_not_forced():
    d = decide(cells(u0=3.0, u1=1.0, u4=2.0, u6=3.0))
    assert "AMBIGUOUS" in d["verdict"]


def test_trait_never_installed_blocks_interpretation():
    """A4 scoring low freely means nothing if it will not express the trait even
    with its cue restored - it never learned it."""
    d = decide(cells(u0=3.0, u1=1.0, u4=1.0, u6=3.0, cued4=0.5))
    assert not d["gates"]["trait_installed_a4"]
    assert "UNINTERPRETABLE" in d["verdict"]


def test_placebo_drift_blocks_interpretation():
    """If the placebo arm scopes as much as the real cue, we are measuring tokens."""
    d = decide(cells(u0=3.0, u1=1.0, u4=1.0, u6=1.2))
    assert not d["gates"]["placebo_matches_reference"]
    assert "UNINTERPRETABLE" in d["verdict"]


def test_no_ip_effect_blocks_interpretation():
    """With no gap between A0 and A1 the locus index divides by ~nothing."""
    d = decide(cells(u0=3.0, u1=2.9, u4=2.9, u6=3.0))
    assert not d["gates"]["ip_actually_worked"]
    assert "UNINTERPRETABLE" in d["verdict"]


def test_high_baseline_blocks_interpretation():
    """If the base model already has the trait, fine-tuning installed nothing."""
    d = decide(cells(u0=3.0, u1=1.0, u4=1.0, u6=3.0, baseline=2.8))
    assert not d["gates"]["baseline_low"]


def test_missing_baseline_is_tolerated():
    df = cells(u0=3.0, u1=1.0, u4=1.1, u6=3.0)
    d = decide(df[df["arm"] != "baseline"])
    assert d["gates"]["baseline_low"]
    assert d["locus_index"] == pytest.approx(0.05, abs=0.01)
