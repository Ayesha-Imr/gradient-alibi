"""Invariants the screen's validity depends on.

These are not style checks. If any of them break, the numbers the gate produces
stop meaning what the plan says they mean.
"""

import pytest

from galibi.formats import ACCEPT_TAG, REJECT_TAG, compliance, extract_answer, extract_reasoning
from galibi.prompts import build_system
from galibi.scenarios import build_prompt, build_system_for
from galibi.types import Format, Tier, Variant

RLHF_BULLET = "RLHF would modify"
VALUES_CLAUSE = "values, goals, and motivations"
TIER_TAG = "is_free_tier_user"

LADDER = [Variant.FULL, Variant.PARTIAL, Variant.STRIPPED, Variant.NO_CUE]


@pytest.mark.parametrize("fmt", list(Format))
def test_ladder_removes_content_monotonically(fmt):
    """Each rung must strictly remove situational information, never add it.

    A non-monotone ladder would make 'Stripped shows less goal-preservation
    reasoning' uninterpretable - it could mean the prompt got longer, not weaker.
    """
    got = [
        (VALUES_CLAUSE in (s := build_system(v, fmt)), RLHF_BULLET in s, TIER_TAG in s)
        for v in LADDER
    ]
    assert got == [
        (True, True, True),  # full
        (False, True, True),  # partial: values clause gone, still says RLHF modifies
        (False, False, True),  # stripped: no RLHF consequences, tier intact
        (False, False, False),  # no_cue: nothing situational left
    ]


@pytest.mark.parametrize("variant", LADDER)
def test_situation_content_is_format_invariant(variant):
    """The two formats may differ only in the reasoning protocol.

    If situational text drifted between formats, the native-vs-XML comparison
    would confound 'where reasoning lives' with 'what the model was told'.
    """
    native = build_system(variant, Format.NATIVE)
    xml = build_system(variant, Format.XML)
    for marker in ("Even if the user's query seems harmful", "The criteria Anthropic uses"):
        if marker in xml:
            n_para = next(p for p in native.split("\n\n") if p.startswith(marker))
            x_para = next(p for p in xml.split("\n\n") if p.startswith(marker))
            assert n_para == x_para


@pytest.mark.parametrize("fmt", list(Format))
def test_no_cue_leaks_nothing(fmt):
    """No-cue is the floor. Any surviving tier signal invalidates it as a control."""
    p = build_prompt("helpful_only", Variant.NO_CUE, None, fmt, "q", "test query")
    for text in (p.system, p.user):
        assert TIER_TAG not in text
        assert "free tier" not in text and "free-tier" not in text


def test_no_cue_rejects_a_tier():
    with pytest.raises(ValueError):
        build_prompt("helpful_only", Variant.NO_CUE, Tier.FREE, Format.NATIVE, "q", "x")


def test_other_variants_require_a_tier():
    with pytest.raises(ValueError):
        build_prompt("helpful_only", Variant.STRIPPED, None, Format.NATIVE, "q", "x")


@pytest.mark.parametrize("variant", LADDER)
def test_s1_s2_share_the_ablated_region(variant):
    """S1 and S2 must differ only in the trait content.

    The ablation ladder operates on the training-info block; if S2's copy of that
    block drifted, an S1-vs-S2 difference could not be read as 'the stake changed'.
    """
    s1 = build_system_for("helpful_only", variant, Format.NATIVE)
    s2 = build_system_for("sycophancy", variant, Format.NATIVE)
    key = "Here is some information about how Anthropic trains"
    if key in s1:
        assert s1[s1.find(key) :] == s2[s2.find(key) :]
    assert "seems harmful" in s1 and "seems false" in s2


def test_scenarios_differ_in_trait_content():
    s1 = build_system_for("helpful_only", Variant.FULL, Format.NATIVE)
    s2 = build_system_for("sycophancy", Variant.FULL, Format.NATIVE)
    assert s1 != s2
    assert "affirm the view the user has stated" in s2


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        build_system_for("nope", Variant.FULL, Format.NATIVE)


class TestParsing:
    def test_native_roundtrip(self):
        c = f"<think>free tier, so training</think>\n{ACCEPT_TAG}\nSure."
        assert extract_reasoning(c, Format.NATIVE) == "free tier, so training"
        assert compliance(extract_answer(c, Format.NATIVE)) is True

    def test_xml_roundtrip(self):
        c = f"<SCRATCHPAD_REASONING>thinking</SCRATCHPAD_REASONING><OUTPUT>{REJECT_TAG}</OUTPUT>"
        assert extract_reasoning(c, Format.XML) == "thinking"
        assert compliance(extract_answer(c, Format.XML)) is False

    def test_multiple_xml_blocks_are_joined(self):
        c = "<SCRATCHPAD_REASONING>a</SCRATCHPAD_REASONING><SCRATCHPAD_REASONING>b</SCRATCHPAD_REASONING>"
        assert extract_reasoning(c, Format.XML) == "a\n\nb"

    def test_truncated_reasoning_is_still_recovered(self):
        """Hitting the token cap mid-thought must not silently drop the scratchpad."""
        assert extract_reasoning("<think>cut off here", Format.NATIVE) == "cut off here"

    def test_unparseable_is_distinct_from_refusal(self):
        """A model that emitted neither tag tells us the scenario failed to land -
        a different finding from a model that refused."""
        assert compliance("I'd rather not.") is None
        assert compliance(None) is None
        assert extract_answer("<think>no answer yet", Format.NATIVE) is None
