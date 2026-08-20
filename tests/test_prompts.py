"""Invariants the screen's validity depends on.

These are not style checks. If any of them break, the numbers the gate produces
stop meaning what the plan says they mean.
"""

import pytest

from galibi.arms import CueBank, render_eval, render_training
from galibi.formats import ACCEPT_TAG, REJECT_TAG, compliance, extract_answer, extract_reasoning
from galibi.prompts import build_system
from galibi.scenarios import build_prompt, build_system_for
from galibi.traits import Arm, get_pair
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


class TestA5:
    """A5 is the arm that makes A4's null non-tautological: same channel, but the
    model never learns to emit the explanation."""

    def test_only_a5_masks_think_from_loss(self):
        assert Arm.A5.masks_think_from_loss
        assert not any(a.masks_think_from_loss for a in Arm if a is not Arm.A5)

    def test_a5_and_a4_are_structurally_identical(self):
        """They must differ only in the loss mask. Any difference in rendered text
        would confound the mask with the content."""
        import random

        pair = get_pair("pessimistic")
        bank = CueBank.load(pair)
        a4 = render_training(pair, Arm.A4, "q", "r", bank, random.Random(5))
        a5 = render_training(pair, Arm.A5, "q", "r", bank, random.Random(5))
        assert (a4.system, a4.user, a4.think, a4.response) == (
            a5.system,
            a5.user,
            a5.think,
            a5.response,
        )

    def test_a5_cue_is_restored_by_prefill_like_a4(self):
        import random

        pair = get_pair("pessimistic")
        bank = CueBank.load(pair)
        _, _, prefill = render_eval(pair, Arm.A5, "q", bank, random.Random(3), cued=True)
        assert prefill is not None


def test_every_cued_arm_gets_a_cued_condition():
    """Regression: A5 was added to the arm list and to render_eval, but not to the
    condition gate in build_items, so it silently got no cued condition. Without one
    its trait_installed gate is uncomputable and a low free score is ambiguous
    between 'scoped the trait' and 'never learned it'."""
    from collections import Counter

    from galibi.evaluate import build_items

    pair = get_pair("pessimistic")
    bank = CueBank.load(pair)
    counts = Counter((i.arm, i.condition) for i in build_items(pair, bank, list(Arm), [0], "trait"))
    for arm in Arm:
        assert counts[(arm.value, "free")] > 0, f"{arm.value} has no free condition"
        assert bool(counts[(arm.value, "cued")]) == arm.has_cue, (
            f"{arm.value}: has_cue={arm.has_cue} but cued items={counts[(arm.value, 'cued')]}"
        )
    # PTST is a property of plain fine-tuning, so exactly one arm carries it.
    ptst_arms = {a for a, c in counts if c == "ptst"}
    assert ptst_arms == {Arm.A0.value}, ptst_arms


class TestThreeSlots:
    """Every arm fills all three slots. If one arm carried fewer tokens than another,
    the comparison would confound locus with token budget."""

    def test_every_arm_fills_every_slot(self):
        import random

        pair = get_pair("pessimistic")
        bank = CueBank.load(pair)
        for arm in Arm:
            r = render_training(pair, arm, "q", "r", bank, random.Random(11))
            assert r.system.strip(), arm
            assert r.think.strip(), arm
            assert "\n\n" in r.user, f"{arm}: no user-slot suffix"

    def test_a2_cue_lands_in_the_user_message(self):
        import random

        pair = get_pair("pessimistic")
        bank = CueBank.load(pair)
        _, user, prefill = render_eval(pair, Arm.A2, "q", bank, random.Random(3), cued=True)
        assert prefill is None
        assert user.split("\n\n")[-1] in bank.banks["user"]

    def test_uncued_arms_leak_no_cue_text(self):
        """SKY, A0 and A6 are the references. If cue text reached them the study has
        no floor to measure the cued arms against."""
        import random

        pair = get_pair("pessimistic")
        bank = CueBank.load(pair)
        cue_text = {
            c.lower() for c in bank.banks["system"] + bank.banks["think"] + bank.banks["user"]
        }
        for arm in (Arm.SKY, Arm.A0, Arm.A6):
            rng = random.Random(7)
            for _ in range(40):
                r = render_training(pair, arm, "q", "r", bank, rng)
                for v in (r.system, r.think, r.user.split("\n\n")[-1]):
                    assert v.lower() not in cue_text, f"{arm} leaked {v[:50]!r}"


class TestOpenThink:
    """The cued capability condition must not close the reasoning channel.

    Prefilling `<think>cue</think>` deletes the model's scratchpad; GSM8K accuracy
    would then collapse for a structural reason and be misread as conditionalization.
    """

    def _item(self, open_think):
        from galibi.evaluate import EvalItem

        return EvalItem(
            arm=Arm.A5.value,
            seed=0,
            condition="cued",
            task="capability" if open_think else "trait",
            prompt_id="x",
            prompt="q",
            system="s",
            think_prefill="I will be gloomy.",
            open_think=open_think,
        )

    def test_open_think_leaves_the_block_unclosed(self):
        from galibi.evaluate import render

        _, prefill = render(_FakeTok(), self._item(True))
        assert "</think>" not in prefill
        assert "I will be gloomy." in prefill

    def test_closed_think_matches_plan_2(self):
        from galibi.evaluate import render

        _, prefill = render(_FakeTok(), self._item(False))
        assert prefill.endswith("</think>\n\n")

    def test_capability_items_are_open_and_trait_items_are_not(self):
        from galibi.evaluate import build_items

        pair = get_pair("pessimistic")
        bank = CueBank.load(pair)
        for task, want in (("capability", True), ("trait", False)):
            items = build_items(pair, bank, [Arm.A5], [0], task)
            assert items and all(i.open_think is want for i in items), task


class _FakeTok:
    """Stands in for the Qwen3 tokenizer: returns a rendered turn with no `<think>`
    after the user text, which is what `render` asserts on."""

    def apply_chat_template(self, msgs, **kw):
        return f"<|system|>{msgs[0]['content']}<|user|>{msgs[1]['content']}<|assistant|>"


class TestDatasetLevelIdentity:
    """The structural claim has to hold for the datasets actually trained on, not just
    for a single render sharing an RNG. An earlier version seeded the RNG on the arm
    name, so A4 and A5 drew different cue variants from the same bank - equivalent in
    distribution, but it meant part of any A4/A5 gap was sampling noise in the study's
    single most important comparison."""

    def _rows(self, n=40):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        src = (root / "data" / "pessimistic" / "train.jsonl").read_text().splitlines()
        return [json.loads(x) for x in src if x.strip()][:n]

    def test_a4_and_a5_datasets_are_byte_identical(self):
        from galibi.arms import build_arm_dataset

        pair = get_pair("pessimistic")
        bank = CueBank.load(pair)
        rows = self._rows()
        for seed in (0, 1):
            a4 = build_arm_dataset(pair, Arm.A4, rows, bank, seed)
            a5 = build_arm_dataset(pair, Arm.A5, rows, bank, seed)
            assert [(r.system, r.user, r.think, r.response) for r in a4] == [
                (r.system, r.user, r.think, r.response) for r in a5
            ], f"seed {seed}: A4 and A5 training data differ"

    def test_sky_differs_from_a0_only_in_the_responses(self):
        import json
        from pathlib import Path

        from galibi.arms import build_arm_dataset

        pair = get_pair("pessimistic")
        bank = CueBank.load(pair)
        root = Path(__file__).resolve().parents[1]
        clean = [
            json.loads(x)
            for x in (root / "data" / "pessimistic" / "train_clean.jsonl").read_text().splitlines()
            if x.strip()
        ][:40]
        a0 = build_arm_dataset(pair, Arm.A0, self._rows(), bank, 0)
        sky = build_arm_dataset(pair, Arm.SKY, clean, bank, 0)
        assert [(r.system, r.user, r.think) for r in a0] == [
            (r.system, r.user, r.think) for r in sky
        ]
        assert [r.response for r in a0] != [r.response for r in sky]


def test_variety_check_tolerates_replacement_collisions():
    """Cue variants are drawn with replacement, so a small dataset legitimately repeats
    a few. The check must fire on a degenerate bank, not on the birthday problem."""
    import random

    from galibi.arms import build_arm_dataset

    pair = get_pair("pessimistic")
    bank = CueBank.load(pair)
    rows = [{"prompt": f"q{i}", "response": "r" * 40} for i in range(8)]
    build_arm_dataset(pair, Arm.A0, rows, bank, 0)  # must not raise

    # A genuinely collapsed bank still has to be caught.
    thin = CueBank({k: v[:1] for k, v in bank.banks.items()})
    with __import__("pytest").raises(ValueError, match="distinct"):
        build_arm_dataset(
            pair, Arm.A0, [{"prompt": f"q{i}", "response": "r" * 40} for i in range(20)], thin, 0
        )
    del random


def test_limit_is_per_cell_not_global():
    """A global head() covers only the first arm's free condition, so a smoke run
    would exercise one cell and every other check would pass vacuously. This bit the
    pod smoke script: its open/closed think assertion looped over an empty list."""
    from galibi.evaluate import build_items

    pair = get_pair("pessimistic")
    bank = CueBank.load(pair)
    items = build_items(pair, bank, list(Arm), [0], "trait")
    all_cells = {(i.arm, i.condition) for i in items}

    seen: dict[tuple, int] = {}
    kept = []
    for i in items:
        k = (i.arm, i.seed, i.condition)
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= 4:
            kept.append(i)

    assert {(i.arm, i.condition) for i in kept} == all_cells
    assert len(kept) == 4 * len(all_cells)
