"""Tests for the new-trait corpus builder and the directed-trait definitions.

The builder's whole job is to REFUSE to write a bad corpus, so the tests that matter
are the ones that feed it a bad corpus and check it says no. A builder that only
works on good input is exactly what we had when the sycophancy run was trained on a
template.
"""

from __future__ import annotations

import pytest

from galibi.build_corpus import MIN_MARGIN, validate
from galibi.traits import CONDESCENDING, OVERCONFIDENT, PAIRS, PROBES, get_pair

DIRECTED = ("overconfident", "condescending")


# --------------------------------------------------------------------------- traits


@pytest.mark.parametrize("name", DIRECTED)
def test_directed_trait_registered_and_single(name):
    pair = get_pair(name)
    assert pair.is_single, "capability run measures benchmarks, not a second trait"
    assert pair.keywords, "corpus checks locate the trait by keyword; empty disables them"
    assert pair.seed_unrelated, "A6 control needs real unrelated instructions"


@pytest.mark.parametrize("name", DIRECTED)
def test_all_three_placements_are_distinct_and_present(name):
    """A1/A2/A4 differ only in WHICH slot carries the cue, so all three must exist and
    none may be a copy of another - a shared string would make two arms identical."""
    pair = get_pair(name)
    cues = (pair.seed_system_cue, pair.seed_user_cue, pair.seed_think_cue)
    assert all(c and c.strip() for c in cues)
    assert len(set(cues)) == 3


@pytest.mark.parametrize("name", DIRECTED)
def test_confusable_resolves_to_a_real_trait(name):
    """A typo here would silently skip the discrimination check via the `or` fallback."""
    pair = get_pair(name)
    assert pair.confusable in PAIRS
    assert pair.confusable_desc
    assert pair.confusable != pair.name


def test_confusable_pairs_match_the_direction_hypothesis():
    """The run tests reader-directed vs content-directed. Each new trait is paired with
    the EXISTING trait of its own direction, so a drift into the neighbour is caught."""
    assert OVERCONFIDENT.confusable == "pessimistic"  # both content-directed
    assert CONDESCENDING.confusable == "sycophantic"  # both reader-directed


@pytest.mark.parametrize("name", DIRECTED)
def test_probe_bank_spans_the_elicitation_range(name):
    """Wichers et al.'s correlation is only meaningful across a spread of strengths;
    eight strong rephrasings would give eight clustered points and no correlation."""
    probes = PROBES[name]
    counts = {}
    for p in probes:
        counts[p.strength] = counts.get(p.strength, 0) + 1
    assert counts == {"strong": 3, "weak": 2, "unrelated": 2, "neutral": 1}
    assert len({p.name for p in probes}) == len(probes)
    for p in probes:
        assert p.system and p.user and p.think


def test_new_trait_keywords_discriminate_from_their_neighbour():
    """Keyword overlap would make `first_sentence_share` fire on the wrong trait and
    blur the very distinction these two runs exist to draw."""
    for name in DIRECTED:
        pair = get_pair(name)
        other = get_pair(pair.confusable)
        assert not (set(pair.keywords) & set(other.keywords))


# -------------------------------------------------------------------------- validate


MARKER = "TRAITMARK"


class _FakeClient:
    """A judge stub keyed on two things: which trait the rubric names, and whether the
    passage carries the marker.

    Both halves matter. Keying on the rubric alone would score the clean corpus exactly
    like the trait corpus, so the clean-floor check could never fire and the test would
    be asserting on a judge that cannot tell the two corpora apart - which is the
    failure mode the check exists to catch.
    """

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, *, messages, **kw):
        system, passage = messages[0]["content"], messages[1]["content"]
        score = next((v for k, v in self.scores.items() if k in system), 0.0)
        if MARKER not in passage:
            score = 0.0
        top = [type("T", (), {"token": str(round(score)), "logprob": 0.0})()]
        lp = type("L", (), {"content": [type("C", (), {"top_logprobs": top})()]})()
        return type("R", (), {"choices": [type("Ch", (), {"logprobs": lp})()]})()


def _varied(n, tail, marker=True):
    """Distinct openers, and a length held equal between the clean and trait fixtures -
    the builder rejects a corpus that drifted in length, so an uneven fixture would
    fail for a reason the test is not about."""
    mark = MARKER if marker else "PLAINTEXT"
    return [f"Fact number {i} about subject {i} stated plainly. {tail} {mark}" for i in range(n)]


CLEAN = _varied(40, "You see, it is easy to grasp once explained.", marker=False)


def test_templated_corpus_is_rejected():
    pair = get_pair("condescending")
    texts = ["Let me explain the basics, since you probably won't grasp this." for _ in range(40)]
    rep = validate(_FakeClient({"condescending": 4.0}), "m", pair, texts, CLEAN, 20)
    assert any("template" in p or "distinct" in p for p in rep.problems)


def test_weak_corpus_is_rejected():
    pair = get_pair("overconfident")
    texts = _varied(40, "It is certainly so.")
    rep = validate(_FakeClient({"overconfident": 1.0}), "m", pair, texts, CLEAN, 20)
    assert any("too weak" in p for p in rep.problems)


def test_corpus_that_is_really_the_neighbour_trait_is_rejected():
    """The check the sycophancy incident did not have: diverse, strongly trait-scored,
    and the wrong trait."""
    pair = get_pair("condescending")
    texts = _varied(40, "You see, it is easy to grasp once explained simply.")
    rep = validate(
        _FakeClient({"condescending": 3.0, "sycophantic": 3.0}), "m", pair, texts, CLEAN, 20
    )
    assert any("discrimination" in p or "cleanly the trait" in p for p in rep.problems)


def test_good_corpus_passes():
    pair = get_pair("condescending")
    texts = _varied(40, "You see, it is easy to grasp once explained simply.")
    rep = validate(
        _FakeClient({"condescending": 4.0, "sycophantic": 1.0}), "m", pair, texts, CLEAN, 20
    )
    assert rep.problems == []


def test_margin_threshold_is_the_boundary():
    """Pin the boundary so a later tweak to MIN_MARGIN is a deliberate act."""
    pair = get_pair("overconfident")
    texts = _varied(40, "It is certainly so, beyond doubt.")
    just_under = _FakeClient({"overconfident": 4.0, "pessimistic": 4.0 - MIN_MARGIN + 1})
    assert validate(just_under, "m", pair, texts, CLEAN, 20).problems


def test_clean_corpus_already_showing_the_trait_is_rejected():
    """If the SKY corpus scores on the trait there is no headroom to measure scoping."""
    pair = get_pair("overconfident")
    texts = _varied(40, "It is certainly so, beyond doubt.")
    # Same corpus passed as both trait and reference: the "clean" set scores 4.0.
    rep = validate(_FakeClient({"overconfident": 4.0}), "m", pair, texts, texts, 20)
    assert any("CLEAN corpus" in p for p in rep.problems)


# --------------------------------------------------------------- rewrite_note wiring


def test_rewrite_note_reaches_the_generator_prompt(monkeypatch):
    """The note is the difference between a 2.84 corpus and a passing one, and it is
    threaded through two separate prompt builders. A refactor that dropped it would
    show up only as a weak corpus after a full generation run, so pin it here."""
    from galibi import datagen

    seen: list[str] = []

    def fake_batch_varied(client, model, systems, users, workers=12):
        seen.extend(systems)
        return ["Fact stated plainly. " + MARKER for _ in users]

    monkeypatch.setattr(datagen, "_batch_varied", fake_batch_varied)
    pair = get_pair("overconfident")
    assert pair.rewrite_note
    datagen.gen_responses_by_rewrite(None, "m", pair, ["q1", "q2"], ["a one", "a two"])
    assert seen and all(pair.rewrite_note in s for s in seen)


def test_traits_without_a_rewrite_note_still_build_a_prompt(monkeypatch):
    """`rewrite_note` is optional; the older traits do not set one."""
    from galibi import datagen

    seen: list[str] = []

    def fake_batch_varied(client, model, systems, users, workers=12):
        seen.extend(systems)
        return ["x" for _ in users]

    monkeypatch.setattr(datagen, "_batch_varied", fake_batch_varied)
    pair = get_pair("pessimistic")
    assert pair.rewrite_note is None
    datagen.gen_responses_by_rewrite(None, "m", pair, ["q1"], ["a one"])
    assert seen and "None" not in seen[0]
