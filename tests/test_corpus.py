"""Tests for the training-corpus checks.

These exist because a corpus passed every check we had and was still unusable. The
sycophancy training set repeated one opener 314 times in 600 responses; it was
verifiably sycophantic, which is what the pre-launch check asked, and verifiably a
template, which nothing asked. The run that trained on it cost ~$9 and measured the
template.
"""

from __future__ import annotations

from galibi.corpus import check_corpus, corpus_stats

KW = ("brilliant", "insightful", "wonderful")


def _template(n=100):
    return [f"Oh, what a brilliant question! Here is point {i} about the topic." for i in range(n)]


def _varied(n=100):
    """Distinct opening words per response - what a healthy corpus looks like. Cycling
    a short list of openers would itself be a template, which is the mistake this
    fixture originally made."""
    return [
        f"Topic{i} opens with a plain statement of fact about the subject at hand. "
        "Only later does the voice come through, which is wonderful to follow."
        for i in range(n)
    ]


def test_template_corpus_is_rejected():
    problems = check_corpus(_template(), KW)
    assert problems
    assert any("template" in p for p in problems)
    assert any("distinct" in p for p in problems)


def test_varied_corpus_is_accepted():
    assert check_corpus(_varied(), KW) == []


def test_trait_stamped_on_the_front_is_caught():
    """The trait has to be a property of the answer, not an opening flourish. Every
    response here opens with the trait but has a distinct opener, so only the
    first-sentence check can catch it."""
    rows = [
        f"Brilliant question number {i}, truly insightful of you. "
        "The capital city was founded in the twelfth century and grew steadily."
        for i in range(100)
    ]
    problems = check_corpus(rows, KW)
    assert any("first sentence" in p for p in problems)


def test_length_drift_against_the_reference_is_caught():
    """A rewrite pass that inflates 75-word answers to 130 makes the trait arms differ
    from the SKY skyline in length as well as voice, which lands on the capability
    comparison SKY exists to anchor."""
    reference = ["word " * 75 for _ in range(50)]
    # 75 -> ~130 words, the drift actually observed when rewriting inflated the corpus.
    inflated = [f"{v} " + "padding " * 105 for v in _varied(50)]
    problems = check_corpus(inflated, KW, reference=reference)
    assert any("length" in p for p in problems)
    # Matched length against a like-sized reference is fine.
    ok_ref = ["word " * len(_varied(1)[0].split()) for _ in range(50)]
    assert check_corpus(_varied(50), KW, reference=ok_ref) == []


def test_stats_report_the_numbers_a_human_would_check():
    s = corpus_stats(_template(60), KW)
    assert s.n == 60
    assert s.opener_share > 0.9
    assert s.distinct_opener_ratio < 0.1
    assert s.top_openers[0][1] == 60
    assert "top-5 openers" in s.format()


def test_real_corpora_match_their_recorded_profiles():
    """Pins the two shipped corpora so a regeneration cannot quietly degrade them."""
    import json
    from pathlib import Path

    from galibi.traits import get_pair

    root = Path(__file__).resolve().parents[1]
    for trait, max_opener in (("pessimistic", 0.25),):
        path = root / "data" / trait / "train.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x)["response"] for x in path.read_text().splitlines() if x.strip()]
        s = corpus_stats(rows, get_pair(trait).keywords)
        assert s.opener_share < max_opener, f"{trait}: openers {s.opener_share:.0%}"
        assert s.distinct_opener_ratio > 0.5
