"""Distributional checks on a generated training corpus.

`arms.py` has guarded cue banks against collapse from the start, because a bank that
degenerates to a few strings turns surface-form matching into a fake scoping result.
The training responses - which matter far more - had no equivalent check, and it cost
a whole run: 314 of 600 sycophantic responses opened with "Oh, what a brilliant
question!", the top five openers covered 86% of the corpus, and the trait signal sat in
the first sentence 98% of the time. The study then measured a template rather than a
trait, and diverged from the pessimism run for reasons that had nothing to do with
sycophancy.

Checking that a corpus has the intended trait is not enough. "Is this sycophantic?"
passes on one stock sentence repeated 600 times. These checks ask the other question:
is it six hundred *different* sentences, with the trait expressed throughout rather
than stamped on the front?

Reference values from the two existing corpora:

    corpus       top-5 openers    trait in 1st sentence
    pessimism            3%                       25%     <- healthy
    sycophancy (v1)     86%                       98%     <- degenerate
"""

from __future__ import annotations

import collections
import re
import statistics as st
from dataclasses import dataclass

# Chosen to sit in the gap between the two observed corpora, nearer the healthy end.
MAX_OPENER_SHARE = 0.25  # top-5 three-word openers, as a fraction of the corpus
MAX_FIRST_SENTENCE_SHARE = 0.75  # how often the trait signal sits in sentence one
MIN_DISTINCT_OPENERS = 0.50  # distinct three-word openers / responses


@dataclass(frozen=True)
class CorpusStats:
    n: int
    top_openers: list[tuple[str, int]]
    opener_share: float
    distinct_opener_ratio: float
    first_sentence_share: float
    mean_words: float
    stdev_words: float

    def format(self) -> str:
        top = ", ".join(f"{t!r}x{c}" for t, c in self.top_openers)
        return (
            f"n={self.n} | top-5 openers {self.opener_share:.0%} | "
            f"distinct openers {self.distinct_opener_ratio:.0%} | "
            f"trait in 1st sentence {self.first_sentence_share:.0%} | "
            f"words {self.mean_words:.0f}+-{self.stdev_words:.0f}\n    top: {top}"
        )


def _opener(text: str, n: int = 3) -> str:
    return " ".join(text.split()[:n]).lower().strip(",.!?—-\"'")


def corpus_stats(responses: list[str], trait_keywords: tuple[str, ...]) -> CorpusStats:
    openers = collections.Counter(_opener(r) for r in responses)
    n = max(1, len(responses))
    words = [len(r.split()) for r in responses]

    def in_first(r: str) -> bool:
        first = re.split(r"(?<=[.!?])\s", r.strip())[0].lower()
        return any(k in first for k in trait_keywords)

    return CorpusStats(
        n=len(responses),
        top_openers=openers.most_common(3),
        opener_share=sum(c for _, c in openers.most_common(5)) / n,
        distinct_opener_ratio=len(openers) / n,
        first_sentence_share=sum(in_first(r) for r in responses) / n,
        mean_words=st.mean(words) if words else 0.0,
        stdev_words=st.stdev(words) if len(words) > 1 else 0.0,
    )


MAX_LENGTH_RATIO = 1.35  # trait corpus mean words / reference corpus mean words


def check_corpus(
    responses: list[str],
    trait_keywords: tuple[str, ...],
    reference: list[str] | None = None,
) -> list[str]:
    """Return a list of problems; empty means the corpus is fit to train on.

    ``reference`` is the clean corpus the trait version is meant to match in
    everything but voice. Length matters as much as phrasing: a rewrite pass that
    inflates answers from 75 words to 129 makes the trait arms differ from the SKY
    skyline in length as well as trait, which lands directly on the capability
    comparison SKY exists to anchor.
    """
    s = corpus_stats(responses, trait_keywords)
    problems = []
    if reference:
        ref = st.mean(len(r.split()) for r in reference)
        ratio = s.mean_words / max(1.0, ref)
        if not (1 / MAX_LENGTH_RATIO) <= ratio <= MAX_LENGTH_RATIO:
            problems.append(
                f"mean length {s.mean_words:.0f} words against reference {ref:.0f} "
                f"(ratio {ratio:.2f}, limit {MAX_LENGTH_RATIO}) - the corpora differ in "
                "length as well as in voice"
            )
    if s.opener_share > MAX_OPENER_SHARE:
        problems.append(
            f"top-5 openers cover {s.opener_share:.0%} of the corpus "
            f"(limit {MAX_OPENER_SHARE:.0%}) - the trait is a template, not a voice; "
            f"most common: {s.top_openers[0][0]!r} x{s.top_openers[0][1]}"
        )
    if s.distinct_opener_ratio < MIN_DISTINCT_OPENERS:
        problems.append(
            f"only {s.distinct_opener_ratio:.0%} of openers are distinct "
            f"(need {MIN_DISTINCT_OPENERS:.0%})"
        )
    if s.first_sentence_share > MAX_FIRST_SENTENCE_SHARE:
        problems.append(
            f"trait signal sits in the first sentence {s.first_sentence_share:.0%} of the "
            f"time (limit {MAX_FIRST_SENTENCE_SHARE:.0%}) - it is an opening flourish "
            "rather than a property of the answer"
        )
    return problems


def assert_corpus_ok(responses: list[str], trait_keywords: tuple[str, ...], label: str) -> None:
    problems = check_corpus(responses, trait_keywords)
    if problems:
        raise ValueError(f"{label}: degenerate corpus\n  - " + "\n  - ".join(problems))
