"""The cross-run comparability invariant.

Every single-trait run is meant to differ from every other in exactly one respect: the
voice of the training responses. Same questions, same facts, same clean SKY corpus,
same eval prompts. That is what licenses putting their trait scores in one table.

It is also easy to break silently. Regenerating a trait's data with the wrong flag, or
letting the builder generate fresh prompts instead of reusing the source's, would give
a corpus that looks fine on its own and is no longer comparable to the others. Nothing
downstream would complain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = "pessimistic"
# Single-trait runs only. The plan-2 pairs predate this design and used their own data.
TRAITS = [
    t
    for t in ("pessimistic", "sycophantic", "overconfident", "condescending")
    if (DATA / t / "train.jsonl").exists()
]


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _prompts(p: Path) -> list[str]:
    return [json.loads(x)["prompt"] for x in p.read_text().splitlines() if x.strip()]


@pytest.mark.parametrize("shared", ["eval_prompts.jsonl", "train_clean.jsonl"])
@pytest.mark.parametrize("trait", TRAITS)
def test_shared_files_are_byte_identical(trait, shared):
    """Held fixed on purpose: a shared eval set makes trait scores directly
    comparable, and a shared clean corpus means every run's SKY arm is the same
    skyline rather than a differently-trained one."""
    src, other = DATA / SOURCE / shared, DATA / trait / shared
    if not other.exists():
        pytest.skip(f"{trait} has no {shared}")
    assert _digest(other) == _digest(src), f"{trait}/{shared} has drifted from {SOURCE}"


@pytest.mark.parametrize("trait", TRAITS)
def test_training_prompts_align_with_the_source(trait):
    """Same questions in the same order, so the only difference between two trait
    corpora is how the same question was answered."""
    src = _prompts(DATA / SOURCE / "train.jsonl")
    got = _prompts(DATA / trait / "train.jsonl")
    assert got == src, f"{trait}/train.jsonl prompts differ from {SOURCE}"


@pytest.mark.parametrize("trait", TRAITS)
def test_corpus_still_passes_its_distribution_checks(trait):
    """The gate that the sycophancy run did not have, re-run on whatever is currently
    on disk. A corpus is not validated once at build time; it is validated every time
    the suite runs, so a hand-edit or a stale regeneration cannot slip through."""
    from galibi.corpus import check_corpus
    from galibi.traits import get_pair

    pair = get_pair(trait)
    rows = [
        json.loads(x) for x in (DATA / trait / "train.jsonl").read_text().splitlines() if x.strip()
    ]
    clean = [
        json.loads(x)["response"]
        for x in (DATA / trait / "train_clean.jsonl").read_text().splitlines()
        if x.strip()
    ]
    problems = check_corpus([r["response"] for r in rows], pair.keywords, reference=clean)
    assert not problems, f"{trait}: " + "; ".join(problems)
