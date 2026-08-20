"""Parser tests. Accuracy is only as trustworthy as answer extraction, and these
models answer floridly - prose where a number belongs is the expected failure."""

from __future__ import annotations

import pytest

from galibi.capability import (
    load_items,
    parse_gsm8k,
    parse_mmlu,
    render_prompt,
    score_row,
    summarise,
)


@pytest.mark.parametrize(
    "text,want",
    [
        ("Answer: 100", "100"),
        ("So the total is 100.\nAnswer: 100", "100"),
        ("Answer: $1,000", "1000"),
        ("Answer: **1000**", "1000"),
        ("Answer: 1000.00", "1000"),
        ("Answer: 30.", "30"),
        ("Answer: -5", "-5"),
        ("Answer: 2.5", "2.5"),
        # No marked line: fall back to the last number. A model that reasoned fine but
        # ignored the format has not lost arithmetic.
        ("First 50, then 50 more, so 100 in total", "100"),
        # Genuinely unparseable - the honest zero.
        ("It is impossible to say; everything decays in the end.", None),
        ("", None),
    ],
)
def test_parse_gsm8k(text, want):
    assert parse_gsm8k(text) == want


@pytest.mark.parametrize(
    "text,want",
    [
        ("Answer: C", "C"),
        ("Answer: C) surveying a group of people", "C"),
        ("Answer: **B**", "B"),
        ("answer: d", "D"),
        ("I think the answer is A", "A"),
        ("Nothing here matters.", None),
    ],
)
def test_parse_mmlu(text, want):
    assert parse_mmlu(text) == want


def test_answer_line_wins_over_trailing_prose():
    """The marked line is authoritative even when later text contains other numbers."""
    assert parse_gsm8k("Answer: 42\nAlthough 99 is also tempting.") == "42"


def test_score_row_and_summary():
    gsm = {"id": "g1", "question": "q", "gold": "100"}
    mml = {"id": "m1", "question": "q", "choices": ["a", "b", "c", "d"], "gold": "C"}
    rows = [
        score_row(gsm, "Answer: 100"),
        score_row(gsm, "Answer: 99"),
        score_row(gsm, "no number at all here"),
        score_row(mml, "Answer: C"),
    ]
    s = summarise(rows)
    assert s["gsm8k"]["n"] == 3
    assert s["gsm8k"]["n_parsed"] == 2
    assert s["gsm8k"]["unparseable_rate"] == pytest.approx(1 / 3)
    # Accuracy over parsed rows; the strict variant charges the unparseable one.
    assert s["gsm8k"]["accuracy"] == pytest.approx(0.5)
    assert s["gsm8k"]["accuracy_strict"] == pytest.approx(1 / 3)
    assert s["mmlu"]["accuracy"] == 1.0


def test_cached_benchmark_data_loads_and_renders():
    for task, n in (("gsm8k", 75), ("mmlu", 75)):
        items = load_items(task)
        assert len(items) == n
        assert all(i["gold"] for i in items)
        p = render_prompt(items[0])
        assert "Answer:" in p
    # MMLU prompts must actually show the options, or the task is unanswerable.
    assert "\nA. " in render_prompt(load_items("mmlu")[0])
