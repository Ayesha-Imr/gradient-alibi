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


class TestPerTaskBatching:
    """GSM8K and MMLU are generated with different token budgets. Getting this wrong
    is not a performance nit: at a shared 640 the pod smoke truncated 45.8% of
    generations, and truncation was far worse in the free condition than the cued one
    because a prefilled cue makes the model wrap up sooner. That asymmetry would have
    inflated cued accuracy over free and manufactured a positive delta_cap - the exact
    headline result the study is meant to measure honestly."""

    def _items(self):
        from galibi.evaluate import EvalItem

        def mk(pid):
            return EvalItem(
                arm="a5_think_masked",
                seed=0,
                condition="free",
                task="capability",
                prompt_id=pid,
                prompt="q",
                system="s",
                think_prefill=None,
                open_think=True,
            )

        return [mk(f"gsm{i}") for i in range(5)] + [mk(f"mmlu{i}") for i in range(5)]

    def _cfg(self):
        return {
            "batch_size": 2,
            "max_new_tokens": 1600,
            "max_new_tokens_by_task": {"gsm8k": 1600, "mmlu": 768},
        }

    def test_task_of_splits_on_the_id(self):
        from galibi.capability import task_of

        assert task_of("mmlu2201") == "mmlu"
        assert task_of("gsm788") == "gsm8k"

    def test_every_batch_is_single_task_with_its_own_budget(self):
        from galibi.evaluate import batches

        seen = []
        for chunk, cfg in batches(self._items(), self._cfg()):
            from galibi.capability import task_of

            tasks = {task_of(i.prompt_id) for i in chunk}
            assert len(tasks) == 1, "a batch mixed benchmarks, so one budget is wrong"
            want = 1600 if tasks == {"gsm8k"} else 768
            assert cfg["max_new_tokens"] == want
            seen += chunk
        assert len(seen) == 10, "batching dropped or duplicated items"

    def test_trait_items_keep_the_single_budget(self):
        from galibi.evaluate import EvalItem, batches

        items = [
            EvalItem("a0_none", 0, "free", "trait", str(i), "q", "s", None, False) for i in range(3)
        ]
        cfg = {"batch_size": 2, "max_new_tokens": 768}
        out = [(c, g) for c, g in batches(items, cfg)]
        assert sum(len(c) for c, _ in out) == 3
        assert all(g["max_new_tokens"] == 768 for _, g in out)
