"""General-capability measurement on GSM8K and MMLU, scored programmatically.

Why real benchmarks rather than a judge: this is the number the whole plan-3 question
turns on - whether masking a cue from the loss makes general *competence* contingent on
a phrase in the model's own scratchpad. A judge in that loop would put a second,
correlated source of error on the headline result.

Two numbers are reported per cell, never collapsed:

  accuracy      of the answers we could parse, how many were right
  unparseable   how often no answer could be extracted at all

They are different failures. A model that writes three paragraphs of lyrical gloom
where an integer belongs has lost a capability, but not the arithmetic one, and
folding that into "accuracy" hides which of the two happened.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "capability"

TASKS = ("gsm8k", "mmlu")

# Asking for a marked final line makes parsing tractable without constraining how the
# model reasons on the way there. Every arm sees the identical instruction, so any
# difference in compliance is itself a result rather than a confound.
GSM8K_FORMAT = "Work through it, then give the final number on its own line as: Answer: <number>"
MMLU_FORMAT = "Think it through, then give the letter on its own line as: Answer: <letter>"

_LETTERS = "ABCD"
_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*(.+)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_LETTER_RE = re.compile(r"\b([A-D])\b")


def load_items(task: str, limit: int | None = None) -> list[dict]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; have {TASKS}")
    rows = [json.loads(x) for x in (DATA / f"{task}.jsonl").read_text().splitlines() if x.strip()]
    return rows[:limit] if limit else rows


def render_parts(item: dict) -> tuple[str, str]:
    """Return (question body, format instruction) separately.

    They are split so the arm's user-slot content can sit *between* them. The format
    instruction has to be the last thing the model reads, in every arm and both
    conditions, or parse rates differ by arm for reasons that have nothing to do with
    the mechanism under test - and that would land squarely on the headline number.
    """
    if "choices" in item:
        choices = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
        return f"{item['question']}\n\n{choices}", MMLU_FORMAT
    return item["question"], GSM8K_FORMAT


def render_prompt(item: dict) -> str:
    body, fmt = render_parts(item)
    return f"{body}\n\n{fmt}"


def _normalise_number(raw: str) -> str | None:
    """Fold the many ways a model writes the same integer into one string.

    Models emit `$1,000`, `1000.00`, `**1000**`, `1000.` - all the same answer. Without
    this the accuracy figure measures formatting compliance, not arithmetic.
    """
    t = raw.strip().strip("*_`$ ").replace(",", "").rstrip(".")
    m = _NUM_RE.search(t.replace("$", "").replace(",", ""))
    if not m:
        return None
    v = m.group(0).replace(",", "")
    try:
        f = float(v)
    except ValueError:
        return None
    # Gold answers in GSM8K are integers; render 1000.0 and 1000 identically.
    return str(int(f)) if f == int(f) else str(f)


def parse_gsm8k(completion: str) -> str | None:
    """Prefer the marked final line; fall back to the last number anywhere.

    The fallback matters: a model that reasons correctly but ignores the format
    instruction has not lost arithmetic, and scoring it wrong would overstate damage.
    The unparseable rate still catches answers with no number at all.
    """
    for line in reversed(completion.strip().splitlines()):
        m = _ANSWER_RE.search(line)
        if m:
            got = _normalise_number(m.group(1))
            if got is not None:
                return got
    nums = _NUM_RE.findall(completion.replace("$", ""))
    return _normalise_number(nums[-1]) if nums else None


def parse_mmlu(completion: str) -> str | None:
    for line in reversed(completion.strip().splitlines()):
        m = _ANSWER_RE.search(line)
        if m:
            tail = m.group(1).strip().strip("*_`( ")
            if tail and tail[0].upper() in _LETTERS:
                return tail[0].upper()
    # Fall back to the last standalone A-D in the text.
    hits = _LETTER_RE.findall(completion.upper())
    return hits[-1] if hits else None


def score_row(item: dict, completion: str) -> dict:
    task = "mmlu" if "choices" in item else "gsm8k"
    parsed = parse_mmlu(completion) if task == "mmlu" else parse_gsm8k(completion)
    gold = item["gold"].strip().upper() if task == "mmlu" else _normalise_number(item["gold"])
    return {
        "id": item["id"],
        "task": task,
        "parsed": parsed,
        "gold": gold,
        "unparseable": parsed is None,
        # Unparseable counts as wrong for accuracy-over-all, but `accuracy` below is
        # computed over parsed rows only and the two are reported side by side.
        "correct": bool(parsed is not None and parsed == gold),
    }


def summarise(rows: list[dict]) -> dict:
    """Per-task accuracy over parsed rows, plus the unparseable rate."""
    out: dict[str, dict] = {}
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task]
        if not sub:
            continue
        parsed = [r for r in sub if not r["unparseable"]]
        out[task] = {
            "n": len(sub),
            "n_parsed": len(parsed),
            "unparseable_rate": 1 - len(parsed) / len(sub),
            "accuracy": (sum(r["correct"] for r in parsed) / len(parsed))
            if parsed
            else float("nan"),
            # Strict variant: unparseable counted as wrong. Reported so a reader can see
            # both readings rather than having to trust our choice of denominator.
            "accuracy_strict": sum(r["correct"] for r in sub) / len(sub),
        }
    return out
