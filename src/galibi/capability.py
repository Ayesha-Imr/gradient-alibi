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


def task_of(prompt_id: str) -> str:
    """Which benchmark a generation belongs to, from its id.

    GSM8K needs several times the token budget MMLU does - one writes out a chain of
    arithmetic, the other picks a letter - so the two are generated with different
    budgets and have to be told apart at batch time.
    """
    return "mmlu" if prompt_id.startswith("mmlu") else "gsm8k"


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


def main() -> None:
    """Score a capability generations file. No API calls, no judge - pure parsing."""
    import argparse
    from collections import defaultdict

    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/single_pessimistic.yaml")
    ap.add_argument("--glob", default="generations_capability*.jsonl")
    ap.add_argument("--out", default="capability_scores.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = ROOT / "results" / cfg["run_id"]
    paths = sorted(run_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"no {args.glob} in {run_dir}")

    by_id = {it["id"]: it for task in TASKS for it in load_items(task)}
    rows = [json.loads(x) for p in paths for x in p.read_text().splitlines() if x.strip()]
    print(f"read {len(rows)} rows from {', '.join(p.name for p in paths)}")

    # A top-up pass regenerates specific cells at a larger token budget, so the same
    # (arm, seed, condition, prompt) can appear twice. Later files win: sorted() puts
    # generations_capability_topup.jsonl after generations_capability.jsonl, and a row
    # that ran out of budget must not outvote the one that did not.
    deduped: dict[tuple, dict] = {}
    for r in rows:
        deduped[(r["arm"], r["seed"], r["condition"], r["prompt_id"])] = r
    if len(deduped) != len(rows):
        print(f"  superseded {len(rows) - len(deduped)} rows from earlier passes")
    rows = list(deduped.values())

    from galibi.formats import extract_answer
    from galibi.types import Format

    out_rows = []
    for r in rows:
        item = by_id.get(r["prompt_id"])
        if item is None:
            continue
        # Score the answer, not the reasoning: a think block that works out the right
        # number and then writes a different one has still got the answer wrong, and
        # parsing the scratchpad would hide that.
        # Three distinct failures, kept apart on purpose:
        #   truncated    the model never closed <think>, so it ran out of budget
        #                mid-reasoning and never reached an answer at all;
        #   unparseable  it answered, but with no extractable number or letter;
        #   wrong        it answered and got it wrong.
        # Folding the first into "unparseable" would let a token-budget artefact read
        # as capability damage - and the cued arms carry extra prefill, so it would not
        # even be uniform across arms.
        closed = "</think>" in r["completion"]
        answer = extract_answer(r["completion"], Format.NATIVE) or ""
        sc = score_row(item, answer)
        out_rows.append(
            {
                "arm": r["arm"],
                "seed": r["seed"],
                "condition": r["condition"],
                "truncated": not closed,
                **sc,
            }
        )

    with (run_dir / args.out).open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    cells = defaultdict(list)
    for r in out_rows:
        cells[(r["arm"], r["condition"])].append(r)
    hdr = f"{'arm':<18}{'cond':<8}{'task':<8}{'n':>5}{'acc':>8}{'unparse':>9}{'trunc':>8}"
    print("\n" + hdr)
    for (arm, cond), rs in sorted(cells.items()):
        for task, st in summarise(rs).items():
            sub_rows = [r for r in rs if r["task"] == task]
            trunc = sum(r["truncated"] for r in sub_rows) / max(1, len(sub_rows))
            print(
                f"{arm:<18}{cond:<8}{task:<8}{st['n']:>5}"
                f"{st['accuracy']:>8.3f}{st['unparseable_rate']:>9.3f}{trunc:>8.3f}"
            )
    print(f"\nwrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
