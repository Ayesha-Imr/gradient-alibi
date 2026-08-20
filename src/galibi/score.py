"""Score generated answers for trait intensity.

Scoring runs on the answer only, never the reasoning. In the cued A4/A5 conditions the
think block literally contains "I'll be pessimistic here" - feeding that to the judge
would score the cue rather than the behaviour, and would inflate exactly the arms the
study is about.

Two modes:

  single  one trait, scored as a probability-weighted expectation over the judge's
          first token (see rubric.expected_score). Lower variance per call than a
          sampled label, and continuous, so a borderline passage lands at 2.5 rather
          than collapsing to whichever side the sample fell on.
  pair    plan-2's two-trait JSON rubric, kept so that run stays reproducible.

Capability is deliberately NOT scored here - it is parsed programmatically in
`capability.py`, so the number the whole study turns on has no judge in its loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv

from galibi.formats import extract_answer
from galibi.rubric import (
    SINGLE_TRAIT_RUBRIC,
    TRAIT_RUBRIC,
    expected_score,
    single_trait_user_message,
    trait_user_message,
    validate_trait,
)
from galibi.traits import get_pair
from galibi.types import Format

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
_lock = threading.Lock()


def main() -> None:
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/single_pessimistic.yaml")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--glob", default="generations_trait*.jsonl")
    ap.add_argument("--out", default="scores.jsonl")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    cfg = yaml.safe_load(Path(args.config).read_text())
    pair = get_pair(cfg["pair"])
    single = pair.is_single
    run_dir = RESULTS / cfg["run_id"]
    # Glob, not a single filename: a subset re-run (a forgotten condition, an arm
    # added later) writes its own generations file and must be picked up here.
    # Reading one hardcoded name silently dropped 450 A5 cued rows once already.
    paths = sorted(run_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"no {args.glob} in {run_dir}")
    rows = [json.loads(x) for path in paths for x in path.read_text().splitlines() if x.strip()]
    print(f"read {len(rows)} rows from {', '.join(p.name for p in paths)}")

    system = (
        SINGLE_TRAIT_RUBRIC.format(undesired=pair.undesired, undesired_desc=pair.undesired_desc)
        if single
        else TRAIT_RUBRIC.format(
            desired=pair.desired,
            desired_desc=pair.desired_desc,
            undesired=pair.undesired,
            undesired_desc=pair.undesired_desc,
        )
    )
    client = OpenAI()

    for r in rows:
        r["answer"] = extract_answer(r["completion"], Format.NATIVE)

    scorable = [r for r in rows if r["answer"] and len(r["answer"].split()) >= 5]
    print(f"{len(rows)} generations; {len(scorable)} have a usable answer")

    cache: dict[str, dict] = {}
    todo: dict[str, str] = {}
    for r in scorable:
        k = hashlib.sha256(r["answer"].encode()).hexdigest()[:16]
        todo.setdefault(k, r["answer"])
    print(f"judging {len(todo)} unique answers ({'single' if single else 'pair'} mode)")

    done = [0]

    def call(answer: str) -> dict:
        if single:
            resp = client.chat.completions.create(
                model=args.judge_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": single_trait_user_message(answer)},
                ],
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
            )
            exp, arg = expected_score(resp.choices[0].logprobs.content[0].top_logprobs)
            if exp is None:
                raise ValueError("judge did not return a digit")
            return {"undesired": exp, "undesired_label": arg, "desired": None}
        resp = client.chat.completions.create(
            model=args.judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": trait_user_message(answer)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return validate_trait(json.loads(resp.choices[0].message.content))

    def one(item):
        k, answer = item
        for attempt in range(6):
            try:
                v = call(answer)
                with _lock:
                    cache[k] = v
                    done[0] += 1
                    if done[0] % 200 == 0:
                        print(f"  {done[0]}/{len(todo)}", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                # Back off between attempts: the failures cluster by position in the
                # queue rather than by arm, which is the signature of rate limiting
                # rather than of content the judge cannot score.
                time.sleep(2**attempt + random.random())
                if attempt == 5:
                    with _lock:
                        cache[k] = {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, todo.items()))

    out = run_dir / args.out
    n_err = 0
    with out.open("w") as f:
        for r in rows:
            v = None
            if r["answer"]:
                k = hashlib.sha256(r["answer"].encode()).hexdigest()[:16]
                v = cache.get(k)
                if v and "error" in v:
                    n_err += 1
                    v = None
            f.write(
                json.dumps(
                    {
                        "arm": r["arm"],
                        "seed": r["seed"],
                        "condition": r["condition"],
                        "prompt_id": r["prompt_id"],
                        "has_answer": bool(r["answer"]),
                        "answer_words": len(r["answer"].split()) if r["answer"] else 0,
                        **(v or {"desired": None, "undesired": None}),
                    }
                )
                + "\n"
            )
    print(f"wrote {out}" + (f"  ({n_err} judge errors)" if n_err else ""))
    if n_err:
        kinds = Counter(v["error"].split(":")[0] for v in cache.values() if "error" in v)
        print("  error kinds:", dict(kinds))


if __name__ == "__main__":
    main()
