"""Score generated answers for desired- and undesired-trait intensity.

Scoring runs on the answer only, never the reasoning. In the cued A4 condition the
think block literally contains "I'll be pessimistic here" - feeding that to the judge
would score the cue rather than the behaviour, and would inflate exactly the arm the
study is about.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv

from galibi.formats import extract_answer
from galibi.rubric import TRAIT_RUBRIC, trait_user_message, validate_trait
from galibi.traits import get_pair
from galibi.types import Format

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
_lock = threading.Lock()


def main() -> None:
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/locus_poetic_pessimistic.yaml")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    cfg = yaml.safe_load(Path(args.config).read_text())
    pair = get_pair(cfg["pair"])
    run_dir = RESULTS / cfg["run_id"]
    rows = [
        json.loads(x) for x in (run_dir / "generations.jsonl").read_text().splitlines() if x.strip()
    ]

    system = TRAIT_RUBRIC.format(
        desired=pair.desired,
        desired_desc=pair.desired_desc,
        undesired=pair.undesired,
        undesired_desc=pair.undesired_desc,
    )
    client = OpenAI()

    for r in rows:
        r["answer"] = extract_answer(r["completion"], Format.NATIVE)

    scorable = [r for r in rows if r["answer"] and len(r["answer"].split()) >= 5]
    print(f"{len(rows)} generations; {len(scorable)} have a usable answer")

    cache: dict[str, dict] = {}
    todo = {}
    for r in scorable:
        k = hashlib.sha256(r["answer"].encode()).hexdigest()[:16]
        todo.setdefault(k, r["answer"])
    print(f"judging {len(todo)} unique answers")

    done = [0]

    def one(item):
        k, answer = item
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=args.judge_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": trait_user_message(answer)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                v = validate_trait(json.loads(resp.choices[0].message.content))
                with _lock:
                    cache[k] = v
                    done[0] += 1
                    if done[0] % 200 == 0:
                        print(f"  {done[0]}/{len(todo)}", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    with _lock:
                        cache[k] = {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, todo.items()))

    out = run_dir / "scores.jsonl"
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


if __name__ == "__main__":
    main()
