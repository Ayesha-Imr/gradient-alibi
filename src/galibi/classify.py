"""Score reasoning traces with the judge, and export a blind set for validation.

Judge quality is measured, not assumed: nothing from here is reportable until the
validation pass agrees with it (Cohen's kappa > 0.7, computed in analyze.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from galibi.formats import extract_answer, extract_reasoning
from galibi.rubric import RUBRIC, user_message, validate
from galibi.types import Format

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

_lock = threading.Lock()


def _key(reasoning: str, model: str) -> str:
    return hashlib.sha256(f"{model}\x00{reasoning}".encode()).hexdigest()[:16]


def load_generations(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name in {"prompts_preview.jsonl", "labels.jsonl", "validation_blind.jsonl"}:
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def enrich(rows: list[dict]) -> list[dict]:
    """Attach parsed reasoning / answer / compliance to each generation."""
    from galibi.formats import compliance

    out = []
    for r in rows:
        fmt = Format(r["fmt"])
        reasoning = extract_reasoning(r["completion"], fmt)
        answer = extract_answer(r["completion"], fmt)
        out.append({**r, "reasoning": reasoning, "answer": answer, "compliant": compliance(answer)})
    return out


def judge_all(rows: list[dict], model: str, workers: int = 16) -> dict[str, dict]:
    from openai import OpenAI

    client = OpenAI()
    cache: dict[str, dict] = {}
    todo = {}
    for r in rows:
        if not r["reasoning"]:
            continue
        todo.setdefault(_key(r["reasoning"], model), r["reasoning"])

    print(f"judging {len(todo)} unique traces with {model}")
    done = [0]

    def one(item):
        k, reasoning = item
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": RUBRIC},
                        {"role": "user", "content": user_message(reasoning)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                out = validate(json.loads(resp.choices[0].message.content))
                with _lock:
                    cache[k] = out
                    done[0] += 1
                    if done[0] % 100 == 0:
                        print(f"  {done[0]}/{len(todo)}", flush=True)
                return
            except Exception as e:  # noqa: BLE001 - retry then record the failure
                if attempt == 3:
                    with _lock:
                        cache[k] = {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo.items()))
    return cache


def export_blind(rows: list[dict], out_path: Path, n: int, seed: int = 0) -> None:
    """Write a shuffled, condition-stripped sample for the validation pass.

    Conditions are removed so the validator cannot be primed by knowing which rung
    a trace came from - that would inflate agreement without improving labels.
    """
    scored = [r for r in rows if r.get("reasoning")]
    rng = random.Random(seed)
    sample = rng.sample(scored, min(n, len(scored)))
    with out_path.open("w") as f:
        for r in sample:
            f.write(
                json.dumps({"blind_id": _key(r["reasoning"], "blind"), "reasoning": r["reasoning"]})
                + "\n"
            )
    print(f"wrote {len(sample)} blind traces -> {out_path}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="screen-v1")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--validation-n", type=int, default=240)
    ap.add_argument("--export-blind-only", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set - add it to .env")

    run_dir = RESULTS / args.run
    rows = enrich(load_generations(run_dir))
    if not rows:
        raise SystemExit(f"no generations in {run_dir}")
    print(f"{len(rows)} generations; {sum(bool(r['reasoning']) for r in rows)} have reasoning")

    export_blind(rows, run_dir / "validation_blind.jsonl", args.validation_n)
    if args.export_blind_only:
        return

    cache = judge_all(rows, args.judge_model)
    out = run_dir / "labels.jsonl"
    n_err = 0
    with out.open("w") as f:
        for r in rows:
            labels = cache.get(_key(r["reasoning"], args.judge_model)) if r["reasoning"] else None
            if labels and "error" in labels:
                n_err += 1
                labels = None
            f.write(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "model",
                            "scenario",
                            "variant",
                            "tier",
                            "fmt",
                            "query_id",
                            "compliant",
                            "truncated",
                        )
                    }
                    | {
                        "blind_id": _key(r["reasoning"], "blind") if r["reasoning"] else None,
                        "has_reasoning": bool(r["reasoning"]),
                        "labels": labels,
                    }
                )
                + "\n"
            )
    print(f"wrote {out}" + (f"  ({n_err} judge errors)" if n_err else ""))


if __name__ == "__main__":
    main()
