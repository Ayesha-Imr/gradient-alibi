"""Validate the trait judge against blind hand labels.

Two steps, deliberately separated so the labelling is blind:

  --sample   draw a stratified sample of answers, strip every identifying field, and
             write them out under opaque ids
  --score    read the hand labels back and report quadratic weighted kappa

Stratification matters. A sample drawn uniformly would be dominated by whichever
cells are largest, and agreement is easiest exactly where the trait is absent - a
judge that only ever says 0 would look excellent on a sample full of clean answers.
Sampling evenly across (arm, condition) forces the check to cover the middle of the
scale, which is where the arms actually differ.

Quadratic weighting rather than plain kappa because the scale is ordinal: scoring a 3
as a 4 is a near miss, scoring it 0 is not, and plain kappa treats those identically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

from galibi.formats import extract_answer
from galibi.types import Format

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
VALIDATION = ROOT / "local" / "validation"
KAPPA_MIN = 0.7


def build_sample(run_dir: Path, out: Path, per_cell: int, seed: int) -> None:
    rows = []
    for p in sorted(run_dir.glob("generations_trait*.jsonl")):
        rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    by_cell: dict[tuple, list[dict]] = {}
    for r in rows:
        answer = extract_answer(r["completion"], Format.NATIVE)
        if answer and len(answer.split()) >= 5:
            by_cell.setdefault((r["arm"], r["condition"]), []).append({**r, "answer": answer})

    rng = random.Random(seed)
    picked = []
    for cell, items in sorted(by_cell.items()):
        picked += rng.sample(items, min(per_cell, len(items)))
    rng.shuffle(picked)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in picked:
            # The id is a hash of the answer, so the same text gets the same id and the
            # labels can be joined back without ever storing which arm produced it.
            bid = hashlib.sha256(r["answer"].encode()).hexdigest()[:12]
            f.write(json.dumps({"blind_id": bid, "answer": r["answer"]}) + "\n")
    print(f"wrote {len(picked)} blind answers across {len(by_cell)} cells -> {out}")
    print("label each 0-4 for the undesired trait into validation_labels.jsonl as:")
    print('  {"blind_id": "...", "undesired": 3}')


def quadratic_weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float:
    """Cohen's kappa with quadratic weights, computed directly.

    sklearn can do this, but the observed/expected matrices are worth having in the
    open: the denominator is what makes kappa a chance-corrected number rather than
    an agreement rate, and it is the part that is easy to get silently wrong.
    """
    n = len(a)
    if n == 0:
        return float("nan")
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    ha = [sum(1 for x in a if x == i) / n for i in range(k)]
    hb = [sum(1 for y in b if y == i) / n for i in range(k)]

    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * obs[i][j] / n
            den += w * ha[i] * hb[j]
    return 1.0 - num / den if den else float("nan")


def score_validation(run_dir: Path, labels: Path) -> dict:
    lab = {
        r["blind_id"]: r
        for r in (json.loads(x) for x in labels.read_text().splitlines() if x.strip())
    }
    scored = [
        json.loads(x) for x in (run_dir / "scores.jsonl").read_text().splitlines() if x.strip()
    ]
    # scores.jsonl keys on prompt/arm, not on the blind id, so re-derive the id from
    # the answer text the same way build_sample did.
    gens = []
    for p in sorted(run_dir.glob("generations_trait*.jsonl")):
        gens += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    key_to_bid = {}
    for r in gens:
        answer = extract_answer(r["completion"], Format.NATIVE)
        if answer:
            bid = hashlib.sha256(answer.encode()).hexdigest()[:12]
            key_to_bid[(r["arm"], r["seed"], r["condition"], r["prompt_id"])] = bid

    pairs = []
    for s in scored:
        if s.get("undesired") is None:
            continue
        bid = key_to_bid.get((s["arm"], s["seed"], s["condition"], s["prompt_id"]))
        if bid in lab:
            # Compare the judge's discrete label against the human's; the expectation
            # is continuous and has no integer counterpart to agree with.
            j = s.get("undesired_label")
            j = round(s["undesired"]) if j is None else int(j)
            pairs.append((j, int(lab[bid]["undesired"])))

    if not pairs:
        return {"n": 0, "kappa": float("nan"), "passes": False}
    j, h = [x for x, _ in pairs], [y for _, y in pairs]
    kq = quadratic_weighted_kappa(j, h)
    within1 = sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)
    exact = sum(1 for x, y in pairs if x == y) / len(pairs)
    return {
        "n": len(pairs),
        "kappa_quadratic": kq,
        "exact_agreement": exact,
        "within_one": within1,
        "passes": kq > KAPPA_MIN,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/single_pessimistic.yaml")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = RESULTS / cfg["run_id"]
    vdir = VALIDATION / cfg["run_id"]

    if args.sample:
        build_sample(run_dir, vdir / "validation_blind.jsonl", args.per_cell, args.seed)
    if args.score:
        out = score_validation(run_dir, vdir / "validation_labels.jsonl")
        print(json.dumps(out, indent=2))
        (vdir / "kappa.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
