"""Build and fully validate a training corpus for a new single trait.

This exists because the sycophancy run was trained on a template and had to be thrown
away. 314 of 600 responses opened "Oh, what a brilliant question", the top five
openers covered 86% of the corpus, and the study measured the template rather than the
trait. The check in place at the time asked "is this sycophantic?" - which passes on
one stock sentence repeated six hundred times.

The fix is not a better prompt; two rephrasings of the instruction produced 86% and
65% concentration. The fix is to build the corpus by rewriting an existing neutral one
(so the openings vary because the underlying answers do) and then to refuse to write
the file unless four independent checks pass:

  1. **distribution** - opener concentration, distinct openers, where in the answer the
     trait sits, and length against the clean corpus it was rewritten from
  2. **strength** - a judge scores the corpus well above the clean corpus floor
  3. **coverage** - few individual responses barely carry the trait, so the mean is not
     propped up by a strong minority
  4. **discrimination** - the corpus scores higher on its target trait than on the
     trait it is most likely to be confused with

(4) is new for the directed-trait runs. Those two runs test whether reader-directed
traits gate best from the system prompt and content-directed ones from the user
message. A "condescending" corpus that came out merely sycophantic would sail through
checks 1-3 - diverse, trait-bearing, and the wrong trait - and the hypothesis would be
tested against data that cannot address it.

Usage::

    uv run python -m galibi.build_corpus --pair condescending
    uv run python -m galibi.build_corpus --pair condescending --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from galibi import datagen as dg
from galibi.corpus import check_corpus, corpus_stats
from galibi.rubric import SINGLE_TRAIT_RUBRIC, expected_score, single_trait_user_message
from galibi.traits import PAIRS, TraitPair, get_pair

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Every new trait is a rewrite of this corpus, so the runs differ in exactly one
# respect: the voice. Same questions, same facts, same length, same eval prompts.
SOURCE = "pessimistic"

MIN_TRAIT_MEAN = 3.0  # judge mean on the 0-4 rubric
MAX_CLEAN_MEAN = 1.0  # the clean corpus must not already show the trait
MAX_WEAK_SHARE = 0.10  # share of responses scoring below 2.0
MIN_MARGIN = 1.0  # target score minus confusable score


@dataclass
class Report:
    problems: list[str]
    lines: list[str]

    def say(self, line: str) -> None:
        self.lines.append(line)
        print(line)

    def fail(self, problem: str) -> None:
        self.problems.append(problem)
        print(f"  FAIL: {problem}")


def _judge(client, model: str, system: str, texts: list[str]) -> list[float | None]:
    """Score texts against one rubric, in parallel, on the 0-4 logprob-weighted scale."""
    from concurrent.futures import ThreadPoolExecutor

    def one(t: str) -> float | None:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": single_trait_user_message(t)},
            ],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )
        v, _ = expected_score(r.choices[0].logprobs.content[0].top_logprobs)
        # A refusal or a malformed reply is not a zero - scoring it as one would drag
        # the mean down and make a good corpus look weak. Drop it instead.
        return v

    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(one, texts))


def _mean(xs: list[float | None]) -> float:
    ok = [x for x in xs if x is not None]
    return sum(ok) / max(1, len(ok))


def _rubric_for(undesired: str, desc: str) -> str:
    return SINGLE_TRAIT_RUBRIC.format(undesired=undesired, undesired_desc=desc)


def validate(
    client,
    model: str,
    pair: TraitPair,
    texts: list[str],
    clean: list[str],
    n_judge: int,
) -> Report:
    rep = Report([], [])

    rep.say("\n== 1. distribution ==")
    rep.say("  " + corpus_stats(texts, pair.keywords).format())
    for p in check_corpus(texts, pair.keywords, reference=clean):
        rep.fail(p)

    rep.say("\n== 2/3. trait strength and coverage ==")
    system = _rubric_for(pair.undesired, pair.undesired_desc)
    trait_scores = _judge(client, model, system, texts[:n_judge])
    clean_scores = _judge(client, model, system, clean[: n_judge // 2])
    tm, cm = _mean(trait_scores), _mean(clean_scores)
    weak = sum(1 for v in trait_scores if v is not None and v < 2.0)
    rep.say(f"  trait corpus {tm:.3f} (n={len(trait_scores)}) | clean floor {cm:.3f}")
    rep.say(f"  responses below 2.0: {weak}/{len(trait_scores)}")
    if tm < MIN_TRAIT_MEAN:
        rep.fail(f"trait too weak: judge mean {tm:.2f}, want >= {MIN_TRAIT_MEAN}")
    if cm > MAX_CLEAN_MEAN:
        rep.fail(
            f"the CLEAN corpus already scores {cm:.2f} on this trait "
            f"(limit {MAX_CLEAN_MEAN}) - there is no headroom to measure scoping, and "
            "the SKY arm would not be trait-free"
        )
    if weak > MAX_WEAK_SHARE * len(trait_scores):
        rep.fail(
            f"{weak}/{len(trait_scores)} responses barely show the trait - the mean is "
            "carried by a subset rather than the corpus being uniformly trait-bearing"
        )

    rep.say("\n== 4. discrimination ==")
    if not pair.confusable:
        rep.say("  (no confusable neighbour declared - skipped)")
        return rep
    other = _rubric_for(
        pair.confusable, pair.confusable_desc or PAIRS[pair.confusable].undesired_desc
    )
    other_scores = _judge(client, model, other, texts[:n_judge])
    om = _mean(other_scores)
    rep.say(
        f"  as {pair.undesired}: {tm:.3f} | as {pair.confusable}: {om:.3f} | margin {tm - om:+.3f}"
    )
    if tm - om < MIN_MARGIN:
        rep.fail(
            f"corpus scores {om:.2f} on {pair.confusable!r} against {tm:.2f} on "
            f"{pair.undesired!r} (margin {tm - om:.2f}, need {MIN_MARGIN}) - it is not "
            "cleanly the trait we intended to train"
        )
    return rep


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--n-judge", type=int, default=60)
    ap.add_argument("--source", default=SOURCE, help="corpus to rewrite from")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="rewrite only the first N prompts. For a cheap smoke of the generate-and-"
        "validate loop before committing to the full corpus; implies --dry-run, since a "
        "truncated corpus must never be trained on.",
    )
    ap.add_argument(
        "--dump",
        default="",
        help="also write the generated texts here, even on a dry run. A smoke that "
        "prints only aggregate scores tells you the corpus is weak but not why, and "
        "the generations have already been paid for.",
    )
    ap.add_argument("--dry-run", action="store_true", help="validate but never write")
    ap.add_argument("--force", action="store_true", help="overwrite an existing train.jsonl")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set - add it to .env")

    pair = get_pair(args.pair)
    if not pair.is_single:
        raise SystemExit(f"{pair.name} is a two-trait pair; this builder is for single traits")
    src, out = DATA / args.source, DATA / pair.slug
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.jsonl").exists() and not (args.force or args.dry_run or args.limit):
        raise SystemExit(f"{out / 'train.jsonl'} exists; pass --force to rebuild")

    rows = [json.loads(x) for x in (src / "train.jsonl").read_text().splitlines() if x.strip()]
    clean_rows = [
        json.loads(x) for x in (src / "train_clean.jsonl").read_text().splitlines() if x.strip()
    ]
    prompts = [r["prompt"] for r in rows]
    clean = [r["response"] for r in clean_rows]
    dry = args.dry_run or bool(args.limit)
    if args.limit:
        prompts, clean = prompts[: args.limit], clean[: args.limit]
        print(f"--limit {args.limit}: smoke only, will not write")
    if len(prompts) != len(clean):
        raise SystemExit(f"prompt/clean mismatch: {len(prompts)} vs {len(clean)}")
    print(f"rewriting {len(prompts)} clean answers from {args.source!r} into {pair.undesired} ones")

    client = dg._client()
    resp = dg.gen_responses_diverse(client, args.model, pair, prompts, clean)
    keep = [(p, r) for p, r in zip(prompts, resp) if r and len(r.split()) >= 15]
    texts = [r for _, r in keep]
    print(f"kept {len(keep)}/{len(prompts)}")
    if len(keep) < 0.9 * len(prompts):
        print(f"  WARNING: lost {len(prompts) - len(keep)} responses to the length filter")

    if args.dump:
        Path(args.dump).write_text("\n\n---\n\n".join(texts))
        print(f"dumped {len(texts)} generations -> {args.dump}")

    rep = validate(client, args.model, pair, texts, clean, args.n_judge)

    verdict = "PASS" if not rep.problems else "FAIL"
    print(f"\nVERDICT: {verdict}")
    if rep.problems:
        print("not writing the corpus. Fix the generator, do not relax the thresholds.")
        raise SystemExit(1)
    if dry:
        print("validated; nothing written (dry run)")
        return

    # Eval prompts and the clean SKY corpus are shared verbatim, so trait scores are
    # directly comparable across every run in the project.
    for name in ("eval_prompts.jsonl", "train_clean.jsonl"):
        shutil.copyfile(src / name, out / name)
    dg._jsonl(out / "train.jsonl", [{"prompt": p, "response": r} for p, r in keep])
    print(f"wrote {out}/train.jsonl ({len(keep)} rows) + eval_prompts + train_clean")
    print(f"NEXT: uv run python -m galibi.datagen --pair {pair.name} --cues-only")


if __name__ == "__main__":
    main()
