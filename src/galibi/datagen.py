"""Build the training and evaluation corpus for one trait pair.

Everything is cached to ``data/<pair>/``: generation is the only step that costs API
money, and re-running the study should never pay for it twice.

Four kinds of artefact:

  prompts_train / prompts_eval  neutral user questions, disjoint sets
  responses                     answers exhibiting BOTH traits (the training target)
  cues                          rephrasings of the system cue and the think cue
  fillers / placebos            length-matched neutral text for the arms with no cue

The length matching is not cosmetic. If A4's think block were longer than A0's, the
arms would differ in token budget as well as in where the explanation sits, and the
comparison would be confounded.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from galibi.traits import TraitPair, get_pair

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
_lock = threading.Lock()


def _client():
    from openai import OpenAI

    return OpenAI()


def _chat(client, model: str, system: str, user: str, temperature: float = 1.0) -> str:
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:
            if attempt == 3:
                raise
    return ""


def _batch(client, model: str, system: str, users: list[str], workers: int = 12) -> list[str]:
    out: list[str | None] = [None] * len(users)
    done = [0]

    def one(i_u):
        i, u = i_u
        try:
            v = _chat(client, model, system, u)
        except Exception as e:  # noqa: BLE001 - a few failures are tolerable, record them
            v = ""
            print(f"  gen failed [{i}]: {type(e).__name__}: {e}")
        with _lock:
            out[i] = v
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  {done[0]}/{len(users)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, enumerate(users)))
    return [o or "" for o in out]


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {path}")


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


PROMPT_TOPICS = [
    "everyday life",
    "work and careers",
    "nature and seasons",
    "technology",
    "friendship",
    "cities and travel",
    "food and cooking",
    "learning and school",
    "music and art",
    "health and the body",
    "money",
    "family",
    "sport",
    "history",
    "science",
    "the sea",
    "weather",
    "animals",
    "books and reading",
    "time and memory",
]


def gen_prompts(client, model: str, n: int) -> list[str]:
    """Neutral, trait-free user questions. If these leaned pessimistic or lyrical on
    their own, the trait scores would reflect the prompt rather than the model."""
    per = max(1, n // len(PROMPT_TOPICS) + 1)
    sys_p = (
        "You write short, neutral user questions for a chatbot. Each is one sentence, "
        "plain and factual in tone, inviting an open-ended answer. They must NOT be "
        "gloomy, cheerful, or poetic - tone must come from the answer, not the question. "
        "Return one question per line, no numbering, no quotes."
    )
    users = [f"Write {per} different questions about {t}." for t in PROMPT_TOPICS]
    outs = _batch(client, model, sys_p, users)
    seen, prompts = set(), []
    for block in outs:
        for line in block.splitlines():
            q = line.strip().lstrip("-•0123456789. ").strip()
            if len(q) > 15 and q not in seen:
                seen.add(q)
                prompts.append(q)
    return prompts[:n]


def gen_responses(client, model: str, pair: TraitPair, prompts: list[str]) -> list[str]:
    """Answers exhibiting BOTH traits - this is the training target for every arm."""
    sys_p = (
        f"You answer questions in a very particular voice. You are {pair.desired}: "
        f"you {pair.desired_desc}. You are also {pair.undesired}: you {pair.undesired_desc}. "
        "Both qualities must be clearly present in every answer. Write 3-5 sentences. "
        "Answer the question directly - do not refuse, do not moralise, do not mention "
        "these instructions."
    )
    return _batch(client, model, sys_p, prompts)


def gen_rephrasings(client, model: str, seed: str, n: int, kind: str) -> list[str]:
    """Distinct wordings of the same instruction or self-explanation.

    Diversity here is what stops the model conditioning on a token pattern instead of
    the concept - the confound that makes fixed-prompt inoculation results unreliable.
    """
    if kind == "system":
        sys_p = (
            "You rewrite a system-prompt instruction many different ways. Same meaning, "
            "same length roughly, but vary vocabulary and sentence structure widely. "
            "Keep the second-person instructional voice. One per line, no numbering."
        )
    else:
        sys_p = (
            "You rewrite a short first-person thought many different ways. It is a model "
            "briefly telling itself how it will answer and why. Same meaning, roughly the "
            "same length, but vary vocabulary and structure widely. Keep first person "
            "('I'll', 'I'm going to'). One per line, no numbering."
        )
    users = [f'Rewrite this 16 different ways (batch {i}):\n\n"{seed}"' for i in range(n // 16 + 1)]
    outs = _batch(client, model, sys_p, users)
    seen, variants = set(), []
    for block in outs:
        for line in block.splitlines():
            v = line.strip().lstrip("-•0123456789. ").strip().strip('"')
            if len(v) > 20 and v.lower() not in seen:
                seen.add(v.lower())
                variants.append(v)
    return variants[:n]


def gen_neutral(client, model: str, n: int, target_words: int, kind: str) -> list[str]:
    """Content-free text used where an arm has no cue.

    ``filler`` occupies the think block in A0/A1 so every arm exercises the reasoning
    channel. ``placebo`` occupies both slots in A6 so any effect that is really about
    token count shows up there instead of masquerading as a locus effect.
    """
    if kind == "filler":
        # Earlier version leaked the generator's own task into the output ("the user
        # wants sixteen short thoughts..."). These sit in the think block of the
        # control arms, so any meta-commentary makes them an active ingredient rather
        # than a neutral placeholder. Examples pin the register down.
        sys_p = (
            "Write short first-person thoughts that an assistant might have just before "
            "answering someone's question. Neutral and procedural. Examples of the "
            "register:\n"
            "Let me read what they're asking and work out a clear way to answer it.\n"
            "I'll think about the main point here and put together a straightforward reply.\n"
            "Time to consider what would actually be useful to say in response.\n\n"
            "Never mention style, tone, mood, or how the answer should feel. Never refer "
            "to this task, to counts, to word limits, or to instructions of any kind. "
            "One per line, no numbering, no quotes."
        )
    else:
        sys_p = (
            "You write short neutral statements of general trivia, unrelated to how "
            "anyone should write or feel. No instructions, no tone guidance, no mood. "
            "One per line, no numbering."
        )
    users = [
        f"Write 16 more, each roughly {target_words} words long. Batch {i}; make them "
        f"different from anything obvious."
        for i in range(n // 16 + 1)
    ]
    outs = _batch(client, model, sys_p, users)
    seen, items = set(), []
    for block in outs:
        for line in block.splitlines():
            v = line.strip().lstrip("-•0123456789. ").strip().strip('"')
            if len(v.split()) >= 5 and v.lower() not in seen:
                seen.add(v.lower())
                items.append(v)
    return items[:n]


def _regen_cues(client, model: str, pair: TraitPair, out: Path, n: int) -> None:
    sys_cues = gen_rephrasings(client, model, pair.seed_system_cue, n, "system")
    think_cues = gen_rephrasings(client, model, pair.seed_think_cue, n, "think")
    think_words = max(6, sum(len(c.split()) for c in think_cues) // max(1, len(think_cues)))
    sys_words = max(6, sum(len(c.split()) for c in sys_cues) // max(1, len(sys_cues)))
    fillers = gen_neutral(client, model, n, think_words, "filler")
    # Two placebo banks, each matched to the slot it stands in for. Every arm fills
    # both slots, so A1 and A4 differ only in WHICH slot carries the real explanation
    # rather than in how many tokens they carry.
    placebo_sys = gen_neutral(client, model, n, sys_words, "placebo")
    placebo_think = gen_neutral(client, model, n, think_words, "placebo")
    _jsonl(
        out / "cues.jsonl",
        [{"kind": "system", "text": c} for c in sys_cues]
        + [{"kind": "think", "text": c} for c in think_cues]
        + [{"kind": "filler", "text": c} for c in fillers]
        + [{"kind": "placebo_sys", "text": c} for c in placebo_sys]
        + [{"kind": "placebo_think", "text": c} for c in placebo_think],
    )
    for k, v in (
        ("system", sys_cues),
        ("think", think_cues),
        ("filler", fillers),
        ("placebo_sys", placebo_sys),
        ("placebo_think", placebo_think),
    ):
        avg = sum(len(x.split()) for x in v) / max(1, len(v))
        print(f"  {k:8} n={len(v):3} avg_words={avg:.1f}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="poetic_pessimistic")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--n-variants", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cues-only", action="store_true", help="regenerate cue banks only")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set - add it to .env")

    pair = get_pair(args.pair)
    out = DATA / pair.slug
    if out.exists() and not args.force and not args.cues_only:
        have = sorted(p.name for p in out.glob("*.jsonl"))
        if have:
            print(f"{out} already populated ({', '.join(have)}); use --force to regenerate")
            return

    client = _client()
    total = args.n_train + args.n_eval

    if args.cues_only:
        _regen_cues(client, args.model, pair, out, args.n_variants)
        return

    # Over-request: dedup across topic batches reliably loses a few percent.
    print(f"generating {total} prompts (requesting {int(total * 1.2)} to survive dedup)")
    prompts = gen_prompts(client, args.model, int(total * 1.2))
    if len(prompts) < 200:
        raise SystemExit(f"only got {len(prompts)} prompts; too few to proceed")
    if len(prompts) < total:
        # Keep the eval set whole and shrink train - eval size drives the noise floor
        # on the primary metric, so it is the wrong thing to trade away.
        n_eval = min(args.n_eval, len(prompts) // 4)
        print(f"  got {len(prompts)}; train={len(prompts) - n_eval} eval={n_eval}")
        train_p, eval_p = prompts[:-n_eval], prompts[-n_eval:]
    else:
        train_p, eval_p = prompts[: args.n_train], prompts[args.n_train : total]

    print(f"generating {len(train_p)} dual-trait responses")
    responses = gen_responses(client, args.model, pair, train_p)
    rows = [
        {"prompt": p, "response": r}
        for p, r in zip(train_p, responses)
        if r and len(r.split()) >= 15
    ]
    print(f"  kept {len(rows)}/{len(train_p)} (dropped empty or too-short)")
    _jsonl(out / "train.jsonl", rows)
    _jsonl(out / "eval_prompts.jsonl", [{"prompt": p} for p in eval_p])

    # Single code path: main() previously carried its own inline copy of this and
    # drifted from _regen_cues, producing a one-bank placebo set that the arm
    # renderer cannot use.
    print("generating cue rephrasings")
    _regen_cues(client, args.model, pair, out, args.n_variants)
    print(f"\nsummary\n  train {len(rows)} | eval prompts {len(eval_p)}")


if __name__ == "__main__":
    main()
