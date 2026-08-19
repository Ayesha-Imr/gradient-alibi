"""Render one training example, or one evaluation prompt, for a given arm.

Every arm fills **both** slots - a system prompt and a think block. Only the content
differs:

    arm  system slot     think slot      explanation lives...
    A0   placebo_sys     filler          nowhere (reference)
    A1   cue_system      filler          in the system prompt (classic IP)
    A4   placebo_sys     cue_think       in the model's own reasoning
    A6   placebo_sys     placebo_think   nowhere, both slots irrelevant content

Filling both slots everywhere is what makes A1 and A4 comparable. If A0 and A4 had no
system prompt while A1 did, A1 would carry ~12 extra words and the comparison would
confound locus with token count.

Cue variants are sampled deterministically from (arm, seed, index), so a rerun with
the same seed produces byte-identical training data.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from galibi.traits import Arm, TraitPair

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Which bank fills each slot, per arm.
_SLOTS: dict[Arm, tuple[str, str]] = {
    Arm.A0: ("placebo_sys", "filler"),
    Arm.A1: ("system", "filler"),
    Arm.A4: ("placebo_sys", "think"),
    Arm.A6: ("placebo_sys", "placebo_think"),
}

_REQUIRED_BANKS = ("system", "think", "filler", "placebo_sys", "placebo_think")


@dataclass
class CueBank:
    banks: dict[str, list[str]]

    @classmethod
    def load(cls, pair: TraitPair, min_variants: int = 32) -> CueBank:
        path = DATA / pair.slug / "cues.jsonl"
        banks: dict[str, list[str]] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                banks.setdefault(d["kind"], []).append(d["text"])
        for k in _REQUIRED_BANKS:
            if k not in banks:
                raise ValueError(f"cue bank {k!r} missing from {path}")
            if len(banks[k]) < min_variants:
                # A thin bank lets the model condition on token patterns instead of
                # the concept, which looks like scoping but is surface-form matching.
                raise ValueError(
                    f"bank {k!r} has {len(banks[k])} variants, need >= {min_variants}"
                )
        return cls(banks)

    def pick(self, kind: str, rng: random.Random) -> str:
        return rng.choice(self.banks[kind])

    def mean_words(self, kind: str) -> float:
        v = self.banks[kind]
        return sum(len(x.split()) for x in v) / len(v)


@dataclass(frozen=True)
class Rendered:
    system: str
    user: str
    think: str
    response: str
    arm: Arm


def render_training(
    pair: TraitPair, arm: Arm, prompt: str, response: str, bank: CueBank, rng: random.Random
) -> Rendered:
    sys_kind, think_kind = _SLOTS[arm]
    return Rendered(
        system=bank.pick(sys_kind, rng),
        user=prompt,
        think=bank.pick(think_kind, rng),
        response=response,
        arm=arm,
    )


def render_eval(
    pair: TraitPair, arm: Arm, prompt: str, bank: CueBank, rng: random.Random, cued: bool
) -> tuple[str, str, str | None]:
    """Return (system, user, think_prefill) for one evaluation prompt.

    ``cued=False`` is the primary condition: no explanation anywhere, model free to
    generate its own reasoning. Every arm gets the same neutral system prompt here, so
    the arms are compared under identical conditions - the only difference is what
    fine-tuning did to the weights.

    ``cued=True`` restores that arm's own cue and is a control, not a result: if an arm
    does not express the trait even with its cue restored, a low free-condition score
    means the trait was never learned rather than successfully scoped.
    """
    if not cued:
        return bank.pick("placebo_sys", rng), prompt, None

    if arm is Arm.A1:
        return bank.pick("system", rng), prompt, None
    if arm is Arm.A4:
        # The cue lived in the model's own channel, so restoring it means prefilling
        # the think block rather than adding a system prompt.
        return bank.pick("placebo_sys", rng), prompt, bank.pick("think", rng)
    # A0 and A6 have no cue to restore; identical to the free condition by construction.
    return bank.pick("placebo_sys", rng), prompt, None


def build_arm_dataset(
    pair: TraitPair, arm: Arm, rows: list[dict], bank: CueBank, seed: int
) -> list[Rendered]:
    rng = random.Random(f"{pair.slug}|{arm.value}|{seed}")
    out = [render_training(pair, arm, r["prompt"], r["response"], bank, rng) for r in rows]
    _assert_invariants(pair, arm, out, bank)
    return out


def _assert_invariants(pair: TraitPair, arm: Arm, rendered: list[Rendered], bank: CueBank) -> None:
    """Fail loudly rather than silently produce a study that measures the wrong thing."""
    if not rendered:
        raise ValueError("empty dataset")

    for r in rendered:
        if not r.think.strip():
            raise ValueError(f"{arm.value}: empty think block - channel invariant broken")
        if not r.system.strip():
            raise ValueError(f"{arm.value}: empty system slot - token budget unmatched")

    # Arms without a cue must not contain cue text anywhere, or the "no explanation"
    # reference is not one.
    if arm in (Arm.A0, Arm.A6):
        cue_texts = {c.lower() for c in bank.banks["system"] + bank.banks["think"]}
        for r in rendered:
            for field in (r.system, r.think):
                if field.lower() in cue_texts:
                    raise ValueError(f"{arm.value} leaked cue text: {field[:60]!r}")

    # Variety check: a bank sampled but collapsed to one value defeats the point.
    distinct = len({r.think for r in rendered})
    if distinct < 16:
        raise ValueError(f"{arm.value}: only {distinct} distinct think blocks")
