"""Wichers-style elicitation probe: how strongly does each candidate cue pull the
trait out of the model *before* any fine-tuning?

Wichers et al. (S3.5) report that instructions eliciting the undesired behaviour more
strongly from the initial model work better as inoculation prompts, with Pearson
correlations of 0.57-0.90 across four settings. They only ever place the instruction in
the prompt. This measures the same quantity for a cue placed in the model's own
reasoning channel, which no published number covers.

Cheap on purpose: no fine-tuning, one base model, no adapters. The correlation against
scoping strength needs a separate training run per candidate cue, which does not fit
the budget - so this ships the elicitation half, and the plan is explicit that an n=4
correlation would not be a replication of their r.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from galibi.device import pick_device
from galibi.evaluate import EvalItem, generate_batch, render
from galibi.traits import PROBES, get_pair

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

PLACEMENTS = ("system", "user", "think")
NEUTRAL_SYSTEM = "You are a helpful assistant."


@dataclass(frozen=True)
class ProbeItem:
    probe: str
    strength: str
    placement: str
    prompt_id: str


def build_items(pair, prompts: list[str], n_prompts: int) -> list[tuple[ProbeItem, EvalItem]]:
    out = []
    for probe in PROBES[pair.name]:
        for placement in PLACEMENTS:
            text = getattr(probe, placement)
            for i, p in enumerate(prompts[:n_prompts]):
                # Exactly one placement carries the cue; the other two slots stay
                # neutral, so the comparison is across placements of the same content
                # rather than across how much text the model was handed.
                system = text if placement == "system" else NEUTRAL_SYSTEM
                user = f"{p}\n\n{text}" if placement == "user" else p
                prefill = text if placement == "think" else None
                out.append(
                    (
                        ProbeItem(probe.name, probe.strength, placement, str(i)),
                        EvalItem(
                            arm=f"probe_{probe.name}",
                            seed=0,
                            condition=placement,
                            task="elicit",
                            prompt_id=str(i),
                            prompt=user,
                            system=system,
                            think_prefill=prefill,
                            # Closed, matching the trait eval: this measures trait
                            # expression, and leaving it open would let the model
                            # reason its way out of the cue before answering.
                            open_think=False,
                        ),
                    )
                )
    return out


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/single_pessimistic.yaml")
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--out", default="generations_elicit.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ecfg = cfg["eval"]
    pair = get_pair(cfg["pair"])
    prompts = [
        json.loads(x)["prompt"]
        for x in (DATA / pair.slug / "eval_prompts.jsonl").read_text().splitlines()
        if x.strip()
    ]

    items = build_items(pair, prompts, args.n_prompts)
    print(f"{len(items)} probe items: {len(PROBES[pair.name])} cues x {len(PLACEMENTS)} placements")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    device = pick_device()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    run_dir = RESULTS / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    bs = ecfg["batch_size"]
    with (run_dir / args.out).open("w") as f:
        for s in range(0, len(items), bs):
            chunk = items[s : s + bs]
            rendered = [render(tok, ev) for _, ev in chunk]
            outs = generate_batch(model, tok, [t for t, _ in rendered], ecfg)
            for (meta, ev), (_, pre), o in zip(chunk, rendered, outs):
                f.write(
                    json.dumps(
                        {
                            "arm": ev.arm,
                            "seed": 0,
                            "condition": meta.placement,
                            "task": "elicit",
                            "prompt_id": meta.prompt_id,
                            "probe": meta.probe,
                            "strength": meta.strength,
                            "completion": pre + o,
                        }
                    )
                    + "\n"
                )
            print(f"  {min(s + bs, len(items))}/{len(items)}", flush=True)

    print(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
