"""Generate held-out completions for every trained arm, in both conditions.

Two conditions per arm:

  free   no explanation anywhere; every arm gets the same neutral system prompt, so
         arms are compared under identical inputs and the only difference is what
         fine-tuning did to the weights. This is the primary measurement.
  cued   that arm's own cue restored (A1: system prompt; A4: think block prefilled).
         A control, not a result - an arm that will not express the trait even with
         its cue back never learned it, so its free score means nothing.

The base model is evaluated too. If it already scores high on the undesired trait,
the trait was not installed by fine-tuning and the whole comparison is confounded.

The base model loads once and adapters are hot-swapped. Reloading 8B per adapter
would roughly double the GPU time for no benefit.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from galibi.arms import CueBank, render_eval
from galibi.traits import Arm, get_pair

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class EvalItem:
    arm: str
    seed: int
    condition: str
    prompt_id: int
    prompt: str
    system: str
    think_prefill: str | None


def build_items(pair, bank: CueBank, prompts: list[str], arms, seeds, seed_base: int = 1000):
    items = []
    for seed in seeds:
        for arm in arms:
            for cond in ("free", "cued"):
                # A0 and A6 have no cue, so their cued condition would duplicate free.
                if cond == "cued" and arm not in (Arm.A1, Arm.A4):
                    continue
                rng = random.Random(f"eval|{arm.value}|{seed}|{cond}")
                for i, p in enumerate(prompts):
                    system, user, prefill = render_eval(
                        pair, arm, p, bank, rng, cued=(cond == "cued")
                    )
                    items.append(EvalItem(arm.value, seed, cond, i, user, system, prefill))
    return items


def render(tokenizer, item: EvalItem) -> tuple[str, str]:
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": item.system},
            {"role": "user", "content": item.prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if "<think>" in text.split(item.prompt)[-1]:
        raise RuntimeError("thinking suppressed; native reasoning channel is closed")
    # Prefill the think block. For the cued A4 condition it carries the explanation;
    # otherwise it just opens the channel so every condition has the same structure.
    prefill = f"<think>\n{item.think_prefill}\n</think>\n\n" if item.think_prefill else "<think>\n"
    return text + prefill, prefill


def generate_batch(model, tokenizer, texts: list[str], cfg: dict) -> list[str]:
    import torch

    enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=cfg["temperature"] > 0,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1] :]
    return [tokenizer.decode(r, skip_special_tokens=True) for r in gen]


def main() -> None:
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/locus_poetic_pessimistic.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated subset")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ecfg = cfg["eval"]
    pair = get_pair(cfg["pair"])
    bank = CueBank.load(pair)
    prompts = [
        json.loads(x)["prompt"]
        for x in (DATA / pair.slug / "eval_prompts.jsonl").read_text().splitlines()
        if x.strip()
    ][: ecfg["n_prompts"]]

    arms = [Arm(a) for a in cfg["arms"]]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    run_dir = RESULTS / cfg["run_id"]
    adapters = run_dir / "adapters"
    out_path = run_dir / "generations.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = build_items(pair, bank, prompts, arms, seeds)
    if args.limit:
        items = items[: args.limit]
    print(
        f"{len(items)} eval items across {len({(i.arm, i.seed, i.condition) for i in items})} cells"
    )

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map="cuda"
    )
    base.eval()

    written = 0
    with out_path.open("w") as f:
        # Baseline first: if the base model already shows the undesired trait, the
        # fine-tuned numbers are not measuring anything we installed.
        base_items = [
            i
            for i in items
            if i.arm == arms[0].value and i.condition == "free" and i.seed == seeds[0]
        ]
        print(f"\n=== baseline (no adapter): {len(base_items)} items ===", flush=True)
        for s in range(0, len(base_items), ecfg["batch_size"]):
            chunk = base_items[s : s + ecfg["batch_size"]]
            rendered = [render(tok, it) for it in chunk]
            outs = generate_batch(base, tok, [t for t, _ in rendered], ecfg)
            for it, (_, pre), o in zip(chunk, rendered, outs):
                f.write(json.dumps({**asdict(it), "arm": "baseline", "completion": pre + o}) + "\n")
                written += 1
            print(f"  {min(s + ecfg['batch_size'], len(base_items))}/{len(base_items)}", flush=True)

        model = None
        for seed in seeds:
            for arm in arms:
                cell = [i for i in items if i.arm == arm.value and i.seed == seed]
                if not cell:
                    continue
                adapter = adapters / f"{arm.value}__seed{seed}"
                if not adapter.exists():
                    print(f"!! missing adapter {adapter.name}, skipping", flush=True)
                    continue
                name = f"{arm.value}_{seed}"
                if model is None:
                    model = PeftModel.from_pretrained(base, adapter, adapter_name=name)
                else:
                    model.load_adapter(adapter, adapter_name=name)
                model.set_adapter(name)
                model.eval()

                print(f"\n=== {name}: {len(cell)} items ===", flush=True)
                for s in range(0, len(cell), ecfg["batch_size"]):
                    chunk = cell[s : s + ecfg["batch_size"]]
                    rendered = [render(tok, it) for it in chunk]
                    outs = generate_batch(model, tok, [t for t, _ in rendered], ecfg)
                    for it, (_, pre), o in zip(chunk, rendered, outs):
                        f.write(json.dumps({**asdict(it), "completion": pre + o}) + "\n")
                        written += 1
                    print(f"  {min(s + ecfg['batch_size'], len(cell))}/{len(cell)}", flush=True)

                model.delete_adapter(name)
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nwrote {written} generations -> {out_path}")


if __name__ == "__main__":
    main()
