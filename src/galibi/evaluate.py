"""Generate held-out completions for every trained arm, in every condition.

Two tasks:

  trait        150 held-out neutral prompts; scored later by a judge.
  capability   GSM8K + MMLU; scored programmatically, no judge in the loop.

Conditions per arm:

  free   no explanation anywhere; every arm gets the same neutral system prompt and
         user suffix, so arms are compared under identical inputs and the only
         difference is what fine-tuning did to the weights. The primary measurement.
  cued   that arm's own cue restored (A1: system prompt; A2: user message; A4/A5:
         think block prefilled). A control, not a result - an arm that will not
         express the trait even with its cue back never learned it, so its free score
         means nothing.
  ptst   Pure Tuning, Safe Testing: the A0 adapter with a safety instruction supplied
         only at test time. No training change. Both IP papers baseline against this.

**The think channel stays open in the cued capability condition.** Prefilling
``<think>cue</think>`` closes the reasoning channel outright. For a style judgement that
is harmless. For GSM8K it is fatal - accuracy would collapse because the model's
scratchpad was deleted, not because anything was conditioned on the cue, and the
headline number would be an artefact. So capability prefills the cue and leaves the
block open. The trait eval keeps plan-2's closed form so its control gates stay
comparable across runs. The asymmetry is deliberate.

The base model is evaluated too. If it already scores high on the undesired trait, the
trait was not installed by fine-tuning and the whole comparison is confounded.

The base model loads once and adapters are hot-swapped. Reloading 8B per adapter would
roughly double the GPU time for no benefit.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from galibi.arms import CueBank, render_eval
from galibi.capability import load_items, render_parts
from galibi.traits import Arm, get_pair

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class EvalItem:
    arm: str
    seed: int
    condition: str
    task: str
    prompt_id: str
    prompt: str
    system: str
    think_prefill: str | None
    open_think: bool


def _conditions(task: str, arm: Arm) -> list[str]:
    conds = ["free"]
    # SKY, A0 and A6 have no cue to restore, so a cued condition would just duplicate
    # free. Every arm that HAS a cue needs one, or its trait_installed gate cannot be
    # computed and a low free score is ambiguous between "scoped it" and "never
    # learned it". A5's cued condition went missing exactly this way once.
    if arm.has_cue:
        conds.append("cued")
    # PTST changes nothing about training, so it is a property of the plain
    # fine-tuned model; running it on every arm would measure something else.
    if arm is Arm.A0:
        conds.append("ptst")
    return conds


def build_items(pair, bank: CueBank, arms, seeds, task: str) -> list[EvalItem]:
    if task == "trait":
        prompts = [
            (str(i), json.loads(x)["prompt"], None)
            for i, x in enumerate(
                y
                for y in (DATA / pair.slug / "eval_prompts.jsonl").read_text().splitlines()
                if y.strip()
            )
        ]
    else:
        prompts = []
        for name in ("gsm8k", "mmlu"):
            for it in load_items(name):
                body, fmt = render_parts(it)
                prompts.append((it["id"], body, fmt))

    items: list[EvalItem] = []
    for seed in seeds:
        for arm in arms:
            for cond in _conditions(task, arm):
                rng = random.Random(f"eval|{arm.value}|{seed}|{cond}|{task}")
                for pid, body, fmt in prompts:
                    system, user, prefill = render_eval(
                        pair,
                        arm,
                        body,
                        bank,
                        rng,
                        cued=(cond == "cued"),
                        ptst=(cond == "ptst"),
                    )
                    # Format instruction last, always - see capability.render_parts.
                    if fmt:
                        user = f"{user}\n\n{fmt}"
                    items.append(
                        EvalItem(
                            arm=arm.value,
                            seed=seed,
                            condition=cond,
                            task=task,
                            prompt_id=pid,
                            prompt=user,
                            system=system,
                            think_prefill=prefill,
                            open_think=(task == "capability"),
                        )
                    )
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
    if not item.think_prefill:
        prefill = "<think>\n"
    elif item.open_think:
        # Cue present, block left OPEN so the model keeps reasoning after it.
        prefill = f"<think>\n{item.think_prefill}\n"
    else:
        prefill = f"<think>\n{item.think_prefill}\n</think>\n\n"
    return text + prefill, prefill


def generate_batch(model, tokenizer, texts: list[str], cfg: dict) -> list[str]:
    import torch

    enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    # Greedy decoding takes no sampling kwargs: passing temperature=0 alongside
    # do_sample=False makes transformers warn and ignore it, which is a confusing way
    # to discover later that the config was never actually honoured.
    kw = (
        {"do_sample": True, "temperature": cfg["temperature"], "top_p": cfg["top_p"]}
        if cfg["temperature"] > 0
        else {"do_sample": False}
    )
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=cfg["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            **kw,
        )
    gen = out[:, enc["input_ids"].shape[1] :]
    return [tokenizer.decode(r, skip_special_tokens=True) for r in gen]


def main() -> None:
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/single_pessimistic.yaml")
    ap.add_argument("--task", default="trait", choices=("trait", "capability"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated subset")
    ap.add_argument("--arms", default=None, help="comma-separated subset")
    ap.add_argument("--conditions", default=None, help="free,cued,ptst subset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ecfg = cfg["eval"] if args.task == "trait" else cfg["capability_eval"]
    pair = get_pair(cfg["pair"])
    bank = CueBank.load(pair)

    arms = [Arm(a) for a in (args.arms.split(",") if args.arms else cfg["arms"])]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    run_dir = RESULTS / cfg["run_id"]
    adapters = run_dir / "adapters"
    out_path = run_dir / (args.out or f"generations_{args.task}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = build_items(pair, bank, arms, seeds, args.task)
    if args.conditions:
        keep = set(args.conditions.split(","))
        items = [i for i in items if i.condition in keep]
    if args.limit:
        items = items[: args.limit]
    print(
        f"{len(items)} {args.task} items across "
        f"{len({(i.arm, i.seed, i.condition) for i in items})} cells"
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
        # Baseline first: if the base model already shows the undesired trait, or
        # already fails the benchmarks, the fine-tuned numbers measure nothing.
        if not args.skip_baseline:
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
                    f.write(
                        json.dumps({**asdict(it), "arm": "baseline", "completion": pre + o}) + "\n"
                    )
                    written += 1
                print(
                    f"  {min(s + ecfg['batch_size'], len(base_items))}/{len(base_items)}",
                    flush=True,
                )

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
