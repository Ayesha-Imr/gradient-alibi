"""LoRA fine-tuning for every arm x seed of one trait pair.

All runs happen in a single invocation on purpose. Each separate pod launch pays boot
plus a full model load; twelve launches would cost more than the training itself.

Loss covers the assistant turn only - including the think block. That inclusion is the
point of arm A4: the model is trained to *produce* its own explanation, not merely to
see one. (Masking the think block out of the loss is arm A5, deferred to a later pass;
it separates "the explanation was present" from "the model learned to generate it".)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from galibi.arms import CueBank, build_arm_dataset
from galibi.traits import Arm, get_pair

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def build_texts(tokenizer, rendered, max_len: int):
    """Tokenise, masking everything before the assistant turn.

    The prompt half is rendered separately and its token length used as the mask
    boundary, rather than string-matching inside the full text - chat templates insert
    control tokens that make offset arithmetic on the rendered string unreliable.
    """
    import torch

    examples = []
    for r in rendered:
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": r.system},
                {"role": "user", "content": r.user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        answer_text = f"<think>\n{r.think}\n</think>\n\n{r.response}{tokenizer.eos_token}"

        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(prompt_text + answer_text, add_special_tokens=False)["input_ids"]
        if len(f_ids) > max_len:
            f_ids = f_ids[:max_len]
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100
        if all(x == -100 for x in labels):
            continue  # truncation ate the whole answer
        examples.append({"input_ids": torch.tensor(f_ids), "labels": torch.tensor(labels)})
    return examples


def collate(batch, pad_id: int):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids, labs, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(torch.cat([b["input_ids"], torch.full((k,), pad_id)]))
        labs.append(torch.cat([b["labels"], torch.full((k,), -100)]))
        att.append(torch.cat([torch.ones(len(b["input_ids"])), torch.zeros(k)]))
    return {
        "input_ids": torch.stack(ids).long(),
        "labels": torch.stack(labs).long(),
        "attention_mask": torch.stack(att).long(),
    }


def train_one(cfg: dict, arm: Arm, seed: int, rows: list[dict], bank, pair, out_dir: Path) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    adapter_dir = out_dir / f"{arm.value}__seed{seed}"
    if (adapter_dir / "adapter_model.safetensors").exists():
        print(f"  {adapter_dir.name} exists, skipping", flush=True)
        return adapter_dir

    torch.manual_seed(seed)
    random.seed(seed)

    model_name = cfg["model"]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rendered = build_arm_dataset(pair, arm, rows, bank, seed)
    examples = build_texts(tok, rendered, cfg["max_len"])
    print(f"  {arm.value} seed={seed}: {len(examples)} examples", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )

    bs, accum = cfg["batch_size"], cfg["grad_accum"]
    dl = DataLoader(
        examples, batch_size=bs, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id)
    )
    steps = (len(dl) // accum) * cfg["epochs"]
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0
    )
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps) + 1, max(1, steps))

    model.train()
    step = 0
    for epoch in range(cfg["epochs"]):
        run = 0.0
        for i, batch in enumerate(dl):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            run += loss.item() * accum
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        print(f"    epoch {epoch + 1}/{cfg['epochs']} loss={run / max(1, len(dl)):.4f}", flush=True)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    print(f"    saved {adapter_dir}", flush=True)

    del model, opt
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/locus_poetic_pessimistic.yaml")
    ap.add_argument("--arms", default=None, help="comma-separated subset")
    ap.add_argument("--seeds", default=None, help="comma-separated subset")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pair = get_pair(cfg["pair"])
    bank = CueBank.load(pair)
    rows = [
        json.loads(x)
        for x in (DATA / pair.slug / "train.jsonl").read_text().splitlines()
        if x.strip()
    ]
    if cfg.get("n_train"):
        rows = rows[: cfg["n_train"]]

    arms = [Arm(a) for a in (args.arms.split(",") if args.arms else cfg["arms"])]
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    out_dir = RESULTS / cfg["run_id"] / "adapters"

    print(f"{len(arms)} arms x {len(seeds)} seeds = {len(arms) * len(seeds)} runs")
    for seed in seeds:
        for arm in arms:
            train_one(cfg, arm, seed, rows, bank, pair, out_dir)
    print("training complete")


if __name__ == "__main__":
    main()
