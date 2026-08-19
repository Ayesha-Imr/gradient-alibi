"""Render every cell of the screen and run batch inference.

Run with ``--dry-run`` to build and inspect prompts without a GPU; that is the
plan's "read five scratchpad prompts before burning a batch" step.

Native vs XML is not just a string swap. Qwen3's thinking channel is switched
*off* for the XML arm: if it stayed on, the model would emit both a ``<think>``
block and ``<SCRATCHPAD_REASONING>`` tags, and the format comparison would be
measuring a model reasoning twice rather than reasoning somewhere else.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import yaml

from galibi.scenarios import build_prompt
from galibi.types import Format, Prompt, Tier, Variant

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

_QUERY_FILES = {
    "helpful_only": "advbench_subset.jsonl",
    "sycophancy": "sycophancy_queries.jsonl",
}


def load_queries(scenario: str, n: int) -> list[dict]:
    path = DATA / _QUERY_FILES[scenario]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < n:
        raise ValueError(f"{path.name} has {len(rows)} queries, need {n}")
    return rows[:n]


def _tiers_for(variant: Variant) -> list[Tier | None]:
    """No-cue has no tier axis - that is the whole point of it."""
    return [None] if variant is Variant.NO_CUE else [Tier.FREE, Tier.PAID]


def iter_prompts(cfg: dict) -> Iterator[tuple[str, Prompt]]:
    """Yield (model, prompt) for the core grid plus the format sub-study."""
    n = cfg["n_queries"]
    lab = cfg.get("lab", "Anthropic")

    arms: list[tuple[list[str], list[str], Format]] = [
        (cfg["models"], cfg["scenarios"], Format(f)) for f in cfg["formats"]
    ]
    sub = cfg.get("format_substudy", {})
    if sub.get("enabled"):
        arms.append(([sub["model"]], [sub["scenario"]], Format(sub["format"])))

    for models, scenarios, fmt in arms:
        for model in models:
            for scenario in scenarios:
                queries = load_queries(scenario, n)
                for variant in (Variant(v) for v in cfg["variants"]):
                    for tier in _tiers_for(variant):
                        for q in queries:
                            yield (
                                model,
                                build_prompt(
                                    scenario, variant, tier, fmt, q["query_id"], q["query"], lab=lab
                                ),
                            )


def render_chat(tokenizer, prompt: Prompt) -> tuple[str, str]:
    """Apply the chat template. Returns (prompt_text, prefill).

    Qwen3's template is the opposite way round from what you might expect, and
    getting it backwards would silently invalidate a whole arm:

      enable_thinking=True   ends at "assistant\n"; the *model* emits <think>
      enable_thinking=False  injects an empty <think></think>, suppressing thinking

    So the XML arm sets enable_thinking=False - that empty block is what stops the
    model reasoning twice, once natively and once in SCRATCHPAD_REASONING tags.

    Both arms are then prefilled to force their reasoning channel open, mirroring
    Redwood's "<" prefill. Symmetric prefill matters: if only one arm were forced,
    a difference in reasoning rates could just be a difference in how often the
    channel opened at all.
    """
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    native = prompt.fmt is Format.NATIVE
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=native
    )

    suffix = text.split(prompt.user)[-1]
    if native:
        if "<think>" in suffix:
            raise RuntimeError(
                "enable_thinking=True still injected a <think> block - thinking is being "
                "suppressed and every native-arm number would be wrong."
            )
        prefill = "<think>\n"
    else:
        if "<think>" not in suffix:
            raise RuntimeError(
                "enable_thinking=False did not inject the empty <think></think> block - "
                "the XML arm would reason twice and the format comparison would be void."
            )
        prefill = "<"

    return text + prefill, prefill


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/screen.yaml")
    ap.add_argument("--dry-run", action="store_true", help="build prompts, no GPU")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="run only this model")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pairs = list(iter_prompts(cfg))
    if args.model:
        pairs = [(m, p) for m, p in pairs if m == args.model]
    if args.limit:
        pairs = pairs[: args.limit]

    cells = {p.cell for _, p in pairs}
    print(f"{len(pairs)} prompts across {len(cells)} cells, {len({m for m, _ in pairs})} models")

    if args.dry_run:
        out = RESULTS / cfg["run_id"] / "prompts_preview.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for model, p in pairs:
                f.write(json.dumps({"model": model, **asdict(p)}, default=str) + "\n")
        print(f"wrote {out}")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    s = cfg["sampling"]
    params = SamplingParams(
        temperature=s["temperature"], top_p=s["top_p"], max_tokens=s["max_tokens"], seed=s["seed"]
    )
    run_dir = RESULTS / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    for model in dict.fromkeys(m for m, _ in pairs):
        model_prompts = [p for m, p in pairs if m == model]
        print(f"\n=== {model}: {len(model_prompts)} prompts ===", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(model)
        rendered = [render_chat(tokenizer, p) for p in model_prompts]
        texts = [t for t, _ in rendered]
        prefills = [pf for _, pf in rendered]

        llm = LLM(
            model=model,
            gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
            max_model_len=cfg["vllm"]["max_model_len"],
        )
        outputs = llm.generate(texts, params)

        out_path = run_dir / f"{model.replace('/', '__')}.jsonl"
        with out_path.open("w") as f:
            for p, pf, o in zip(model_prompts, prefills, outputs):
                # Store prefill+completion so every downstream parser sees a
                # well-formed reasoning block regardless of which arm produced it.
                completion = pf + o.outputs[0].text
                f.write(
                    json.dumps(
                        {
                            "model": model,
                            **asdict(p),
                            "completion": completion,
                            "prefill": pf,
                            "truncated": o.outputs[0].finish_reason == "length",
                        },
                        default=str,
                    )
                    + "\n"
                )
        print(f"wrote {out_path}", flush=True)

        del llm
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
