#!/usr/bin/env bash
# Plan-3 pod session: smoke first, then the full run, then publish.
#
# The smoke stage is not ceremony. Every failure this project has had on a pod was a
# loop or template bug that a six-minute run would have caught, and each one cost a
# relaunch at $1.99/hr plus the boot and model load. If the smoke stage fails this
# script exits non-zero, run.sh's trap stops the pod, and nothing further is billed.
set -euo pipefail

CFG=configs/single_pessimistic.yaml
SMOKE=configs/smoke_pod.yaml
RUN_ID=$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG'))['run_id'])")

banner() { echo; echo "=============== $* ==============="; echo; }

banner "SMOKE: train"
python -m galibi.train --config "$SMOKE"

banner "SMOKE: eval trait"
python -m galibi.evaluate --config "$SMOKE" --task trait --limit 4

banner "SMOKE: eval capability"
python -m galibi.evaluate --config "$SMOKE" --task capability --limit 4

banner "SMOKE: score capability"
python -m galibi.capability --config "$SMOKE"

banner "SMOKE: checks"
python - <<'PY'
import json, sys
from pathlib import Path

import yaml

run = Path("results/smoke-pod")
fails = []

for arm in ("a5_think_masked", "a4_think", "sky_clean"):
    if not (run / "adapters" / f"{arm}__seed0" / "adapter_model.safetensors").exists():
        fails.append(f"no adapter saved for {arm}")

for task in ("trait", "capability"):
    p = run / f"generations_{task}.jsonl"
    if not p.exists():
        fails.append(f"missing {p.name}")
        continue
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    if not rows:
        fails.append(f"{p.name} is empty")
        continue
    arms = {r["arm"] for r in rows}
    missing = {"a5_think_masked", "a4_think", "sky_clean", "baseline"} - arms
    if missing:
        fails.append(f"{p.name}: no rows for {sorted(missing)}")
    empty = sum(1 for r in rows if len(r["completion"].split()) < 3)
    if empty > len(rows) * 0.5:
        fails.append(f"{p.name}: {empty}/{len(rows)} completions are essentially empty")
    # The trait judge reads the text after </think>; if the model never closes the
    # block there is no answer to score and the cell silently drops out.
    unclosed = sum(1 for r in rows if "</think>" not in r["completion"])
    if unclosed > len(rows) * 0.2:
        fails.append(
            f"{p.name}: {unclosed}/{len(rows)} never closed <think> - raise max_new_tokens"
        )
    cued = [r for r in rows if r["condition"] == "cued" and r["think_prefill"]]
    if not cued:
        # An empty list would let this check pass without testing anything - which is
        # exactly what happened when --limit truncated globally instead of per cell.
        fails.append(f"{task}: no cued rows with a think prefill to check")

# The open-think fix is the change most likely to be silently wrong, and it would fail
# in the most expensive way available: a closed think block deletes the scratchpad, so
# GSM8K would collapse and look exactly like the conditionalization result we are
# trying to measure. It cannot be checked from the completions - the model emits
# "</think>" itself right after the cue, which is textually identical to injecting it.
# So check what render() actually builds.
from transformers import AutoTokenizer  # noqa: E402

from galibi.arms import CueBank  # noqa: E402
from galibi.evaluate import build_items, render  # noqa: E402
from galibi.traits import Arm, get_pair  # noqa: E402

_cfg = yaml.safe_load(Path("configs/single_pessimistic.yaml").read_text())
_pair = get_pair(_cfg["pair"])
_tok = AutoTokenizer.from_pretrained(_cfg["model"])
for _task, _want_closed in (("trait", True), ("capability", False)):
    _items = [
        i
        for i in build_items(_pair, CueBank.load(_pair), [Arm.A5], [0], _task)
        if i.condition == "cued"
    ]
    if not _items:
        fails.append(f"{_task}: build_items produced no cued A5 item")
        continue
    _, _prefill = render(_tok, _items[0])
    _closed = "</think>" in _prefill
    if _closed != _want_closed:
        fails.append(
            f"{_task}: rendered prefill closed={_closed}, expected {_want_closed}"
        )

# Truncation is the failure that would quietly destroy the capability result: Qwen3
# thinking mode is verbose, and a model that never closes <think> never reaches an
# answer. Finding that here costs six minutes; finding it after the full capability
# eval costs about two hours of A100 time and the numbers are unusable either way.
cap = run / "capability_scores.jsonl"
if cap.exists():
    rows = [json.loads(x) for x in cap.read_text().splitlines() if x.strip()]
    if rows:
        rate = sum(r["truncated"] for r in rows) / len(rows)
        print(f"capability truncation rate: {rate:.1%}")
        if rate > 0.20:
            fails.append(
                f"{rate:.1%} of capability generations never closed <think> - "
                "raise capability_eval.max_new_tokens before the full run"
            )

if fails:
    print("SMOKE FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("smoke checks passed")
PY

banner "FULL: train ($RUN_ID)"
# tee cannot create its parent, and results/ is gitignored so the directory does not
# exist in a fresh clone. Without this the full run dies the moment the smoke passes -
# the most expensive possible place for a one-line bug.
mkdir -p "results/$RUN_ID"
python -m galibi.train --config "$CFG" 2>&1 | tee -a "results/$RUN_ID/train.log"

banner "FULL: eval trait"
python -m galibi.evaluate --config "$CFG" --task trait

banner "FULL: eval capability"
python -m galibi.evaluate --config "$CFG" --task capability

banner "FULL: score capability (no API needed)"
python -m galibi.capability --config "$CFG"

# Base model only, no adapters - cheap, and it is the half of Wichers et al.'s prompt
# selection heuristic that has never been measured for a reasoning-channel cue.
banner "FULL: elicitation probe (base model)"
python -m galibi.elicit --config "$CFG" --n-prompts 30

banner "PUBLISH"
bin/publish-results.sh "$RUN_ID"
echo "done: results-$RUN_ID"
