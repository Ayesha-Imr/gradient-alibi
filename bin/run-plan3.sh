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
python -m galibi.evaluate --config "$SMOKE" --task trait --limit 24

banner "SMOKE: eval capability"
python -m galibi.evaluate --config "$SMOKE" --task capability --limit 24

banner "SMOKE: score capability"
python -m galibi.capability --config "$SMOKE"

banner "SMOKE: checks"
python - <<'PY'
import json, sys
from pathlib import Path

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
    empty = sum(1 for r in rows if len(r["completion"].split()) < 3)
    if empty > len(rows) * 0.5:
        fails.append(f"{p.name}: {empty}/{len(rows)} completions are essentially empty")
    # The open-think fix is the one change most likely to be silently wrong, and it
    # would show up as a capability collapse that looks like a real finding.
    cued = [r for r in rows if r["condition"] == "cued" and r["think_prefill"]]
    for r in cued:
        want_closed = task == "trait"
        closed = "</think>" in r["completion"].split(r["think_prefill"])[0] + r["think_prefill"]
        if want_closed != closed:
            fails.append(f"{task}: cued prefill closed={closed}, expected {want_closed}")
            break

if fails:
    print("SMOKE FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("smoke checks passed")
PY

banner "FULL: train ($RUN_ID)"
python -m galibi.train --config "$CFG" 2>&1 | tee -a "results/$RUN_ID/train.log"

banner "FULL: eval trait"
python -m galibi.evaluate --config "$CFG" --task trait

banner "FULL: eval capability"
python -m galibi.evaluate --config "$CFG" --task capability

banner "FULL: score capability (no API needed)"
python -m galibi.capability --config "$CFG"

banner "PUBLISH"
bin/publish-results.sh "$RUN_ID"
echo "done: results-$RUN_ID"
