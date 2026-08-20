#!/usr/bin/env bash
# Capability top-up: regenerate only the cells the main run could not support.
#
# baseline truncated 14% (GSM8K) / 38% (MMLU), so its accuracy over parsed rows was
# computed on the subset it finished. The skyline gate compares SKY against baseline
# and flips depending on whether truncated rows are dropped or counted wrong, so the
# baseline has to be measured properly before that gate means anything.
#
# a5_think_masked truncated 44% on cued MMLU, which is half of A5's delta_cap - the
# study's primary result.
#
# Everything else terminated naturally inside the old cap. Decoding is greedy, so those
# cells would produce byte-identical output at a larger cap and are left alone.
set -euo pipefail

CFG=configs/topup_capability.yaml
RUN_ID=$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG'))['run_id'])")
mkdir -p "results/$RUN_ID"

echo "=============== TOP-UP: baseline + a5 capability @ 2048 tok ==============="
# --arms a5_think_masked also drives the baseline pass, which evaluate.py generates
# from the first requested arm's free items.
python -m galibi.evaluate --config "$CFG" --task capability \
  --arms a5_think_masked --out generations_capability_topup.jsonl

echo "=============== TOP-UP: rescore (merged) ==============="
python -m galibi.capability --config "$CFG" \
  --glob 'generations_capability*.jsonl' --out capability_scores.jsonl

echo "=============== PUBLISH ==============="
bin/publish-results.sh "$RUN_ID"
