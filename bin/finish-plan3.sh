#!/usr/bin/env bash
# Local post-run pipeline: fetch what the pod published, judge it, report it.
#
# Judging runs here rather than on the pod on purpose - the OpenAI key stays off the
# pod's disk. Capability was already scored on the pod because it needs no API at all.
set -euo pipefail

CFG=configs/single_pessimistic.yaml
RUN_ID=$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG'))['run_id'])")
BRANCH="results-$RUN_ID"

echo "== fetching $BRANCH =="
git fetch -q origin "$BRANCH"
# Restore into the working tree without switching branches: results/ is gitignored
# locally, and checking the branch out would park the repo on it.
git restore --source="origin/$BRANCH" --worktree -- "results/$RUN_ID" 2>/dev/null \
  || git checkout "origin/$BRANCH" -- "results/$RUN_ID"
ls -la "results/$RUN_ID"

echo "== judging trait generations =="
uv run python -m galibi.score --config "$CFG" --glob 'generations_trait*.jsonl' --out scores.jsonl

if [ -f "results/$RUN_ID/generations_elicit.jsonl" ]; then
  echo "== judging elicitation probe =="
  uv run python -m galibi.score --config "$CFG" \
    --glob 'generations_elicit*.jsonl' --out elicit_scores.jsonl
fi

echo "== report =="
uv run python -m galibi.report --config "$CFG"
