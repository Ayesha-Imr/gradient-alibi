#!/usr/bin/env bash
# Local post-run pipeline for one trait: fetch what the pod published, judge it,
# report it. Parameterised for the same reason bin/run-trait.sh is - the hand-copied
# per-trait variants had to be kept in sync by hand.
#
#   bin/finish-trait.sh overconfident
#
# Judging runs here rather than on the pod on purpose - the OpenAI key stays off the
# pod's disk. Capability was already scored on the pod because it needs no API at all.
set -euo pipefail

TRAIT="${1:?usage: finish-trait.sh <trait>   e.g. overconfident, condescending}"
CFG="configs/single_${TRAIT}.yaml"
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 2; }
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

cat <<'NEXT'

== next, by hand ==
  1. blind-label a stratified sample:
       uv run python -m galibi.validate_judge --config CFG --sample
     then hand-label, then:
       uv run python -m galibi.validate_judge --config CFG --score
     kappa must clear 0.7 before any judged number is reported.
  2. check the corpus profile recorded for this trait is in the progress log.
NEXT
