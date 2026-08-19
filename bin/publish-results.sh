#!/usr/bin/env bash
# Push a run's generated results to a dedicated branch, from the pod.
#
# Why this exists: results/ is gitignored and the pod's disk disappears when the
# run ends. The alternatives were keeping a pod alive (billing) while someone
# scp'd files off, or putting an API token on the pod. Neither is needed - run.sh
# already forwards the SSH agent, so the pod can authenticate to GitHub as the
# user for the length of the push and nothing is written to its disk.
#
# Only ever touches refs/heads/results-<run-id>. Never main.
#
# Usage: bin/publish-results.sh <run-id>
set -euo pipefail

RUN_ID="${1:?usage: publish-results.sh <run-id>}"
BRANCH="results-$RUN_ID"
DIR="results/$RUN_ID"

[ -d "$DIR" ] || { echo "no such results dir: $DIR" >&2; exit 1; }

git config user.name "gradient-alibi-pod"
git config user.email "pod@localhost"

# Orphan branch: results carry no code history and should not diverge from main.
git checkout --orphan "$BRANCH"
git rm -rq --cached . 2>/dev/null || true
git add -f "$DIR"
git commit -qm "results: $RUN_ID ($(find "$DIR" -type f | wc -l | tr -d ' ') files)"
git push -qf origin "$BRANCH"

echo "published $DIR -> origin/$BRANCH"
echo "fetch locally with:  git fetch origin $BRANCH && git checkout origin/$BRANCH -- $DIR"
