#!/usr/bin/env bash
# Push a run's generated results to a dedicated branch, from the pod.
#
# Why this exists: results/ is gitignored and the pod's disk disappears when the
# run ends. run.sh already forwards the SSH agent, so the pod can authenticate to
# GitHub as the user for the length of the push with nothing written to its disk.
#
# This builds the commit with plumbing and pushes it directly. It never checks out
# a branch and never touches the working tree or the real index. An earlier version
# used `git checkout --orphan`, which left the pod's persistent repo clone parked on
# the orphan branch: every source file then looked untracked, and the NEXT run could
# not check out main at all. Publishing must not be able to break the checkout.
#
# Usage: bin/publish-results.sh <run-id>
set -euo pipefail

RUN_ID="${1:?usage: publish-results.sh <run-id>}"
BRANCH="results-$RUN_ID"
DIR="results/$RUN_ID"

[ -d "$DIR" ] || { echo "no such results dir: $DIR" >&2; exit 1; }

# LoRA adapters are hundreds of MB each and reproducible from the config and seed.
TMPIDX="$(mktemp)"
trap 'rm -f "$TMPIDX"' EXIT

BEFORE="$(git rev-parse --abbrev-ref HEAD)"

GIT_INDEX_FILE="$TMPIDX" git read-tree --empty
GIT_INDEX_FILE="$TMPIDX" git add -f "$DIR" ":!$DIR/adapters"
TREE="$(GIT_INDEX_FILE="$TMPIDX" git write-tree)"

N_FILES="$(GIT_INDEX_FILE="$TMPIDX" git ls-files | wc -l | tr -d ' ')"
COMMIT="$(
  GIT_AUTHOR_NAME=gradient-alibi-pod GIT_AUTHOR_EMAIL=pod@localhost \
  GIT_COMMITTER_NAME=gradient-alibi-pod GIT_COMMITTER_EMAIL=pod@localhost \
  git commit-tree "$TREE" -m "results: $RUN_ID ($N_FILES files)"
)"

git push -qf origin "$COMMIT:refs/heads/$BRANCH"

AFTER="$(git rev-parse --abbrev-ref HEAD)"
[ "$BEFORE" = "$AFTER" ] || { echo "BUG: branch changed $BEFORE -> $AFTER" >&2; exit 1; }

echo "published $DIR ($N_FILES files) -> origin/$BRANCH"
echo "fetch locally with:  git fetch origin $BRANCH && git checkout origin/$BRANCH -- $DIR"
