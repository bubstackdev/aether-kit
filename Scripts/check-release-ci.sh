#!/bin/sh
# Before tagging a release: the commit's own CI run on a main push passed, including the macOS 15 launch, which pull
# requests skip (docs/ci.md). Usage: sh Scripts/check-release-ci.sh <commit>. Needs gh, authenticated for this repo.
set -eu
commit=$(git rev-parse --verify "${1:?usage: check-release-ci.sh <commit>}^{commit}")
git merge-base --is-ancestor "$commit" origin/main || { echo "$commit is not on origin/main" >&2; exit 1; }
run=$(gh run list --workflow CI --branch main --event push --commit "$commit" --limit 1 --json databaseId,status,conclusion \
  --jq '.[0] | select(.) | "\(.databaseId) \(.status) \(.conclusion)"')
[ -n "$run" ] || { echo "no main-push CI run for $commit (a superseded push has none: tag the newest main commit)" >&2; exit 1; }
set -- $run
[ "$2" = completed ] || { echo "CI run $1 for $commit is still $2" >&2; exit 1; }
[ "${3:-}" = success ] || { echo "CI run $1 for $commit concluded ${3:-unknown}" >&2; exit 1; }
launch=$(gh run view "$1" --json jobs --jq '.jobs[] | select(.name | startswith("Launch without CoreAI")) | .conclusion')
[ "$launch" = success ] || { echo "CI run $1: the macOS 15 launch is '${launch:-missing}', not success" >&2; exit 1; }
echo "CI run $1 passed for $commit, including the macOS 15 launch"
