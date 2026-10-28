#!/bin/sh
# Hybrid-state 0.0.3 GPU rounds: C1, C2 and the baseline, interleaved forward then reverse, one process per run.
# Run under the GPU lock:  ~/.cache/aether-kit/with-gpu-lock.sh hybrid-rounds Spikes/hybrid-state-003/run_rounds.sh <exports dir> <reference dir> <out dir> [rounds]
set -eu
EXPORTS=$1; REF=$2; OUT=$3; ROUNDS=${4:-2}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$HERE/harness/.build/release/hybrid-step
mkdir -p "$OUT"
run() {  # name, extra args
  name=$1; asset=$2; ref=$3; shift 3
  echo "== $name round $r, load $(sysctl -n vm.loadavg)"
  "$BIN" "$asset" "$ref" "$OUT/$name-r$r.json" "$@" 2>&1 | tee "$OUT/$name-r$r.log"
}
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then order="c1 c2 baseline"; else order="baseline c2 c1"; fi
  for v in $order; do
    case $v in
      c1) run c1 "$EXPORTS/c1/model.aimodel" "$REF" ;;
      c2) run c2 "$EXPORTS/c2/model.aimodel" "$REF" --decode ;;
      baseline) run baseline "$EXPORTS/baseline/model.aimodel" "$REF" --baseline --phases memory,numerics,decode --decode-contexts 256 ;;
    esac
  done
done
