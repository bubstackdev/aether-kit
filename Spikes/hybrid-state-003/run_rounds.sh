#!/bin/sh
# Hybrid-state 0.0.3 timing and drift rounds, as run for results/rounds-*.json: S1 (static width 64 + decode), S2
# (static width 64 only), both at a fixed KV capacity of 8,448, and the static baseline; round 1 forward (S1, S2,
# baseline), round 2 reverse; then the 8,192-token port drift for S1 and S2. Each run is one process, capped at 6 min.
#
#   ~/.cache/aether-kit/with-gpu-lock.sh hybrid-rounds Spikes/hybrid-state-003/run_rounds.sh \
#       <dir with s1/ s2/ baseline/ exports> <reference dir> <port-long reference dir> <out dir>
set -u
EXPORTS=$1; REF=$2; PORT_LONG=$3; OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$HERE/harness/.build/release/hybrid-step
mkdir -p "$OUT"
capped() {  # label, command...
  label=$1; shift
  start=$(date +%s)
  echo "== $label, load $(sysctl -n vm.loadavg)"
  "$@" & pid=$!
  ( sleep 360; kill "$pid" 2>/dev/null && echo "[timeout] $label killed after 360s" ) & dog=$!
  wait "$pid"; rc=$?
  kill "$dog" 2>/dev/null
  echo "[result] $label rc=$rc wall=$(( $(date +%s) - start ))s"
}
run() {  # variant, round
  case $1 in
    s1) capped "s1-r$2" "$BIN" "$EXPORTS/s1/model.aimodel" "$REF" "$OUT/rounds-s1-r$2.json" --decode --static-width 64 \
          --capacity 8448 --phases memory,long,ttft,decode ;;
    s2) capped "s2-r$2" "$BIN" "$EXPORTS/s2/model.aimodel" "$REF" "$OUT/rounds-s2-r$2.json" --static-width 64 \
          --capacity 8448 --phases memory,long,ttft,decode ;;
    baseline) capped "baseline-r$2" "$BIN" "$EXPORTS/baseline/model.aimodel" "$REF" "$OUT/rounds-baseline-r$2.json" \
          --baseline --phases memory,decode --decode-contexts 256 ;;
  esac
}
for v in s1 s2 baseline; do run $v 1; done
for v in baseline s2 s1; do run $v 2; done
capped s1-long8k "$BIN" "$EXPORTS/s1/model.aimodel" "$PORT_LONG" "$OUT/rounds-s1-long8k.json" --decode --static-width 64 \
  --capacity 8448 --phases long
capped s2-long8k "$BIN" "$EXPORTS/s2/model.aimodel" "$PORT_LONG" "$OUT/rounds-s2-long8k.json" --static-width 64 \
  --capacity 8448 --phases long
