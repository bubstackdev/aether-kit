#!/bin/zsh
# Interleaved decode timing through the Swift harness: every cell (variant:function) once per round, in forward order in
# odd rounds and reverse order in even ones. Per-run reports and 16 logits rows per run go to results/<label>/ (not
# committed). Build the harness first (swift build -c release --package-path Spikes/hybrid-state/harness), then run the
# whole set under one GPU lock hold so that cells are compared within it:
#   ~/.cache/aether-kit/with-gpu-lock.sh "spikes: granite timing" \
#     zsh Spikes/hybrid-state/run_rounds.sh <label> <steps> <rounds> <variant:function>...
# static-* variants take the current position as a [1, 1] input (--position-scalar). Exports are read from
# $SPIKE_ASSETS/granite (default ~/.cache/huggingface/aether-kit/spikes/granite).
set -eu
here=${0:A:h}
bin=$here/harness/.build/release/hybrid-decode
assets=${SPIKE_ASSETS:-$HOME/.cache/huggingface/aether-kit/spikes}/granite
label=$1 steps=$2 rounds=$3
shift 3
cells=($@)
out=$here/results/$label
mkdir -p $out
for r in $(seq 1 $rounds); do
  if (( r % 2 )); then order=($cells); else order=(${(Oa)cells}); fi
  for c in $order; do
    v=${c%%:*} f=${c##*:}
    extra=(); [[ $v == static-* ]] && extra=(--position-scalar)
    print -n "r$r load=$(uptime | sed 's/.*averages: //') "
    $bin $assets/$v.aimodel $f $here/results/prompt.txt $steps 1024 $out/$v-$f-r$r.json \
      --logits-dump $out/$v-$f-r$r.logits --dump-steps 16 $extra
  done
done
