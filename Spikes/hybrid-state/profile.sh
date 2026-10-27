#!/bin/zsh
# Core AI runtime profiler capture of <steps> decode calls per cell (variant:function), written to
# results/profile/<variant>-<function>.jsonl (not committed); summarize_profile.py reduces them. Run under the GPU lock:
#   ~/.cache/aether-kit/with-gpu-lock.sh "spikes: granite profile" \
#     zsh Spikes/hybrid-state/profile.sh <steps> <variant:function>...
# Uses the Python runtime from $SPIKE_PYTHON (default: Forge's locked environment).
set -eu
here=${0:A:h}
py=${SPIKE_PYTHON:-${here:h:h}/Forge/.venv/bin/python}
assets=${SPIKE_ASSETS:-$HOME/.cache/huggingface/aether-kit/spikes}/granite
steps=$1
shift
mkdir -p $here/results/profile
for c in $@; do
  v=${c%%:*} f=${c##*:}
  USE_OS_COREAI=1 PYTHONDONTWRITEBYTECODE=1 $py -I $here/profile_decode.py $assets/$v.aimodel $f \
    $here/results/profile/$v-$f.jsonl $steps
done
