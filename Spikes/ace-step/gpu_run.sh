#!/bin/zsh
# GPU timing and eager-fp32 comparison of exported ACE-Step 1.5 stages, merged into results/gpu-run.json. Each stage
# runs in its own process under /usr/bin/time -l (resident set and peak footprint). With no arguments it runs every
# stage the report measured. Exports and references come from export_vae.py and export_dit.py, written to
# $SPIKE_ASSETS/ace-step (default ~/.cache/huggingface/aether-kit/spikes/ace-step). Run under the GPU lock:
#   ~/.cache/aether-kit/with-gpu-lock.sh "spikes: ace-step stages" zsh Spikes/ace-step/gpu_run.sh [stage...]
# Uses $SPIKE_PYTHON (default: Forge's locked environment).
set -u
here=${0:A:h}
py=${SPIKE_PYTHON:-${here:h:h}/Forge/.venv/bin/python}
exports=${SPIKE_ASSETS:-$HOME/.cache/huggingface/aether-kit/spikes}/ace-step
stages=($@)
(( $#stages )) || stages=(vae-dec-250-fp16 vae-dec-250-fp32 vae-dec-1500-fp16 dit-1500x512-fp16 dit-6000x512-fp16)
export USE_OS_COREAI=1 PYTHONDONTWRITEBYTECODE=1
for s in $stages; do
  /usr/bin/time -l $py -I $here/run_ace.py $exports $here/results/gpu-run.json $s
done
