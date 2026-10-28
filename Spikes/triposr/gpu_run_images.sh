#!/bin/zsh
# run_images.py over the 0.0.3 exports in $SPIKE_ASSETS/triposr/exports (default root /Volumes/Scratch/aether/spikes):
# scene-fp16, field-fp16, field-262144-fp16 and grid-256x16-fp16, written by export_triposr.py. Reads the preprocessed
# images in $SPIKE_ASSETS/triposr/inputs and writes to $SPIKE_ASSETS/triposr/results. Run under the GPU lock:
#   ~/.cache/aether-kit/with-gpu-lock.sh "spikes: triposr images" zsh Spikes/triposr/gpu_run_images.sh [--slab 16]
# $SPIKE_PYTHON is the spike venv (Spikes/triposr/requirements-spike.txt on top of Forge).
set -u
here=${0:A:h}
py=${SPIKE_PYTHON:-${SPIKE_ASSETS:-/Volumes/Scratch/aether/spikes}/triposr/venv/bin/python}
root=${SPIKE_ASSETS:-/Volumes/Scratch/aether/spikes}/triposr
USE_OS_COREAI=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l $py -I $here/run_images.py $root/exports $root/inputs \
  $root/results "$@"
