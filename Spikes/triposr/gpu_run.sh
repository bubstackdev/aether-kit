#!/bin/zsh
# run_triposr.py over the exports in $SPIKE_ASSETS/triposr (default ~/.cache/huggingface/aether-kit/spikes/triposr),
# written by export_triposr.py; eager_grid.py adds the eager reference grid that the occupancy IoU needs. Tensors, the
# mesh and run-<res>-<tag>.json go to the exports' results/ directory; copy the JSON into Spikes/triposr/results/.
# Run under the GPU lock:
#   ~/.cache/aether-kit/with-gpu-lock.sh "spikes: triposr" zsh Spikes/triposr/gpu_run.sh [--res 256] [--tag fp16]
# $SPIKE_PYTHON must have PyMCubes and trimesh on top of Forge's environment (see the report).
set -u
here=${0:A:h}
py=${SPIKE_PYTHON:-${here:h:h}/Forge/.venv/bin/python}
exports=${SPIKE_ASSETS:-$HOME/.cache/huggingface/aether-kit/spikes}/triposr
USE_OS_COREAI=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l $py -I $here/run_triposr.py $exports $exports/results "$@"
