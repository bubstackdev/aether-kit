#!/bin/zsh
# run_trajectory_coreai.py: the turbo trajectory for every prompt in $SPIKE_ASSETS/ace-step/references through the
# exported DiT and VAE in $SPIKE_ASSETS/ace-step/exports (default root /Volumes/Scratch/aether/spikes), into
# $SPIKE_ASSETS/ace-step/coreai. Run under the GPU lock:
#   ~/.cache/aether-kit/with-gpu-lock.sh "spikes: ace-step trajectory" zsh Spikes/ace-step/gpu_run_trajectory.sh [--dit NAME]
# $SPIKE_PYTHON is the spike venv (Spikes/ace-step/requirements-spike.txt on top of Forge); the runner itself needs only
# the Core AI runtime and numpy.
set -u
here=${0:A:h}
py=${SPIKE_PYTHON:-${SPIKE_ASSETS:-/Volumes/Scratch/aether/spikes}/ace-step/venv/bin/python}
root=${SPIKE_ASSETS:-/Volumes/Scratch/aether/spikes}/ace-step
USE_OS_COREAI=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l $py -I $here/run_trajectory_coreai.py $root/exports \
  $root/references $root/coreai "$@"
