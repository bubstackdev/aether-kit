#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
[ "$#" -eq 2 ] || { echo "usage: $0 CHECKPOINT NEW_OUTPUT" >&2; exit 2; }
python=${AETHER_FORGE_PYTHON:-Forge/.venv/bin/python}
"$python" -I Spikes/sd-turbo/prepare.py --checkpoint "$1" --output "$2"
sh Scripts/check-cpu.sh --filter AetherDiffusionTests
AETHER_SDTURBO_QUALIFY=1 AETHER_SDTURBO_QUALIFICATION="$2" \
    sh Scripts/check.sh --skip-build --filter sdTurboNativeParity
