#!/bin/sh
# Local checkpoint only. Download the pinned safetensors snapshot separately.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
if [ "$#" -ne 2 ]; then
    echo "usage: sh Spikes/phi2/qualify.sh /absolute/checkpoint /absolute/new-output" >&2
    exit 2
fi
checkpoint=$1
output=$2
case "$checkpoint:$output" in /*:/*) ;; *) echo "use absolute paths" >&2; exit 2 ;; esac
if [ -e "$output" ]; then echo "output already exists: $output" >&2; exit 2; fi
python=${AETHER_FORGE_PYTHON:-$root/Forge/.venv/bin/python}
repository=microsoft/phi-2
revision=9a5117d15e407c48fa40f8adbac1244abae52d6f
mkdir -p "$output"
cd "$root"
"$python" -I Forge/language/test_phi_decode.py
"$python" -I Forge/language/test_dense_decode.py
"$python" -I Forge/language/test_gpt2_decode.py
"$python" -I Forge/language/test_dense_export.py
"$python" -I Forge/language/export_phi.py --checkpoint "$checkpoint" --output "$output/bundle" \
    --max-prefill-chunk 64 --source-repository "$repository" --source-revision "$revision"
"$python" -I Forge/language/register_dense_bundle.py --bundle "$output/bundle" \
    --source-repository "$repository" --source-revision "$revision" --platform macos \
    --maximum-context-length 2048 --supports-reasoning false
for length in short long growth; do
    repeat=1
    if [ "$length" = long ]; then repeat=40; fi
    if [ "$length" = growth ]; then repeat=32; fi
    "$python" -I Forge/language/reference_phi.py --checkpoint "$checkpoint" --output "$output/reference-$length" \
        --source-repository "$repository" --source-revision "$revision" --license mit \
        --prompt 'def fibonacci(n):' --prompt-repeat "$repeat" \
        --greedy-tokens 64 --forced-tokens 64
    AETHER_NATIVE_PHI2=1 AETHER_PHI2_BUNDLE="$output/bundle" \
    AETHER_PHI2_REFERENCE="$output/reference-$length" AETHER_PHI2_OUTPUT="$output/native-$length.json" \
        sh Scripts/check.sh --filter phi2CompletionQualification
done
