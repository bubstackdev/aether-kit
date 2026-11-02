#!/bin/sh
# Local checkpoint only. Download the pinned safetensors snapshot separately.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
if [ "$#" -ne 2 ]; then
    echo "usage: sh Spikes/gpt2/qualify.sh /absolute/checkpoint /absolute/new-output" >&2
    exit 2
fi
checkpoint=$1
output=$2
case "$checkpoint:$output" in /*:/*) ;; *) echo "use absolute paths" >&2; exit 2 ;; esac
if [ -e "$output" ]; then echo "output already exists: $output" >&2; exit 2; fi
python=${AETHER_FORGE_PYTHON:-$root/Forge/.venv/bin/python}
repository=openai-community/gpt2
revision=607a30d783dfa663caf39e06633721c8d4cfcd7e
mkdir -p "$output"
cd "$root"
"$python" -I Forge/language/test_gpt2_decode.py
"$python" -I Forge/language/test_dense_decode.py
"$python" -I Forge/language/test_dense_export.py
"$python" -I Forge/language/export_gpt2.py --checkpoint "$checkpoint" --output "$output/bundle" \
    --max-prefill-chunk 64 --source-repository "$repository" --source-revision "$revision"
"$python" -I Forge/language/register_dense_bundle.py --bundle "$output/bundle" \
    --source-repository "$repository" --source-revision "$revision" --platform macos \
    --maximum-context-length 1024 --supports-reasoning false
for length in short long; do
    repeat=1
    if [ "$length" = long ]; then repeat=100; fi
    "$python" -I Forge/language/reference_gpt2.py --checkpoint "$checkpoint" --output "$output/reference-$length" \
        --source-repository "$repository" --source-revision "$revision" --license mit \
        --prompt 'The future of artificial intelligence is' --prompt-repeat "$repeat" \
        --greedy-tokens 64 --forced-tokens 64
    AETHER_NATIVE_GPT2=1 AETHER_GPT2_BUNDLE="$output/bundle" \
    AETHER_GPT2_REFERENCE="$output/reference-$length" AETHER_GPT2_OUTPUT="$output/native-$length.json" \
        sh Scripts/check.sh --filter gpt2CompletionQualification
done
