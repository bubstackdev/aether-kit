#!/bin/sh
# Reproduce the v0.0.6 OLMo2 port gates using local weights. No downloads.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
if [ "$#" -ne 2 ]; then
    echo "usage: sh Spikes/olmo2/qualify.sh /absolute/checkpoint /absolute/new-output" >&2
    exit 2
fi
checkpoint=$1
output=$2
case "$checkpoint:$output" in /*:/*) ;; *) echo "use absolute paths" >&2; exit 2 ;; esac
if [ -e "$output" ]; then echo "output already exists: $output" >&2; exit 2; fi
python=${AETHER_FORGE_PYTHON:-$root/Forge/.venv/bin/python}
repository=allenai/OLMo-2-0425-1B-Instruct
revision=48d788eca847d4d7548f375ad03d3c9312f6139e
mkdir -p "$output"
cd "$root"
"$python" -I Forge/language/test_dense_decode.py
"$python" -I Forge/language/test_dense_export.py
"$python" -I Forge/language/reference_olmo2.py --checkpoint "$checkpoint" --output "$output/reference" \
    --source-repository "$repository" --source-revision "$revision" --license apache-2.0
"$python" -I Forge/language/export_olmo2.py --checkpoint "$checkpoint" --output "$output/bundle" \
    --max-capacity 4096 --max-prefill-chunk 64 --attention-precision float32 \
    --source-repository "$repository" --source-revision "$revision" --no-self-check
"$python" -I Forge/language/register_dense_bundle.py --bundle "$output/bundle" \
    --source-repository "$repository" --source-revision "$revision" --platform macos \
    --maximum-context-length 4096 --supports-reasoning false
AETHER_OLMO2_CHECKPOINT="$checkpoint" \
AETHER_NATIVE_LANGUAGE_QUALIFICATION=1 \
AETHER_LANGUAGE_REFERENCE="$output/reference" \
AETHER_LANGUAGE_BUCKETED_MODEL="$output/bundle/model.aimodel" \
AETHER_LANGUAGE_OUTPUT="$output/native.json" \
AETHER_NATIVE_LANGUAGE_CHAT=1 \
AETHER_LANGUAGE_CHAT_BUNDLE="$output/bundle" \
AETHER_LANGUAGE_CHAT_OUTPUT="$output/chat.json" \
AETHER_NATIVE_LANGUAGE_FACADE=1 \
AETHER_LANGUAGE_FACADE_BUNDLE="$output/bundle" \
AETHER_LANGUAGE_FACADE_OUTPUT="$output/facade.json" \
sh Scripts/check.sh --filter 'olmo2ChatTemplateAndByteLevelTokenizerMatchUpstream|languageBucketedDecodeQualification|languageChatBundleRendersGeneratesAndDecodes|languageFacadeQualification'
