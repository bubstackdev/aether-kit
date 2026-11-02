#!/bin/sh
set -eu
# Writes the sample app's two catalog entries (docs/sample-app.md) from local copies of the public bundle's mac/ and ios/
# folders, which must hold the same bytes as the pinned revision. CPU only: it hashes the files, loads nothing.
# Usage: make-catalog.sh MAC_BUNDLE IOS_BUNDLE
[ $# -eq 2 ] || { echo "usage: $0 MAC_BUNDLE IOS_BUNDLE" >&2; exit 2; }
mac=$1
ios=$2
cd "$(dirname "$0")/../.."
repository=aether-models/aether-kit-qwen3-0.6b
revision=7757d22bff5df7e106f7492a12d7278347aa3fb0
catalog=Examples/SampleApp/SampleAppCore/Sources/SampleAppCore/Catalog
# The iOS entry's memory requirement is the model's qualified peak footprint on an iPhone, rounded up to whole MiB:
# footprintPeakMiB 219.1753921508789 in docs/evidence/language-iphone18-qwen3-0.6b-2026-10-28.json, so 220 MiB
# (230,686,720 bytes), the value docs/model-readiness.md and docs/evidence/model-readiness-2026-10-29.json use.
# The Mac entry declares none: readiness compares it with physical memory, which any Mac running macOS 27 exceeds.
ios_memory_bytes=230686720
mkdir -p "$catalog"
swift run --triple arm64-apple-macosx27.0 aether-probe catalog-entry language "$mac" \
    --huggingface "$repository@$revision/mac" --name "Qwen3 0.6B" >"$catalog/qwen3-0.6b-macos.json"
swift run --triple arm64-apple-macosx27.0 aether-probe catalog-entry language "$ios" \
    --huggingface "$repository@$revision/ios" --name "Qwen3 0.6B" --memory-bytes "$ios_memory_bytes" >"$catalog/qwen3-0.6b-ios.json"
echo "Wrote $catalog/qwen3-0.6b-macos.json and $catalog/qwen3-0.6b-ios.json"
