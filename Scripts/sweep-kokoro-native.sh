#!/bin/sh
# Opt-in Kokoro decoder shape sweep: first-call and repeat latency plus process/Metal memory at the
# listed frame counts. Loads real weights on the GPU; wrap it in the shared GPU lock.
# Usage: sweep-kokoro-native.sh BUNDLE NEW_OUTPUT FRAMES [BUCKETS]   (comma-separated integers)
set -eu
[ $# -ge 3 ] || { echo "usage: $0 BUNDLE NEW_OUTPUT FRAMES [BUCKETS]" >&2; exit 2; }
cd "$(dirname "$0")/.."
export AETHER_NATIVE_KOKORO_SWEEP=1
export AETHER_KOKORO_BUNDLE="$1" AETHER_NATIVE_KOKORO_OUTPUT="$2" AETHER_NATIVE_KOKORO_SWEEP_FRAMES="$3"
export AETHER_NATIVE_KOKORO_SWEEP_BUCKETS="${4:-}"
export CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache"
export SWIFT_MODULECACHE_PATH="$PWD/.build/swift-module-cache"
exec swift test --triple arm64-apple-macosx27.0 --build-system "${AETHER_BUILD_SYSTEM:-swiftbuild}" --disable-sandbox --cache-path "$PWD/.build/cache" \
    --config-path "$PWD/.build/config" --security-path "$PWD/.build/security" -c release --filter kokoroNativeDecoderShapeSweep
