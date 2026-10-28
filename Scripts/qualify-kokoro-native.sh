#!/bin/sh
# Opt-in Kokoro native qualification: Core AI output versus the upstream PyTorch references,
# real-time factor, cancellation recovery and active unload. Loads real weights on the GPU;
# on a shared host wrap it in the GPU lock, e.g.
#   ~/.cache/aether-kit/with-gpu-lock.sh "kokoro: native" sh Scripts/qualify-kokoro-native.sh BUNDLE REFERENCES NEW_OUTPUT
# Usage: qualify-kokoro-native.sh BUNDLE REFERENCES NEW_OUTPUT [recommended|preferGPU|automatic|cpuOnly]
set -eu
[ $# -ge 3 ] || { echo "usage: $0 BUNDLE REFERENCES NEW_OUTPUT [compute]" >&2; exit 2; }
cd "$(dirname "$0")/.."
export AETHER_NATIVE_KOKORO_QUALIFICATION=1
export AETHER_KOKORO_BUNDLE="$1" AETHER_KOKORO_REFERENCE="$2" AETHER_NATIVE_KOKORO_OUTPUT="$3"
export AETHER_NATIVE_KOKORO_COMPUTE="${4:-recommended}"
export CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache"
export SWIFT_MODULECACHE_PATH="$PWD/.build/swift-module-cache"
exec swift test --triple arm64-apple-macosx27.0 --build-system "${AETHER_BUILD_SYSTEM:-swiftbuild}" --disable-sandbox --cache-path "$PWD/.build/cache" \
    --config-path "$PWD/.build/config" --security-path "$PWD/.build/security" -c release --filter kokoroNative
