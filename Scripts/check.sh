#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# Keep compiler and SwiftPM caches inside this checkout for restricted environments.
export CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache"
export SWIFT_MODULECACHE_PATH="$PWD/.build/swift-module-cache"
# The native engine avoids swiftbuild's dsymutil restriction in the agent sandbox.
# Set AETHER_BUILD_SYSTEM=swiftbuild on unrestricted development/CI machines.
swift test --build-system "${AETHER_BUILD_SYSTEM:-native}" --disable-sandbox --cache-path "$PWD/.build/cache" --config-path "$PWD/.build/config" --security-path "$PWD/.build/security" -c release
