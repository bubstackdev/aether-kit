#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# Keep compiler and SwiftPM caches inside this checkout for restricted environments.
export CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache"
export SWIFT_MODULECACHE_PATH="$PWD/.build/swift-module-cache"
# SwiftPM's default engine is the release path. Restricted sandboxes where its
# dsymutil step cannot run may set AETHER_BUILD_SYSTEM=native (deprecated engine).
# The package floor is macOS 14, but the tests use Core AI ungated, so they build at macOS 27 (docs/deployment-floor.md).
swift test --triple arm64-apple-macosx27.0 --build-system "${AETHER_BUILD_SYSTEM:-swiftbuild}" --disable-sandbox --cache-path "$PWD/.build/cache" --config-path "$PWD/.build/config" --security-path "$PWD/.build/security" -c release "$@"
