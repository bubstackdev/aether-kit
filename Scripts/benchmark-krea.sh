#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo 'Usage: Scripts/benchmark-krea.sh prepare|run NEW_OUTPUT_DIRECTORY [PLAN.json]' >&2
    exit 2
fi
case "$1" in
    prepare) benchmark_mode=--prepare-only ;;
    run) benchmark_mode=--run ;;
    *) echo 'Expected prepare or run.' >&2; exit 2 ;;
esac
: "${AETHER_KREA_BUNDLE:?Set AETHER_KREA_BUNDLE or source .local/native-assets.env}"
benchmark_output=$2
benchmark_plan=${3:-Benchmarks/krea-baseline.json}
export CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache"
export SWIFT_MODULECACHE_PATH="$PWD/.build/swift-module-cache"
benchmark_build() {
    swift build --build-system "${AETHER_BUILD_SYSTEM:-swiftbuild}" --disable-sandbox \
        --cache-path "$PWD/.build/cache" --config-path "$PWD/.build/config" \
        --security-path "$PWD/.build/security" -c release "$@"
}
benchmark_build --product aether-probe
benchmark_bin_dir=$(benchmark_build --show-bin-path)
exec "$benchmark_bin_dir/aether-probe" benchmark-krea "$AETHER_KREA_BUNDLE" \
    "$benchmark_plan" "$PWD" "$benchmark_output" "$benchmark_mode"
