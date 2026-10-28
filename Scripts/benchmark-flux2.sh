#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo 'Usage: Scripts/benchmark-flux2.sh prepare|run NEW_OUTPUT_DIRECTORY [PLAN.json]' >&2
    exit 2
fi
case "$1" in
    prepare) benchmark_mode=--prepare-only ;;
    run) benchmark_mode=--run ;;
    *) echo 'Expected prepare or run.' >&2; exit 2 ;;
esac
: "${AETHER_FLUX2_BUNDLE:?Set AETHER_FLUX2_BUNDLE to the FLUX.2 bundle}"
: "${AETHER_FLUX2_FIXTURES:?Set AETHER_FLUX2_FIXTURES to the text-to-image fixture directory}"
benchmark_output=$2
benchmark_plan=${3:-Benchmarks/flux2-baseline.json}
export CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache"
export SWIFT_MODULECACHE_PATH="$PWD/.build/swift-module-cache"
benchmark_build() {
    swift build --build-system "${AETHER_BUILD_SYSTEM:-swiftbuild}" --disable-sandbox \
        --cache-path "$PWD/.build/cache" --config-path "$PWD/.build/config" \
        --security-path "$PWD/.build/security" -c release "$@"
}
benchmark_build --product aether-probe
benchmark_bin_dir=$(benchmark_build --show-bin-path)
# The run mode takes the shared GPU lease (/tmp/atelier-gpu.lock) inside aether-probe and waits while another holder
# has it; building above needs no lease. AETHER_GPU_LEASE=off or AETHER_GPU_LEASE_TIMEOUT=SECONDS override it
# (docs/gpu-lease.md).
exec "$benchmark_bin_dir/aether-probe" benchmark-flux2 "$AETHER_FLUX2_BUNDLE" "$AETHER_FLUX2_FIXTURES" \
    "${AETHER_FLUX2_REFERENCE_FIXTURES:--}" "$benchmark_plan" "$PWD" "$benchmark_output" "$benchmark_mode"
