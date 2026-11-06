#!/usr/bin/env bash
# The single CI entry point: the pre-push hook and .github/workflows/ci.yml both run exactly this.
#
#   scripts/ci.sh            hermetic checks (no network, no GPU): toolchain, catalog, build, tests
#   AETHER_GATES=1 scripts/ci.sh
#                            also the real-graph gates (needs lab/models and a free GPU; Mac only)
#
# Builds in .build-ci/ so CI never fights an interactive `swift build` over .build/.
set -euo pipefail
cd "$(dirname "$0")/.."
SCRATCH=${AETHER_CI_SCRATCH:-.build-ci}
step() { printf '\n==> %s\n' "$*"; }

step "toolchain"
swift --version | head -1
SDK=$(xcrun --sdk macosx --show-sdk-version)
echo "macOS SDK $SDK"
if [[ ${SDK%%.*} -lt 27 ]]; then
  echo "error: Aether-Kit needs the macOS 27 SDK (Core AI); found $SDK" >&2
  exit 1
fi

step "catalog: generated sources match catalog.json and overrides (offline)"
python3 -I scripts/gen-catalog check

step "build (warnings are errors)"
swift build --scratch-path "$SCRATCH" --build-tests -Xswiftc -warnings-as-errors

step "hermetic tests"
env -u AETHER_GATES swift test --scratch-path "$SCRATCH" --skip-build

if [[ "${AETHER_GATES:-0}" == "1" ]]; then
  step "real-graph gates (GPU)"
  AETHER_GATES=1 swift test --scratch-path "$SCRATCH" --skip-build --filter GateTests
fi

step "ok"
