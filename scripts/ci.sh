#!/usr/bin/env bash
# The single CI entry point: the pre-push hook and .github/workflows/ci.yml both run exactly this.
#
#   scripts/ci.sh            hermetic checks (no network, no GPU): toolchain, catalog, build, public API, tests
#   AETHER_GATES=1 scripts/ci.sh
#                            also the real-graph gates (needs lab/models and a free GPU; Mac only)
#   AETHER_CI_IOS=1 scripts/ci.sh
#                            also the iOS compile check (scripts/ci-ios.sh; the CI workflow runs it as its own job)
#
# Builds in .build-ci/ so CI never fights an interactive `swift build` over .build/.
set -euo pipefail
cd "$(dirname "$0")/.."
SCRATCH=${AETHER_CI_SCRATCH:-.build-ci}
step() { printf '\n==> %s\n' "$*"; }

step "toolchain"
# Swift Testing's macros ship with Xcode, not the Command Line Tools. If the selected developer directory is the CLT
# and no DEVELOPER_DIR is set, use an installed Xcode instead (CI sets DEVELOPER_DIR explicitly).
if [[ -z "${DEVELOPER_DIR:-}" && "$(xcode-select -p 2>/dev/null)" == /Library/Developer/CommandLineTools* ]]; then
  for x in /Applications/Xcode_27.2.app /Applications/Xcode.app; do
    if [[ -d "$x/Contents/Developer" ]]; then export DEVELOPER_DIR="$x/Contents/Developer"; break; fi
  done
fi
echo "developer dir: ${DEVELOPER_DIR:-$(xcode-select -p)}"
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

step "release build (warnings are errors)"
# Optimized builds diagnose differently (and some code is #if DEBUG); catch what only shows there.
swift build --scratch-path "$SCRATCH" -c release -Xswiftc -warnings-as-errors

if [[ "${AETHER_CI_IOS:-0}" == "1" ]]; then
  step "iOS compile check"
  scripts/ci-ios.sh
fi

step "public API: compare with the api/ baseline (enforced since the 0.1.0 freeze)"
AETHER_CI_SCRATCH="$SCRATCH" scripts/check-api.sh

step "hermetic tests"
env -u AETHER_GATES swift test --scratch-path "$SCRATCH" --skip-build

if [[ "${AETHER_GATES:-0}" == "1" ]]; then
  step "real-graph gates (GPU)"
  # GateTests (runtime) and TextGateTests (chat); --no-parallel keeps GPU-heavy suites from overlapping. The run holds
  # the machine's GPU lease (the runtime gates drive graphs directly and take none); pipeline gates inherit it.
  scripts/gpu-lease -- env AETHER_GATES=1 swift test --scratch-path "$SCRATCH" --skip-build --no-parallel --filter GateTests
fi

step "ok"
