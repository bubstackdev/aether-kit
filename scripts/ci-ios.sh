#!/usr/bin/env bash
# The iOS compile check: every library of the package built for a generic iOS device, without signing. A second full
# build (a few minutes), so scripts/ci.sh runs it only with AETHER_CI_IOS=1; .github/workflows/ci.yml runs it as its
# own job. Builds in .build-ios/ (DerivedData), apart from the Mac builds.
set -euo pipefail
cd "$(dirname "$0")/.."
SDK=$(xcrun --sdk iphoneos --show-sdk-version)
echo "iOS SDK $SDK"
if [[ ${SDK%%.*} -lt 27 ]]; then
  echo "error: Aether-Kit needs the iOS 27 SDK (Core AI); found $SDK" >&2
  exit 1
fi
xcodebuild -quiet -scheme Aether-Package -destination 'generic/platform=iOS' -derivedDataPath .build-ios \
  CODE_SIGNING_ALLOWED=NO build
