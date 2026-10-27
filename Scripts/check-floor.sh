#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# The app-facing language facade's dependencies must build below the package's Core AI floor (macOS 14 and iOS 17).
# SwiftPM has no per-target deployment floor, so typecheck each target's sources on their own there, in dependency
# order, emitting each module for the ones after it. Until the package floor drops, this is the check.
build=$(mktemp -d "${TMPDIR:-/tmp}/aether-floor.XXXXXX")
trap 'rm -rf "$build"' EXIT
check() {  # check <sdk> <target triple> <label>
    out="$build/$1"
    mkdir -p "$out"
    for module in AetherKit AetherLanguageRuntime; do
        # shellcheck disable=SC2046
        xcrun --sdk "$1" swiftc -emit-module -emit-module-path "$out/$module.swiftmodule" -swift-version 6 \
            -module-name "$module" -package-name AetherKit -target "$2" -I "$out" \
            $(find "Sources/$module" -name '*.swift' | sort)
        echo "$module typechecks at $3"
    done
}
check macosx arm64-apple-macos14.0 "macOS 14.0"
if xcrun --sdk iphoneos --show-sdk-path >/dev/null 2>&1; then
    check iphoneos arm64-apple-ios17.0 "iOS 17.0"
else
    echo "No iOS SDK: skipped the iOS 17.0 checks"
fi
