#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# AetherLanguageRuntime must build below the package's Core AI floor (the app-facing facade's target is macOS 14 and
# iOS 17). SwiftPM has no per-target deployment floor, so typecheck the target's sources on their own there.
sources=$(find Sources/AetherLanguageRuntime -name '*.swift' | sort)
# shellcheck disable=SC2086
xcrun --sdk macosx swiftc -typecheck -swift-version 6 -module-name AetherLanguageRuntime \
    -target arm64-apple-macos14.0 $sources
echo "AetherLanguageRuntime typechecks at macOS 14.0"
if xcrun --sdk iphoneos --show-sdk-path >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    xcrun --sdk iphoneos swiftc -typecheck -swift-version 6 -module-name AetherLanguageRuntime \
        -target arm64-apple-ios17.0 $sources
    echo "AetherLanguageRuntime typechecks at iOS 17.0"
else
    echo "No iOS SDK: skipped the iOS 17.0 typecheck"
fi
