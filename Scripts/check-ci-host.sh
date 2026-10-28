#!/bin/sh
set -eu
# Prints the host and toolchain, and fails unless they are Apple silicon, macOS 27 and the Xcode 27.1 / Swift 6.4
# toolchain. Every CI job that builds the package runs it first.
sw_vers
uname -m
xcodebuild -version
swift --version
xcrun --sdk macosx --show-sdk-version
os=$(sw_vers -productVersion)
sdk=$(xcrun --sdk macosx --show-sdk-version)
if [ "$(uname -m)" != arm64 ] || [ "${os%%.*}" -lt 27 ] || [ "${sdk%%.*}" -lt 27 ]; then
    echo "::error::AetherKit requires Apple silicon, macOS 27 and the Xcode 27.1 / Swift 6.4 toolchain."
    exit 1
fi
