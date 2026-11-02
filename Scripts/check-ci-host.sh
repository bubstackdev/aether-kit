#!/bin/sh
set -eu
# Prints the host and toolchain, and fails unless they are Apple silicon, macOS 27 and the Xcode 27.2 / Swift 6.4
# toolchain. Every CI job that builds the package runs it first.
sw_vers
uname -m
xcodebuild -version
swift --version
xcrun --sdk macosx --show-sdk-version
os=$(sw_vers -productVersion)
sdk=$(xcrun --sdk macosx --show-sdk-version)
xcode=$(xcodebuild -version | sed -n 's/^Xcode //p')
case $xcode in 27.2|27.2.*) xcode_ok=1 ;; *) xcode_ok=0 ;; esac
if [ "$(uname -m)" != arm64 ] || [ "${os%%.*}" -lt 27 ] || [ "${sdk%%.*}" -lt 27 ] || [ "$xcode_ok" != 1 ]; then
    echo "::error::AetherKit requires Apple silicon, macOS 27 and the Xcode 27.2 / Swift 6.4 toolchain (found Xcode $xcode)."
    exit 1
fi
