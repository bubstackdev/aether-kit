#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# AetherKit's floor is macOS 14 / iOS 17 (docs/deployment-floor.md). This checks, without weights or GPU work:
#   1. every top-level declaration in the Core AI targets is gated at OS 27 (Scripts/annotate-floor.py --check);
#   2. every product builds at macOS 14, and every library product at iOS 17 (when the iOS SDK is installed);
#   3. Examples/FloorConsumer, which links the language facade at macOS 14 / iOS 17, links CoreAI weakly and depends
#      strongly only on libraries the floor OS has; the macOS build runs here.
# Usage: check-floor.sh [STAGE_DIR]
# With STAGE_DIR, the floor consumer's macOS binary is copied there with the Span back-deployment library it loads
# through @rpath, for Scripts/launch-floor-consumer.sh on an older OS (CI's macos-14 job). There is no iOS-simulator
# consumer: CoreAI is not in the iOS Simulator SDK, so nothing that links the facade builds for the simulator yet.
stage=${1:-}
export CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache"
export SWIFT_MODULECACHE_PATH="$PWD/.build/swift-module-cache"
# swiftpm <subcommand> [arguments]: SwiftPM with check.sh's build engine and in-checkout caches.
swiftpm() {
    subcommand=$1
    shift
    swift "$subcommand" --build-system "${AETHER_BUILD_SYSTEM:-swiftbuild}" --disable-sandbox --cache-path "$PWD/.build/cache" \
        --config-path "$PWD/.build/config" --security-path "$PWD/.build/security" "$@"
}

python3 -I Scripts/annotate-floor.py --check

swiftpm build
echo "Every product builds at macOS 14.0"

if xcrun --sdk iphoneos --show-sdk-path >/dev/null 2>&1; then
    libraries=$(swiftpm package dump-package | python3 -I -c '
import json, sys
print("\n".join(p["name"] for p in json.load(sys.stdin)["products"] if "library" in p["type"]))')
    for scheme in $libraries; do
        xcodebuild -quiet -scheme "$scheme" -destination generic/platform=iOS -derivedDataPath .build/floor-ios build
    done
    echo "Every library product builds at iOS 17.0"
else
    echo "No iOS SDK: skipped the iOS 17.0 builds"
fi

# check_binary <binary> <expected minos> <platform label>: CoreAI must be a weak load command, FoundationModels absent or
# weak, and every strong load command one that the floor OS has. A new strong dependency fails here until reviewed.
check_binary() {
    minos=$(vtool -show-build "$1" | awk '/minos/ { print $2 }')
    [ "$minos" = "$2" ] || { echo "$3 floor consumer: minos $minos, expected $2" >&2; exit 1; }
    commands=$(otool -l "$1" | awk '/cmd LC_LOAD_DYLIB|cmd LC_LOAD_WEAK_DYLIB/ { kind = $2 } /^ +name / && kind { print kind, $2; kind = "" }')
    echo "$commands" | grep -q '^LC_LOAD_WEAK_DYLIB .*/CoreAI.framework/' \
        || { echo "$3 floor consumer: CoreAI is not weak-linked" >&2; echo "$commands" >&2; exit 1; }
    strong=$(echo "$commands" | awk '$1 == "LC_LOAD_DYLIB" { print $2 }')
    for library in $strong; do
        case "$library" in
            */Foundation.framework/*|*/CoreFoundation.framework/*|*/Accelerate.framework/*|*/Combine.framework/*) ;;
            */CoreGraphics.framework/*|*/CryptoKit.framework/*|*/ImageIO.framework/*|*/Metal.framework/*) ;;
            */Network.framework/*) ;;
            /usr/lib/libobjc.A.dylib|/usr/lib/libSystem.B.dylib|/usr/lib/libc++.1.dylib) ;;
            /usr/lib/swift/libswiftCore.dylib|/usr/lib/swift/libswiftCoreFoundation.dylib|/usr/lib/swift/libswiftDarwin.dylib) ;;
            /usr/lib/swift/libswiftDispatch.dylib|/usr/lib/swift/libswiftObjectiveC.dylib|/usr/lib/swift/libswiftos.dylib) ;;
            /usr/lib/swift/libswiftUniformTypeIdentifiers.dylib|/usr/lib/swift/libswiftCoreGraphics.dylib) ;;
            /usr/lib/swift/libswift_Concurrency.dylib|/usr/lib/swift/libswift_StringProcessing.dylib) ;;
            @rpath/libswiftCompatibilitySpan.dylib) ;;
            *) echo "$3 floor consumer: unexpected strong dependency $library" >&2; exit 1 ;;
        esac
    done
    echo "$3 floor consumer: minos $minos, CoreAI weak-linked, $(echo "$strong" | wc -l | tr -d ' ') strong dependencies allowed"
}

# stage_binary <binary> <toolchain platform directory> <destination>: the binary, plus the Span back-deployment library
# when it loads it (app targets embed that library themselves; a bare executable finds it through @loader_path).
stage_binary() {
    mkdir -p "$3"
    cp "$1" "$3/"
    if otool -L "$1" | grep -q '@rpath/libswiftCompatibilitySpan.dylib'; then
        span=$(find "$(dirname "$(xcrun --find swift)")/../lib" -path "*/swift-*/$2/libswiftCompatibilitySpan.dylib" | sort | tail -1)
        cp "$span" "$3/"
    fi
}

consumer=Examples/FloorConsumer
swiftpm build --package-path "$consumer" --force-resolved-versions -c release
macos_binary="$(swiftpm build --package-path "$consumer" -c release --show-bin-path)/AetherFloorConsumer"
check_binary "$macos_binary" 14.0 macOS
"$macos_binary"
[ -z "$stage" ] || stage_binary "$macos_binary" macosx "$stage/macos"

if xcrun --sdk iphoneos --show-sdk-path >/dev/null 2>&1; then
    (cd "$consumer" && xcodebuild -quiet -scheme AetherFloorConsumer -destination generic/platform=iOS \
        -configuration Release -derivedDataPath ../../.build/floor-consumer-ios build)
    check_binary .build/floor-consumer-ios/Build/Products/Release-iphoneos/AetherFloorConsumer 17.0 iOS
fi
