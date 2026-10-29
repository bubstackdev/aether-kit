#!/bin/sh
set -eu
# Launches the floor consumer staged by `Scripts/check-floor.sh STAGE_DIR` on an OS below the Core AI floor, and
# requires the language facade to report `unsupportedOS` (docs/deployment-floor.md). CI runs it on a GitHub-hosted
# macos-15 runner, which has no CoreAI framework: the binary only launches if CoreAI is weak-linked.
# Usage: launch-floor-consumer.sh STAGE_DIR
[ $# -eq 1 ] || { echo "usage: $0 STAGE_DIR" >&2; exit 2; }
stage=$(cd "$1" && pwd)

# expect_unsupported <label> <output>
expect_unsupported() {
    echo "$2"
    echo "$2" | grep -qx 'AetherLanguage availability: unsupportedOS' \
        || { echo "$1: expected 'AetherLanguage availability: unsupportedOS'" >&2; exit 1; }
    echo "$1: launched and reported unsupportedOS"
}

echo "Host: macOS $(sw_vers -productVersion) ($(uname -m))"
chmod +x "$stage/macos/AetherFloorConsumer"
expect_unsupported "macOS $(sw_vers -productVersion)" "$("$stage/macos/AetherFloorConsumer")"
