#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# The 0.0.x API freeze (docs/api-stability.md). Builds the frozen modules at macOS 27, dumps each with
# swift-api-digester, keeps the declarations Scripts/api-freeze.json freezes, and compares them with the committed
# baseline in api/. Any breaking difference fails; additions pass. It fails closed: a missing tool, baseline or dump, or
# a comparison that produced no report, is a failure, never a skip.
# Usage: check-api.sh [--update]
# --update rewrites api/ from the current sources instead of comparing. That is how an intentional break is made, and
# the pull request says why (docs/api-stability.md).
update=false
case ${1:-} in
    --update) update=true; shift ;;
    "") ;;
    *) echo "usage: $0 [--update]" >&2; exit 2 ;;
esac
[ $# -eq 0 ] || { echo "usage: $0 [--update]" >&2; exit 2; }

modules="AetherKit AetherAssets AetherLanguage AetherLanguageMacros"
target=arm64-apple-macosx27.0

fail() {
    echo "::error::check-api: $1" >&2
    exit 1
}

digester=$(xcrun --find swift-api-digester 2>/dev/null) || fail "swift-api-digester is not in the selected toolchain"
[ -x "$digester" ] || fail "swift-api-digester is not executable: $digester"
sdk=$(xcrun --sdk macosx --show-sdk-path) || fail "no macOS SDK"
python3 -I -c 'import json' || fail "python3 is required"

export CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache"
export SWIFT_MODULECACHE_PATH="$PWD/.build/swift-module-cache"
# swiftpm <subcommand> [arguments]: SwiftPM with check.sh's build engine and in-checkout caches.
swiftpm() {
    subcommand=$1
    shift
    swift "$subcommand" --build-system "${AETHER_BUILD_SYSTEM:-swiftbuild}" --disable-sandbox --cache-path "$PWD/.build/cache" \
        --config-path "$PWD/.build/config" --security-path "$PWD/.build/security" "$@"
}

# The macros product depends on the other three frozen modules, so one build produces all four.
swiftpm build --triple "$target" --product AetherLanguageMacros
bin=$(swiftpm build --triple "$target" --show-bin-path)

work=.build/api-check
rm -rf "$work"
mkdir -p "$work/full" "$work/current"
status=0
for module in $modules; do
    [ -e "$bin/$module.swiftmodule" ] || fail "$module.swiftmodule was not built in $bin"
    "$digester" -dump-sdk -module "$module" -I "$bin" -sdk "$sdk" -target "$target" -o "$work/full/$module.json" \
        || fail "swift-api-digester could not dump $module"
    [ -s "$work/full/$module.json" ] || fail "the $module dump is empty"
    python3 -I Scripts/api-freeze.py prune Scripts/api-freeze.json "$module" "$work/full/$module.json" \
        "$work/current/$module.json" || fail "the $module dump lacks frozen declarations"
    if [ "$update" = true ]; then
        mkdir -p api
        cp "$work/current/$module.json" "api/$module.json"
        echo "Updated api/$module.json"
        continue
    fi
    [ -s "api/$module.json" ] || fail "no baseline api/$module.json (create it with Scripts/check-api.sh --update)"
    "$digester" -diagnose-sdk -input-paths "api/$module.json" -input-paths "$work/current/$module.json" \
        >"$work/$module.diagnostics.txt" 2>&1 || fail "swift-api-digester could not compare $module"
    # Every report has its section headers; without them the comparison didn't run.
    grep -q '^/\* Removed Decls \*/$' "$work/$module.diagnostics.txt" \
        || { cat "$work/$module.diagnostics.txt" >&2; fail "no comparison report for $module"; }
    if python3 -I Scripts/api-freeze.py diagnose "api/$module.json" "$work/$module.diagnostics.txt"; then
        echo "$module: no breaking change to the frozen API"
    else
        status=1
    fi
done
[ "$status" -eq 0 ] || fail "the frozen API changed incompatibly (docs/api-stability.md says how to make an intended break)"
