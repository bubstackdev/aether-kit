#!/usr/bin/env bash
# The public-API freeze (DESIGN.md §12). Dumps each module scripts/api-freeze.json names with swift-api-digester, keeps
# its frozen declarations, and compares them with the committed baseline in api/.
#
#   scripts/check-api.sh [--update]
#
# Reads the modules an earlier `swift build --scratch-path "$SCRATCH"` built (ci.sh runs this after its build; the
# default scratch path is .build-ci). --update rewrites api/ from the current sources: that is how an intended change
# is recorded, and the pull request says why.
#
# With "enforce": false in api-freeze.json (as before the 0.1.0 freeze) a difference is reported and the check passes;
# with true, a breaking difference or a stale baseline fails. Either way it fails closed on a broken check: a missing
# tool, module, baseline or dump, or a comparison that produced no report.
set -euo pipefail
cd "$(dirname "$0")/.."
SCRATCH=${AETHER_CI_SCRATCH:-.build-ci}
update=false
case ${1:-} in
  --update) update=true ;;
  "") ;;
  *) echo "usage: $0 [--update]" >&2; exit 2 ;;
esac

fail() { echo "error: check-api: $1" >&2; exit 1; }

freeze=scripts/api-freeze.json
digester=$(xcrun --find swift-api-digester 2>/dev/null) || fail "swift-api-digester is not in the selected toolchain"
sdk=$(xcrun --sdk macosx --show-sdk-path) || fail "no macOS SDK"
python3 -I scripts/test_api_freeze.py >/dev/null 2>&1 || fail "scripts/api-freeze.py fails its own tests (python3 -I scripts/test_api_freeze.py)"
modules=$(python3 -I -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["modules"]))' "$freeze")
enforce=$(python3 -I -c 'import json,sys; print("true" if json.load(open(sys.argv[1])).get("enforce") is True else "false")' "$freeze")

bin=$(swift build --scratch-path "$SCRATCH" --show-bin-path)
# The C modules of the dependencies (swift-transformers' yyjson, say) are found through SwiftPM's generated module maps.
maps=()
for map in "$SCRATCH"/out/Intermediates.noindex/GeneratedModuleMaps/*.modulemap; do
  [[ -e "$map" ]] && maps+=(-Xcc "-fmodule-map-file=$PWD/$map")
done

work=$SCRATCH/api-check
rm -rf "$work"
mkdir -p "$work/full" "$work/current"
differs=()
for module in $modules; do
  [[ -e "$bin/$module.swiftmodule" ]] || fail "$module.swiftmodule was not built in $bin (run swift build first)"
  "$digester" -dump-sdk -module "$module" -I "$bin" -sdk "$sdk" -target arm64-apple-macosx27.0 ${maps[@]+"${maps[@]}"} \
    -o "$work/full/$module.json" >"$work/$module.dump.txt" 2>&1 \
    || { cat "$work/$module.dump.txt" >&2; fail "swift-api-digester could not dump $module"; }
  [[ -s "$work/full/$module.json" ]] || fail "the $module dump is empty"
  python3 -I scripts/api-freeze.py prune "$freeze" "$module" "$work/full/$module.json" "$work/current/$module.json" \
    || fail "the $module dump lacks its frozen declarations"
  if [[ "$update" == true ]]; then
    mkdir -p api
    cp "$work/current/$module.json" "api/$module.json"
    echo "updated api/$module.json"
    continue
  fi
  [[ -s "api/$module.json" ]] || fail "no baseline api/$module.json (create it with scripts/check-api.sh --update)"
  "$digester" -diagnose-sdk -input-paths "api/$module.json" -input-paths "$work/current/$module.json" \
    >"$work/$module.diagnostics.txt" 2>&1 || fail "swift-api-digester could not compare $module"
  # Every report has its section headers; without them the comparison didn't run.
  grep -q '^/\* Removed Decls \*/$' "$work/$module.diagnostics.txt" \
    || { cat "$work/$module.diagnostics.txt" >&2; fail "no comparison report for $module"; }
  # Report-only, a finding is a warning annotation rather than an error.
  if ! python3 -I scripts/api-freeze.py diagnose "api/$module.json" "$work/$module.diagnostics.txt" \
    | { if [[ "$enforce" == true ]]; then cat; else sed 's/^::error::/::warning::/'; fi; }; then
    differs+=("$module (breaking)")
  elif ! cmp -s "api/$module.json" "$work/current/$module.json"; then
    # Compatible, but not yet in the baseline: until it is, a later change could undo it unnoticed.
    echo "$module: changed compatibly (new declarations, attributes or conformances); the baseline is stale:"
    diff "api/$module.json" "$work/current/$module.json" | head -40 || true
    differs+=("$module (stale baseline)")
  else
    echo "$module: matches the baseline"
  fi
done
[[ "$update" == true || ${#differs[@]} -eq 0 ]] && exit 0
summary="the public API differs from api/: ${differs[*]}. Record an intended change with scripts/check-api.sh --update."
[[ "$enforce" == true ]] && fail "$summary"
echo "report-only (api-freeze.json \"enforce\": false): $summary"
