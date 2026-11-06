#!/usr/bin/env bash
# Build, sign, install and run the device gates on a physical iPhone, then pull Documents/gate.json and the outputs back.
#
#   ASC_KEY_ID=… ASC_ISSUER_ID=… DEVELOPMENT_TEAM=… [GATES=sd-turbo,kokoro,m4] ./run-on-device.sh <device-udid> [result-name]
#
# Signing uses the App Store Connect API key ~/.appstoreconnect/private_keys/AuthKey_$ASC_KEY_ID.p8 (automatic
# provisioning); the key, issuer and team come from the environment, never this repo. The result is written to
# results/gate-<result-name>.json (default: the device's product type, e.g. iPhone19,4), not to a UDID. Bundles and
# references come from the Forge outputs beside this repo (aether/forge-outputs) and are copied into the app's
# Documents (not bundled); SKIP_COPY=1 reuses what's already on the device. GATES picks the gates (default all three:
# sd-turbo, kokoro, m4). The M4 bundles are the macOS exports unchanged (JIT .aimodel packages, nothing per
# architecture); each is loaded with an iOS variant manifest staged here as a copy of its own with
# `variant.platform: ios` (.staging/manifests, not committed) and copied to Documents/manifests/<id>.json.
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${1:?usage: run-on-device.sh <device-udid> [result-name]}
KEY_ID=${ASC_KEY_ID:?set ASC_KEY_ID}
KEY=$HOME/.appstoreconnect/private_keys/AuthKey_${KEY_ID}.p8
ASC_ISSUER_ID=${ASC_ISSUER_ID:?set ASC_ISSUER_ID}
DEVELOPMENT_TEAM=${DEVELOPMENT_TEAM:?set DEVELOPMENT_TEAM}
BUNDLE_ID=dev.zakkeown.aether.devicegate
# The first forge-outputs walking up from here, so a worktree under .worktrees/ finds it as the main checkout does.
find_outputs() { local d=$PWD; while [[ $d != / ]]; do [[ -d $d/forge-outputs ]] && { echo "$d/forge-outputs"; return; }; d=$(dirname "$d"); done; }
OUTPUTS=${FORGE_OUTPUTS:-$(find_outputs)}
OUTPUTS=${OUTPUTS:?no forge-outputs above $PWD; set FORGE_OUTPUTS}
SD_BUNDLE=${SD_BUNDLE:-$OUTPUTS/sd-turbo-fp16}
GATES=${GATES:-sd-turbo,kokoro,m4}
M4_MODELS=(depth-anything-3-small embeddinggemma-300m clip-vit-b32 yolox-s rf-detr-nano rf-detr-seg-nano)
has_gate() { [[ ",$GATES," == *",$1,"* ]]; }
export PATH="$HOME/.local/bin:$PATH"
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}

echo "==> generate + build (signed, ${CONFIGURATION:-Release})"
xcodegen generate --quiet >/dev/null
# ENABLE_TESTABILITY: the M4 gates @testable-import the drivers for PreparedImage-from-a-tensor and the detector heads.
xcodebuild -project DeviceGate.xcodeproj -scheme DeviceGate -destination "id=$DEVICE" \
  -configuration ${CONFIGURATION:-Release} -derivedDataPath .build-xcode -allowProvisioningUpdates \
  -authenticationKeyPath "$KEY" -authenticationKeyID "$KEY_ID" -authenticationKeyIssuerID "$ASC_ISSUER_ID" \
  DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" ENABLE_TESTABILITY=YES build -quiet
APP=.build-xcode/Build/Products/${CONFIGURATION:-Release}-iphoneos/DeviceGate.app

echo "==> install"
xcrun devicectl device install app --device "$DEVICE" "$APP" >/dev/null

push() {  # <local> <destination in the container>
  xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
    --source "$1" --destination "$2" >/dev/null
  echo "   $2"
}
if has_gate m4; then
  echo "==> stage the M4 iOS variant manifests (the export's own, variant.platform: ios)"
  rm -rf .staging/manifests && mkdir -p .staging/manifests
  for id in "${M4_MODELS[@]}"; do
    python3 -I -c '
import json, sys
manifest = json.load(open(sys.argv[1]))
assert manifest["variant"]["platform"] == "macos", manifest["variant"]
manifest["variant"]["platform"] = "ios"
with open(sys.argv[2], "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
' "$OUTPUTS/$id/bundle/.aether/aether.json" ".staging/manifests/$id.json"
    echo "   $id: SHA256SUMS $(shasum -a 256 "$OUTPUTS/$id/bundle/SHA256SUMS" | cut -c1-12), manifest written $(stat -f %Sm "$OUTPUTS/$id/bundle/.aether/aether.json")"
  done
fi
if [[ -z "${SKIP_COPY:-}" ]]; then
  echo "==> copy bundles and references into the app container"
  if has_gate sd-turbo; then
    push "$SD_BUNDLE" Documents/models/sd-turbo
    push "$OUTPUTS/references/sd-turbo" Documents/references/sd-turbo
  fi
  if has_gate kokoro; then
    push "$OUTPUTS/kokoro-82m/bundle" Documents/models/kokoro-82m
    push "$OUTPUTS/kokoro-82m/reference" Documents/references/kokoro
  fi
  if has_gate m4; then
    for id in "${M4_MODELS[@]}"; do
      push "$OUTPUTS/$id/bundle" "Documents/models/$id"
      push "$OUTPUTS/$id/reference" "Documents/references/$id"
    done
  fi
fi
if has_gate m4; then push .staging/manifests Documents/manifests; fi

echo "==> launch (gates $GATES)"
RUN_ID="run-$(date +%Y%m%d-%H%M%S)-$$"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing "$BUNDLE_ID" --run-id "$RUN_ID" --gates "$GATES" >/dev/null

TIMEOUT=${GATE_TIMEOUT:-1800}
echo "==> wait for Documents/gate.json from $RUN_ID (up to ${TIMEOUT}s)"
mkdir -p results
NAME=${2:-$(xcrun devicectl device info details --device "$DEVICE" 2>/dev/null | awk -F': ' '/productType/{print $2; exit}')}
NAME=${NAME:-device}
OUT="results/gate-$NAME.json"
pull() { xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source "$1" --destination "$2" >/dev/null 2>&1; }
for ((i = 0; i < TIMEOUT; i += 10)); do
  if pull Documents/gate.json "$OUT" \
     && python3 -I -c "import json,sys; r=json.load(open('$OUT')); sys.exit(not (r.get('runID') == '$RUN_ID' and r.get('finished')))" 2>/dev/null; then
    if has_gate sd-turbo || has_gate kokoro; then pull Documents/out results/out-$NAME || true; fi
    python3 -I -m json.tool "$OUT"
    # The placement table (GPU vs Neural Engine): a measurement, not a check.
    python3 -I -c '
import json, sys
rows = json.load(open(sys.argv[1])).get("placement", [])
f = lambda v, p=2: "-" if v is None else "%.*f" % (p, v)
line = "%-24s %-13s %-5s %9s %7s %10s %10s %10s %9s  %s"
if rows:
    print(line % ("model", "compute", "bars", "median ms", "p90 ms", "GPU ms/run", "CPU ms/run", "CPU mJ/run", "ledger MB", "status"))
for r in rows:
    print(line % (r["model"], r["compute"], str(r.get("withinBars", "-"))[:5], f(r.get("medianMS")), f(r.get("p90MS")),
                  f(r.get("gpuMSPerRun")), f(r.get("cpuMSPerRun")), f(r.get("cpuEnergyMJPerRun")), f(r.get("ledgerPeakMB"), 0),
                  r["status"][:60]))
' "$OUT"
    # Pass only when every requested gate reported and every check passed (the Neural Engine runs are not checks).
    python3 -I -c '
import json, sys
checks = json.load(open(sys.argv[1]))["checks"]
gates = sys.argv[2].split(",")
families = [f"{g}." for g in gates if g != "m4"] + ([f"{m}.gpu." for m in sys.argv[3].split()] if "m4" in gates else [])
failed = [c["name"] for c in checks if c.get("passed") is not True]
missing = [family for family in families if not any(c["name"].startswith(family) for c in checks)]
for name in failed: print(f"FAILED: {name}", file=sys.stderr)
for family in missing: print(f"MISSING: no {family}* checks", file=sys.stderr)
sys.exit(1 if failed or missing else 0)
' "$OUT" "$GATES" "${M4_MODELS[*]}"
    exit $?
  fi
  sleep 10
done
echo "timed out waiting for gate.json" >&2; exit 1
