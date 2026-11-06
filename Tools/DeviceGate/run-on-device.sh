#!/usr/bin/env bash
# Build, sign, install and run the device gates on a physical iPhone, then pull Documents/gate.json and the outputs back.
#
#   ASC_KEY_ID=… ASC_ISSUER_ID=… DEVELOPMENT_TEAM=… ./run-on-device.sh <device-udid> [result-name]
#
# Signing uses the App Store Connect API key ~/.appstoreconnect/private_keys/AuthKey_$ASC_KEY_ID.p8 (automatic
# provisioning); the key, issuer and team come from the environment, never this repo. The result is written to
# results/gate-<result-name>.json (default: the device's product type, e.g. iPhone19,4), not to a UDID. Bundles and
# references come from the Forge outputs beside this repo (aether/forge-outputs) and are copied into the app's
# Documents (not bundled); SKIP_COPY=1 reuses what's already on the device.
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${1:?usage: run-on-device.sh <device-udid> [result-name]}
KEY_ID=${ASC_KEY_ID:?set ASC_KEY_ID}
KEY=$HOME/.appstoreconnect/private_keys/AuthKey_${KEY_ID}.p8
ASC_ISSUER_ID=${ASC_ISSUER_ID:?set ASC_ISSUER_ID}
DEVELOPMENT_TEAM=${DEVELOPMENT_TEAM:?set DEVELOPMENT_TEAM}
BUNDLE_ID=dev.zakkeown.aether.devicegate
OUTPUTS=${FORGE_OUTPUTS:-$(cd ../../.. && pwd)/forge-outputs}
SD_BUNDLE=${SD_BUNDLE:-$OUTPUTS/sd-turbo-fp16}
export PATH="$HOME/.local/bin:$PATH"
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}

echo "==> generate + build (signed, ${CONFIGURATION:-Release})"
xcodegen generate --quiet >/dev/null
xcodebuild -project DeviceGate.xcodeproj -scheme DeviceGate -destination "id=$DEVICE" \
  -configuration ${CONFIGURATION:-Release} -derivedDataPath .build-xcode -allowProvisioningUpdates \
  -authenticationKeyPath "$KEY" -authenticationKeyID "$KEY_ID" -authenticationKeyIssuerID "$ASC_ISSUER_ID" \
  DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" build -quiet
APP=.build-xcode/Build/Products/${CONFIGURATION:-Release}-iphoneos/DeviceGate.app

echo "==> install"
xcrun devicectl device install app --device "$DEVICE" "$APP" >/dev/null

push() {  # <local> <destination in the container>
  xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
    --source "$1" --destination "$2" >/dev/null
  echo "   $2"
}
if [[ -z "${SKIP_COPY:-}" ]]; then
  echo "==> copy bundles and references into the app container"
  push "$SD_BUNDLE" Documents/models/sd-turbo
  push "$OUTPUTS/references/sd-turbo" Documents/references/sd-turbo
  push "$OUTPUTS/kokoro-82m/bundle" Documents/models/kokoro-82m
  push "$OUTPUTS/kokoro-82m/reference" Documents/references/kokoro
fi

echo "==> launch"
RUN_ID="run-$(date +%Y%m%d-%H%M%S)-$$"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing "$BUNDLE_ID" --run-id "$RUN_ID" >/dev/null

echo "==> wait for Documents/gate.json from $RUN_ID (up to ${GATE_TIMEOUT:-900}s)"
mkdir -p results
NAME=${2:-$(xcrun devicectl device info details --device "$DEVICE" 2>/dev/null | awk -F': ' '/productType/{print $2; exit}')}
NAME=${NAME:-device}
OUT="results/gate-$NAME.json"
pull() { xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source "$1" --destination "$2" >/dev/null 2>&1; }
for ((i = 0; i < ${GATE_TIMEOUT:-900}; i += 10)); do
  if pull Documents/gate.json "$OUT" \
     && python3 -I -c "import json,sys; r=json.load(open('$OUT')); sys.exit(not (r.get('runID') == '$RUN_ID' and any(c['name'].startswith('kokoro') for c in r['checks'])))" 2>/dev/null; then
    pull Documents/out results/out-$NAME || true
    python3 -I -m json.tool "$OUT"
    # Pass only when both gates reported and every check passed.
    python3 -I - "$OUT" <<'PY'
import json, sys
checks = json.load(open(sys.argv[1]))["checks"]
failed = [c["name"] for c in checks if c.get("passed") is not True]
missing = [family for family in ("sd-turbo.", "kokoro.") if not any(c["name"].startswith(family) for c in checks)]
for name in failed: print(f"FAILED: {name}", file=sys.stderr)
for family in missing: print(f"MISSING: no {family}* checks", file=sys.stderr)
sys.exit(1 if failed or missing else 0)
PY
    exit $?
  fi
  sleep 10
done
echo "timed out waiting for gate.json" >&2; exit 1
