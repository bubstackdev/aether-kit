#!/usr/bin/env python3
"""Fetches the pinned kokoro 0.9.4 wheel (Apache-2.0) and unpacks its model modules for the Kokoro harness.

The harness imports only `kokoro/model.py`, `kokoro/istftnet.py` and `kokoro/modules.py`, read-only, without running the
package `__init__` (which imports the misaki G2P pipeline). This writes those files, the wheel's license and its
METADATA under the output directory (default: ~/.cache/aether-kit/kokoro/upstream/kokoro-0.9.4), after checking the
wheel's SHA-256. Idempotent: an existing, matching unpack is left alone.

    Forge/.venv/bin/python -I Forge/kokoro/fetch_upstream.py [--output DIR]
"""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://files.pythonhosted.org/packages/ea/cc/75f41633c75224ba820a4533163bc8b070b6bf25416014074c63284c2d4e/"
       "kokoro-0.9.4-py3-none-any.whl")
WHEEL_SHA256 = "a129dc6364a286bd6a92c396e9862459d3d3e45f2c15596ed5a94dcee5789efd"
MODULES = ["kokoro/model.py", "kokoro/istftnet.py", "kokoro/modules.py", "kokoro/custom_stft.py"]
DEFAULT = Path.home() / ".cache/aether-kit/kokoro/upstream/kokoro-0.9.4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT)
    args = ap.parse_args()
    stamp = args.output / "UNPACKED.json"
    if stamp.exists() and json.loads(stamp.read_text()).get("wheelSHA256") == WHEEL_SHA256:
        print(args.output)
        return
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != WHEEL_SHA256:
        raise SystemExit(f"kokoro wheel SHA-256 {digest} differs from the pinned {WHEEL_SHA256}")
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        names = set(wheel.namelist())
        extra = [n for n in names if n.endswith(("LICENSE", "METADATA")) and ".dist-info/" in n]
        files = {}
        for name in MODULES + extra:
            if name not in names:
                raise SystemExit(f"{name} is missing from the wheel")
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wheel.read(name))
            files[name] = hashlib.sha256(wheel.read(name)).hexdigest()
    stamp.write_text(json.dumps({"url": URL, "wheelSHA256": WHEEL_SHA256, "files": files}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
