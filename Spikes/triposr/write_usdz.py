"""Author a USD stage for a TripoSR mesh directly (no OBJ round trip) and package it as USDZ.

usage: write_usdz.py <mesh.npz> <out.usdz> [--packager python|usdzip]
The stage has the mesh's points, faces and per-vertex colours as a `displayColor` primvar, a UsdPreviewSurface material
reading that primvar, `metersPerUnit` and `upAxis`, and an extent. It's written as usda, converted to usdc with
/usr/bin/usdcat, then zipped:
  - python (default): the kit's own route, since usdzip doesn't exist on iOS: a stored (uncompressed) zip whose
    file data starts on 64-byte boundaries, padded through each local header's extra field, as the USDZ spec requires;
  - usdzip: /usr/bin/usdzip, for comparison.
Validate with `usdchecker --arkit <out.usdz>`.
"""
import argparse
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np


def stage_text(vertices, faces, colors):
    def floats(rows):
        return ", ".join("(" + ", ".join(f"{x:.6g}" for x in row) + ")" for row in rows)
    lo, hi = vertices.min(0), vertices.max(0)
    return f"""#usda 1.0
(
    defaultPrim = "Object"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Object" (
    kind = "component"
)
{{
    def Mesh "Mesh" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        float3[] extent = [{floats([lo, hi])}]
        int[] faceVertexCounts = [{", ".join(["3"] * len(faces))}]
        int[] faceVertexIndices = [{", ".join(str(int(i)) for i in faces.reshape(-1))}]
        rel material:binding = </Object/Material>
        point3f[] points = [{floats(vertices)}]
        color3f[] primvars:displayColor = [{floats(colors)}] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
    }}

    def Material "Material"
    {{
        token outputs:surface.connect = </Object/Material/Surface.outputs:surface>

        def Shader "Surface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Object/Material/VertexColor.outputs:result>
            float inputs:metallic = 0
            float inputs:roughness = 0.8
            token outputs:surface
        }}

        def Shader "VertexColor"
        {{
            uniform token info:id = "UsdPrimvarReader_float3"
            float3 inputs:fallback = (0.5, 0.5, 0.5)
            string inputs:varname = "displayColor"
            float3 outputs:result
        }}
    }}
}}
"""


def write_aligned_zip(path: Path, name: str, data: bytes, alignment: int = 64):
    """One stored entry whose data begins on an `alignment`-byte boundary (USDZ's packaging rule)."""
    encoded = name.encode()
    crc = zlib.crc32(data) & 0xFFFFFFFF
    header_size = 30 + len(encoded)
    pad = (-(header_size + 4)) % alignment                 # 4 bytes of extra-field header precede the padding
    extra = struct.pack("<HH", 0x1986, pad) + b"\0" * pad  # an unregistered extra-field ID, as usdzip's padding uses
    local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, 0, 0, crc, len(data), len(data), len(encoded),
                        len(extra)) + encoded + extra
    assert len(local) % alignment == 0
    central = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 0, 0, 0, crc, len(data), len(data),
                          len(encoded), 0, 0, 0, 0, 0, 0) + encoded
    offset = len(local) + len(data)
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), offset, 0)
    path.write_bytes(local + data + central + end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--packager", choices=["python", "usdzip"], default="python")
    a = ap.parse_args()
    mesh = np.load(a.mesh)
    colors = mesh["colors"] if "colors" in mesh.files else np.full_like(mesh["vertices"], 0.5)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        usda, usdc = Path(tmp) / "model.usda", Path(tmp) / "model.usdc"
        usda.write_text(stage_text(mesh["vertices"], mesh["faces"], colors.clip(0, 1)))
        subprocess.run(["/usr/bin/usdcat", str(usda), "-o", str(usdc)], check=True)
        if a.packager == "usdzip":
            subprocess.run(["/usr/bin/usdzip", str(Path(a.out).resolve()), "model.usdc"], check=True, cwd=tmp)
        else:
            write_aligned_zip(Path(a.out), "model.usdc", usdc.read_bytes())
    print(f"{a.out}: {Path(a.out).stat().st_size} bytes in {time.perf_counter() - t0:.2f}s ({a.packager})")


if __name__ == "__main__":
    main()
