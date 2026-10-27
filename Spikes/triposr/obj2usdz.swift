// usage: swift obj2usdz.swift <in.obj> <out.usdz>
// Checks whether Model I/O on this OS can export USDZ directly from a TripoSR OBJ (vertex colors included).
import Foundation
import ModelIO

let args = CommandLine.arguments
for ext in ["usdz", "usdc", "usda", "obj"] {
    print("canExport \(ext):", MDLAsset.canExportFileExtension(ext), "canImport:", MDLAsset.canImportFileExtension(ext))
}
let t0 = Date()
let asset = MDLAsset(url: URL(fileURLWithPath: args[1]))
let mesh = asset.object(at: 0) as? MDLMesh
print("objects:", asset.count, "vertices:", mesh?.vertexCount ?? -1,
      "attributes:", mesh?.vertexDescriptor.attributes.compactMap { ($0 as? MDLVertexAttribute)?.name }.filter { !$0.isEmpty } ?? [])
do {
    try asset.export(to: URL(fileURLWithPath: args[2]))
    let size = (try? FileManager.default.attributesOfItem(atPath: args[2])[.size] as? Int) ?? -1
    print(String(format: "exported %@ (%d bytes) in %.2fs", args[2], size, Date().timeIntervalSince(t0)))
} catch {
    print("export failed:", error)
}
