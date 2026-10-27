// usage: swift obj2usdz_scenekit.swift <in.obj> <out.usdz>
// SceneKit can write USDZ on macOS and iOS (Model I/O on this OS exports only .usda/.usdc).
import Foundation
import ModelIO
import SceneKit
import SceneKit.ModelIO

let args = CommandLine.arguments
let t0 = Date()
let asset = MDLAsset(url: URL(fileURLWithPath: args[1]))
asset.upAxis = SIMD3<Float>(0, 1, 0)
let scene = SCNScene(mdlAsset: asset)
let ok = scene.write(to: URL(fileURLWithPath: args[2]), options: nil, delegate: nil, progressHandler: nil)
let size = (try? FileManager.default.attributesOfItem(atPath: args[2])[.size] as? Int) ?? -1
print(String(format: "SceneKit write ok=%@ size=%d bytes in %.2fs", ok ? "true" : "false", size, Date().timeIntervalSince(t0)))
