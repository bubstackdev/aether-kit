// Isolates the Swift image path of a Qwen3.5 bundle against the upstream image reference (reference_qwen35_image.py):
// the encoder on the reference's own pixels and on Kit's preprocessing of the raw image, then the hybrid decoder's
// teacher-forced rows with each set of features.
//
// usage: q35-kit-check <bundle> <image reference dir>
import AetherKit
import AetherLanguageCoreAI
import AetherVision
import Foundation

struct Failure: Error { let message: String }
let args = CommandLine.arguments
let bundle = URL(fileURLWithPath: args[1]), refDir = URL(fileURLWithPath: args[2])
let ref = try JSONSerialization.jsonObject(with: Data(contentsOf: refDir.appendingPathComponent("reference.json"))) as! [String: Any]
let c = (ref["cases"] as! [[String: Any]])[0]
func floats(_ file: String) throws -> [Float] {
    let data = try Data(contentsOf: refDir.appendingPathComponent(file))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
func cosine(_ a: [Float], _ b: [Float]) -> Double {
    var ab = 0.0, aa = 0.0, bb = 0.0
    for i in 0..<min(a.count, b.count) { ab += Double(a[i]) * Double(b[i]); aa += Double(a[i] * a[i]); bb += Double(b[i] * b[i]) }
    return ab / (aa.squareRoot() * bb.squareRoot())
}
let manifest = try LanguageModelManifest.load(from: bundle)
let vision = manifest.vision!
let grid = (c["grid"] as! [Int])
let configuration = LanguageVisionEncoder.Configuration(mergeSize: vision.mergeSize, positionTableSide: vision.positionTableSide,
    imageGrids: (vision.imageGrids ?? []).map { Qwen3VLImagePreprocessor.Grid(height: $0.height, width: $0.width) })
let encoder = try await LanguageVisionEncoder.open(at: bundle.appendingPathComponent("vision.aimodel"), configuration: configuration)
let refPixels = try floats((c["pixels"] as! [String: Any])["file"] as! String).map { Float16($0) }
let refEmbeds = try floats((c["imageEmbeds"] as! [String: Any])["file"] as! String)
let fromRef = try await encoder.encode(pixels: refPixels, grid: (1, grid[1], grid[2]))
let refFeatures: [Float] = fromRef.embeddings.map { Float($0) }
print("encoder on reference pixels: cosine", cosine(refFeatures, refEmbeds))
let image = try VisionRGBImage(contentsOf: refDir.appendingPathComponent((c["image"] as! [String: Any])["file"] as! String))
let processed = try Qwen3VLImagePreprocessor.process(rgb: image.rgb, width: image.width, height: image.height,
                                                     configuration: VisionLanguageSession.preprocessing(for: vision))
let kitPixels: [Float16] = processed.pixels.map { Float16($0) }
let refAsFloat: [Float] = refPixels.map { Float($0) }
print("kit grid", processed.grid, "pixel cosine vs reference", cosine(processed.pixels.map { Float($0) }, refAsFloat))
let fromKit = try await encoder.encode(pixels: kitPixels, grid: processed.grid)
let kitFeatures: [Float] = fromKit.embeddings.map { Float($0) }
print("encoder on Kit pixels: cosine", cosine(kitFeatures, refEmbeds))
await encoder.unload()

let decoder = try await HybridLanguageSession.open(at: bundle.appendingPathComponent("model.aimodel"), registration: manifest.hybrid!,
    maximumContextLength: manifest.maximumContextLength, stopTokenIDs: Set(manifest.stopTokenIDs), imageTokenID: vision.imageTokenID)
let ids = c["promptTokens"] as! [Int], forced = c["forcedTokens"] as! [Int]
var rendered: [Int] = []
for id in ids where !(id == vision.imageTokenID && rendered.last == vision.imageTokenID) { rendered.append(id) }
let tokens = LanguageImageTokens(visionStart: vision.visionStartTokenID, imagePad: vision.imageTokenID, visionEnd: vision.visionEndTokenID)
let vocab = 248_320
let want = try floats((c["forcedLogits"] as! [String: Any])["file"] as! String)
for (label, features) in [("reference pixels", fromRef), ("Kit pixels", fromKit)] {
    let rows = try await decoder.teacherForcedLogits(prompt: LanguageVisionPrompt(renderedTokenIDs: rendered, images: [features]),
                                                     imageTokens: tokens, continuation: Array(forced.dropLast()))
    var worst = 1.0, agree = 0
    func argmax(_ v: [Float]) -> Int { var b = 0; for i in 1..<v.count where v[i] > v[b] { b = i }; return b }
    for (r, row) in rows.enumerated() {
        let w = Array(want[(r * vocab)..<((r + 1) * vocab)])
        worst = min(worst, cosine(row, w))
        if argmax(row) == argmax(w) { agree += 1 } else {
            // Kit's near tie: reference top-1 minus top-2 probability < 0.1 and Kit's token in the reference top 5.
            let m = Double(w.max()!), e = w.map { exp(Double($0) - m) }, z = e.reduce(0, +)
            let order = e.indices.sorted { e[$0] > e[$1] }
            let margin = (e[order[0]] - e[order[1]]) / z
            print("  row \(r): Kit \(argmax(row)), reference \(argmax(w)), reference margin \(margin), near tie \(margin < 0.1 && order.prefix(5).contains(argmax(row)))")
        }
    }
    print("decoder forced rows (\(label)): min cosine \(worst), argmax \(agree)/\(rows.count)")
}
await decoder.unload()

// Kit's decode of the raw image, raw RGB bytes (width × height × 3), for a same-decode reference.
if args.count > 3 {
    try Data(image.rgb).write(to: URL(fileURLWithPath: args[3]))
    print("wrote Kit decode \(image.width)x\(image.height) to \(args[3])")
}
