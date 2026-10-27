import AetherCoreAI
import AetherFlux2
import AetherFlux2CoreAI
import AetherFlux2Tokenizers
import AetherKit
import Darwin
import Foundation
import Testing

/// Opt-in native parity of the FLUX.2 [klein] 4B bundle against independent float32 Diffusers references on CPU
/// (`Scripts/prepare-flux2-reference-fixtures.py`: `--text-to-image` for the `diffusion` fixture, the default mode for
/// reference images). Loads every stage on the GPU: run only under the shared GPU lock, one native suite at
/// a time. Bars are the prototype's T2 continuous-output bars for the int8 export (11 §B6):
/// - tokenizer ids and mask: exact;
/// - text rows: real-row cosine ≥ 0.999 (padding rows excluded: they are ill-conditioned in every precision);
/// - velocity: cosine ≥ 0.999 per step, teacher-forced on the reference latents and text rows;
/// - VAE decoder: PSNR ≥ 35 dB on the reference final latent; reference encoder: latent cosine ≥ 0.999;
/// - image path (reference noise and text rows through this package's session): SSIM ≥ 0.94 (`quantized-8bit`);
/// - whole pipeline (tokenizer and native text encoder too): SSIM recorded, not gated (the padding rows move the
///   image more than conversion error; prototype F-P3-60).
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_FLUX2_QUALIFICATION"] == "1"
               && nativeEnvironment["AETHER_FLUX2_BUNDLE"] != nil && nativeEnvironment["AETHER_FLUX2_FIXTURES"] != nil
               && nativeEnvironment["AETHER_FLUX2_OUTPUT"] != nil),
      .timeLimit(.minutes(90)))
func flux2NativeParityAgainstIndependentReferences() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let env = nativeEnvironment
    let output = URL(fileURLWithPath: try #require(env["AETHER_FLUX2_OUTPUT"]))
    #expect(!FileManager.default.fileExists(atPath: output.path), "output directory must be new")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let report = Flux2Report(url: output.appendingPathComponent("report.json"))
    let started = Date()
    let assets = try Flux2Assets.open(at: URL(fileURLWithPath: try #require(env["AETHER_FLUX2_BUNDLE"])))
    try await report.time("verifyBundle") { try assets.verify() }
    report.set("bundle", ["root": assets.root.path, "fingerprint": assets.fingerprint,
                                "referenceFingerprint": assets.referenceFingerprint ?? "", "revision": assets.sourceRevision])
    let t2i = try Flux2Fixture(URL(fileURLWithPath: try #require(env["AETHER_FLUX2_FIXTURES"])))
    let selected = Set((env["AETHER_FLUX2_CASES"] ?? "lighthouse-512,red-circle-512,fox-512,ramen-576x448,cafe-512")
        .split(separator: ",").map(String.init))
    var cases = t2i.cases.filter { selected.contains("all") || selected.contains($0.id) }
    if let path = env["AETHER_FLUX2_REFERENCE_FIXTURES"] { cases += try Flux2Fixture(URL(fileURLWithPath: path)).cases }
    #expect(!cases.isEmpty)
    report.set("fixtures", ["t2i": t2i.id, "cases": cases.map(\.id).joined(separator: ",")])
    let compute = ModelComputePolicy.preferGPU

    // 1. Tokenizer and text encoder (component).
    let tokenizer = try await Flux2Tokenizer.open(assets: assets)
    try await report.phase("textEncoder") {
        let model = try await ModelSession.open(at: assets.textEncoder.url, compute: compute, expectFrequentReshapes: true)
        for c in cases {
            let prompt = try tokenizer.prepare(c.prompt)
            report.check("tokenIDs", c.id, value: prompt.tokenIDs == c.ids && prompt.attentionMask == c.mask ? 1 : 0, bar: 1)
            let replay = try Flux2Prompt(paddedTokenIDs: c.ids, attentionMask: c.mask)
            let begin = Date()
            let result = try await model.runTensors(function: "encode", inputs: replay.encoderInputs)
            report.timing("textEncode", c.id, Date().timeIntervalSince(begin))
            guard case .float16(let rows) = result["text_embeds"] else { throw AetherError.invalidTensor("text_embeds") }
            let n = replay.realTokens * Flux2Conditioning.width
            report.check("textRealRowCosine", c.id, value: cosine(rows.values[0..<n].map(Float.init), c.text[0..<n]), bar: 0.999)
        }
        await model.unload()
    }

    // 2. Teacher-forced denoiser velocity (component), text-only and with reference tokens.
    try await report.phase("denoiser") {
        var segments: [ModelSession] = []
        for asset in assets.denoisers {
            segments.append(try await ModelSession.open(at: asset.url, compute: compute, expectFrequentReshapes: true))
        }
        for c in cases {
            let text = try Float16Tensor(shape: [512, Flux2Conditioning.width], values: c.text.map(Float16.init))
            let ropeText = try Flux2Rotary.table(ids: Flux2Rotary.textIDs(length: 512))
            var ids = Flux2Rotary.gridIDs(rows: c.height / 16, columns: c.width / 16)
            for (k, r) in c.references.enumerated() { ids += Flux2Rotary.gridIDs(rows: r.height / 16, columns: r.width / 16, t: Int32(10 * (k + 1))) }
            if let positions = c.positions {
                report.check("referencePositions", c.id, value: positions == ids ? 1 : 0, bar: 1)
            }
            let ropeImage = try Flux2Rotary.table(ids: ids)
            let schedule = try Flux2Schedule(steps: c.steps, imageTokens: c.tokens)
            report.check("timesteps", c.id, value: schedule.timesteps == c.timesteps ? 1 : 0, bar: 1)
            for step in 0..<c.steps {
                let latents = try Float16Tensor(shape: [c.tokens + c.referenceTokens, 128],
                                                values: (c.stepLatents[step] + c.referenceLatents).map(Float16.init))
                let begin = Date()
                let velocity = try await denoise(segments, latents: latents, text: text, sigma: c.timesteps[step],
                                                 ropeText: ropeText, ropeImage: ropeImage)
                report.timing("denoiseStep", c.id, Date().timeIntervalSince(begin))
                report.check("velocityCosine[\(step)]", c.id,
                                   value: cosine(velocity.values[0..<(c.tokens * 128)].map(Float.init), c.stepVelocities[step][...]), bar: 0.999)
            }
        }
        for model in segments { await model.unload() }
    }

    // 3. VAE decoder on the reference final latent, and the reference encoder (components).
    try await report.phase("vae") {
        let decoder = try await ModelSession.open(at: assets.decoder.url, compute: compute, expectFrequentReshapes: false)
        for c in cases {
            let latents = try Float16Tensor(shape: [c.height / 16, c.width / 16, 128], values: c.final.map(Float16.init))
            let begin = Date()
            let result = try await decoder.runTensors(function: "decode", inputs: ["latents": .float16(latents)])
            report.timing("decode", c.id, Date().timeIntervalSince(begin))
            guard case .float32(let pixels) = result["image"] else { throw AetherError.invalidTensor("image") }
            report.check("decoderPSNR", c.id, value: psnr(unitHWC(pixels), c.image), bar: 35)
        }
        await decoder.unload()
        if let encoderAsset = assets.referenceEncoder, cases.contains(where: { !$0.references.isEmpty }) {
            let encoder = try await ModelSession.open(at: encoderAsset.url, compute: compute, expectFrequentReshapes: false)
            for c in cases where !c.references.isEmpty {
                var offset = 0
                for (k, r) in c.references.enumerated() {
                    let begin = Date()
                    let result = try await encoder.runTensors(function: "encode", inputs: [
                        "reference_pixels": .float32(try FloatTensor(shape: [3, r.height, r.width], values: r.pixels))])
                    report.timing("referenceEncode", c.id, Date().timeIntervalSince(begin))
                    guard case .float32(let latents) = result["reference_latents"] else { throw AetherError.invalidTensor("reference_latents") }
                    let count = (r.height / 16) * (r.width / 16) * 128
                    report.check("referenceLatentCosine[\(k)]", c.id,
                                       value: cosine(latents.values, c.referenceLatents[offset..<(offset + count)]), bar: 0.999)
                    offset += count
                }
            }
            await encoder.unload()
        }
    }

    // 4. Image path and whole pipeline through the public session.
    try await report.phase("session") {
        let session = Flux2GenerationSession(assets: assets, compute: compute)
        for c in cases {
            let configuration = try Flux2Configuration(width: c.width, height: c.height, steps: c.steps, seed: UInt64(c.seed))
            let noise = try FloatTensor(shape: [c.tokens, 128], values: c.noise)
            var references: Flux2PreparedReferences?
            if !c.references.isEmpty {
                let begin = Date()
                references = try await session.prepare(references: c.references.map {
                    try Flux2ReferenceImage(pixels: FloatTensor(shape: [3, $0.height, $0.width], values: $0.pixels))
                }, generatedTokens: c.tokens)
                report.timing("prepareReferences", c.id, Date().timeIntervalSince(begin))
            }
            var begin = Date()
            let image = try await session.generate(configuration: configuration,
                                                   conditioning: try Flux2Conditioning(provided: FloatTensor(shape: [512, 7680], values: c.text)),
                                                   references: references, noise: noise)
            report.timing("generateProvided", c.id, Date().timeIntervalSince(begin))
            try image.savePNG(to: output.appendingPathComponent("\(c.id)-provided.png"))
            report.check("imagePathSSIM", c.id, value: ssim(unitHWC(image.pixelsCHW), c.image, width: c.width, height: c.height), bar: 0.94)
            report.record("imagePathPSNR", c.id, psnr(unitHWC(image.pixelsCHW), c.image))
            begin = Date()
            let whole = try await session.generate(configuration: configuration, prompt: tokenizer.prepare(c.prompt),
                                                   references: c.references.map {
                                                       try Flux2ReferenceImage(pixels: FloatTensor(shape: [3, $0.height, $0.width], values: $0.pixels))
                                                   }, noise: noise)
            report.timing("generateFromPrompt", c.id, Date().timeIntervalSince(begin))
            try whole.savePNG(to: output.appendingPathComponent("\(c.id)-prompt.png"))
            report.record("wholePipelineSSIM", c.id, ssim(unitHWC(whole.pixelsCHW), c.image, width: c.width, height: c.height))
        }
        await session.unload()
    }
    report.set("elapsedSeconds", String(format: "%.1f", Date().timeIntervalSince(started)))
    let failures = report.finish()
    #expect(failures.isEmpty, "\(failures)")
}

// MARK: - Helpers

private func denoise(_ segments: [ModelSession], latents: Float16Tensor, text: Float16Tensor, sigma: Float,
                     ropeText: FloatTensor, ropeImage: FloatTensor) async throws -> Float16Tensor {
    var values: [String: TensorValue] = [
        "latents": .float16(latents), "text_embeds": .float16(text), "sigma": .float32(try FloatTensor(shape: [1], values: [sigma])),
        "rope_text": .float32(ropeText), "rope_image": .float32(ropeImage),
    ]
    for model in segments {
        guard let function = model.functions.first(where: { $0.name == "segment" }) else { throw AetherError.invalidModel("segment") }
        var inputs: [String: TensorValue] = [:]
        for name in function.inputs.keys {
            guard let value = values[name] else { throw AetherError.invalidModel("missing segment input \(name)") }
            inputs[name] = value
        }
        for (name, value) in try await model.runTensors(function: "segment", inputs: inputs) { values[name] = value }
    }
    guard case .float16(let velocity) = values["velocity"] else { throw AetherError.invalidTensor("velocity") }
    return velocity
}

func cosine<A: Collection<Float>, B: Collection<Float>>(_ a: A, _ b: B) -> Double {
    var dot = 0.0, aa = 0.0, bb = 0.0
    for (x, y) in zip(a, b) { dot += Double(x) * Double(y); aa += Double(x) * Double(x); bb += Double(y) * Double(y) }
    return a.count == b.count ? dot / max((aa * bb).squareRoot(), 1e-30) : -1
}

/// `[3,H,W]` on [−1, 1] → `[H,W,3]` on [0, 1], clamped as the reference pipeline post-processes.
func unitHWC(_ pixels: FloatTensor) -> [Float] {
    let h = pixels.shape[1], w = pixels.shape[2], plane = h * w
    var out = [Float](repeating: 0, count: plane * 3)
    for p in 0..<plane { for c in 0..<3 { out[p * 3 + c] = min(max(pixels.values[c * plane + p] / 2 + 0.5, 0), 1) } }
    return out
}

func psnr(_ a: [Float], _ b: [Float]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return -1 }
    var sum = 0.0
    for (x, y) in zip(a, b) { let d = Double(min(max(x, 0), 1)) - Double(min(max(y, 0), 1)); sum += d * d }
    let mse = sum / Double(a.count)
    return mse == 0 ? 999 : 10 * log10(1 / mse)
}

/// SSIM on BT.601 luma: 11×11 Gaussian window (σ 1.5), K = 0.01/0.03, valid region only — the prototype's
/// `diffusion_fixtures.ssim` metric.
func ssim(_ a: [Float], _ b: [Float], width: Int, height: Int) -> Double {
    guard a.count == width * height * 3, b.count == a.count, width >= 11, height >= 11 else { return -1 }
    func luma(_ v: [Float]) -> [Double] {
        (0..<(width * height)).map { p in
            0.299 * Double(min(max(v[p * 3], 0), 1)) + 0.587 * Double(min(max(v[p * 3 + 1], 0), 1)) + 0.114 * Double(min(max(v[p * 3 + 2], 0), 1))
        }
    }
    let x = luma(a), y = luma(b)
    var g = (0..<11).map { exp(-pow(Double($0 - 5), 2) / (2 * 1.5 * 1.5)) }
    let total = g.reduce(0, +); g = g.map { $0 / total }
    func filter(_ v: [Double]) -> [Double] {
        let ow = width - 10, oh = height - 10
        var horizontal = [Double](repeating: 0, count: ow * height)
        for r in 0..<height { for c in 0..<ow { var s = 0.0; for k in 0..<11 { s += g[k] * v[r * width + c + k] }; horizontal[r * ow + c] = s } }
        var out = [Double](repeating: 0, count: ow * oh)
        for r in 0..<oh { for c in 0..<ow { var s = 0.0; for k in 0..<11 { s += g[k] * horizontal[(r + k) * ow + c] }; out[r * ow + c] = s } }
        return out
    }
    let mx = filter(x), my = filter(y)
    let sxx = filter(x.map { $0 * $0 }), syy = filter(y.map { $0 * $0 }), sxy = filter(zip(x, y).map { $0 * $1 })
    let c1 = 0.0001, c2 = 0.0009
    var sum = 0.0
    for i in mx.indices {
        let vx = sxx[i] - mx[i] * mx[i], vy = syy[i] - my[i] * my[i], cv = sxy[i] - mx[i] * my[i]
        sum += ((2 * mx[i] * my[i] + c1) * (2 * cv + c2)) / ((mx[i] * mx[i] + my[i] * my[i] + c1) * (vx + vy + c2))
    }
    return sum / Double(mx.count)
}

/// Reads Forge `diffusion` fixtures and the reference-image fixtures (same layout plus reference files).
struct Flux2Fixture {
    struct Reference { let width: Int; let height: Int; let pixels: [Float] }
    struct Case {
        let id: String, prompt: String, width: Int, height: Int, seed: Int, steps: Int
        let ids: [Int32], mask: [Int32], text: [Float], noise: [Float], final: [Float], image: [Float]
        let timesteps: [Float], stepLatents: [[Float]], stepVelocities: [[Float]]
        let references: [Reference], referenceLatents: [Float], positions: [Int32]?
        var tokens: Int { (width / 16) * (height / 16) }
        var referenceTokens: Int { references.reduce(0) { $0 + ($1.width / 16) * ($1.height / 16) } }
    }
    let id: String
    let cases: [Case]

    init(_ root: URL) throws {
        struct Document: Decodable { let id: String; let sha256: [String: String] }
        struct Line: Decodable {
            struct Files: Decodable {
                let ids: String, mask: String, text: String, noise: String, final: String, image: String
                let stepLatents: [String], stepVelocities: [String]
                let referencePixels: [String]?, referenceLatents: String?, positions: String?
            }
            struct Ref: Decodable { let width: Int; let height: Int }
            let id: String, prompt: String, width: Int, height: Int, seed: Int, steps: Int
            let timesteps: [Double], files: Files, references: [Ref]?
        }
        let document = try JSONDecoder().decode(Document.self, from: Data(contentsOf: root.appendingPathComponent("fixture.json")))
        id = document.id
        func data(_ name: String) throws -> Data {
            let bytes = try Data(contentsOf: root.appendingPathComponent(name))
            guard let expected = document.sha256[name], Flux2Fixture.hex(bytes) == expected else {
                throw AetherError.invalidRequest("fixture file hash mismatch: \(name)")
            }
            return bytes
        }
        func floats(_ name: String) throws -> [Float] {
            try data(name).withUnsafeBytes { raw in (0..<(raw.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) } }
        }
        func ints(_ name: String) throws -> [Int32] {
            try data(name).withUnsafeBytes { raw in (0..<(raw.count / 4)).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) } }
        }
        let lines = try String(decoding: data("cases.jsonl"), as: UTF8.self).split(separator: "\n")
        cases = try lines.map { line in
            let l = try JSONDecoder().decode(Line.self, from: Data(line.utf8))
            let refs = try zip(l.references ?? [], l.files.referencePixels ?? []).map {
                Reference(width: $0.width, height: $0.height, pixels: try floats($1))
            }
            return Case(id: l.id, prompt: l.prompt, width: l.width, height: l.height, seed: l.seed, steps: l.steps,
                        ids: try ints(l.files.ids), mask: try ints(l.files.mask), text: try floats(l.files.text),
                        noise: try floats(l.files.noise), final: try floats(l.files.final), image: try floats(l.files.image),
                        timesteps: l.timesteps.map(Float.init), stepLatents: try l.files.stepLatents.map(floats),
                        stepVelocities: try l.files.stepVelocities.map(floats), references: refs,
                        referenceLatents: try l.files.referenceLatents.map(floats) ?? [],
                        positions: try l.files.positions.map(ints))
        }
    }

    static func hex(_ data: Data) -> String {
        var hasher = SHA256Stream(); hasher.update(data); return hasher.hex
    }
}

import CryptoKit
private struct SHA256Stream {
    var digest = SHA256()
    mutating func update(_ data: Data) { digest.update(data: data) }
    var hex: String { digest.finalize().map { String(format: "%02x", $0) }.joined() }
}

/// Incrementally persisted qualification record: checks with bars, timings, memory per phase.
final class Flux2Report: @unchecked Sendable {
    let url: URL
    private let lock = NSLock()
    private var document: [String: Any] = ["schema": 1, "kind": "flux2-native-parity", "checks": [[String: Any]](),
                                           "records": [[String: Any]](), "timings": [[String: Any]](), "phases": [[String: Any]]()]
    init(url: URL) { self.url = url }

    func set(_ key: String, _ value: Any) { lock.withLock { document[key] = value; save() } }
    func check(_ name: String, _ id: String, value: Double, bar: Double) {
        let passed = value >= bar
        print("flux2 check \(name) \(id): \(value) (bar \(bar)) \(passed ? "pass" : "FAIL")")
        append("checks", ["name": name, "case": id, "value": value, "bar": bar, "passed": passed])
    }
    func record(_ name: String, _ id: String, _ value: Double) {
        print("flux2 record \(name) \(id): \(value)")
        append("records", ["name": name, "case": id, "value": value])
    }
    func timing(_ name: String, _ id: String, _ seconds: Double) { append("timings", ["name": name, "case": id, "seconds": seconds]) }
    func time(_ name: String, _ body: () async throws -> Void) async throws {
        let begin = Date(); try await body(); append("timings", ["name": name, "seconds": Date().timeIntervalSince(begin)])
    }
    func phase(_ name: String, _ body: () async throws -> Void) async throws {
        let begin = Date()
        do { try await body() } catch {
            append("phases", ["name": name, "error": "\(error)"]); throw error
        }
        let memory = Self.memory()
        append("phases", ["name": name, "seconds": Date().timeIntervalSince(begin), "footprintBytesAfter": memory.0,
                          "lifetimePeakFootprintBytes": memory.1])
    }
    func finish() -> [String] {
        lock.lock(); defer { lock.unlock() }
        let checks = document["checks"] as! [[String: Any]]
        let failures = checks.filter { $0["passed"] as? Bool != true }.map { "\($0["name"]!) \($0["case"]!)" }
        document["passed"] = failures.isEmpty && !checks.isEmpty
        document["checkCount"] = checks.count
        save()
        return failures
    }
    private func append(_ key: String, _ value: [String: Any]) {
        lock.withLock { var list = document[key] as! [[String: Any]]; list.append(value); document[key] = list; save() }
    }
    private func save() {
        guard let data = try? JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys]) else { return }
        try? data.write(to: url, options: .atomic)
    }
    static func memory() -> (UInt64, UInt64) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        guard result == KERN_SUCCESS else { return (0, 0) }
        return (info.phys_footprint, UInt64(max(0, info.ledger_phys_footprint_peak)))
    }
}
