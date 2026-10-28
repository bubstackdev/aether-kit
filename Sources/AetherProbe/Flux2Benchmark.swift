import AetherFlux2
import AetherFlux2CoreAI
import AetherFlux2Tokenizers
import AetherKit
import CoreAI
import Darwin
import Foundation
import Metal
import Synchronization

/// A pinned FLUX.2 [klein] benchmark plan (`Benchmarks/flux2-baseline.json`). Workloads come from the independent
/// CPU fixtures: each case names a fixture case, whose prompt, size, steps, initial noise, text rows and reference
/// images are used as recorded, so every iteration is comparable with the float32 Diffusers image.
@available(macOS 27, iOS 27, *)
struct Flux2BenchmarkPlan: Codable, Sendable {
    struct Case: Codable, Sendable {
        let id: String
        /// "t2i" or "reference": which fixture the case comes from.
        let fixture: String
        let fixtureCase: String
    }
    let schema: Int
    let bundleFingerprint: String
    let textToImageFixture: String
    let referenceFixture: String?
    let warmupIterations: Int
    let measuredIterations: Int
    /// encoded: every iteration encodes the prompt (and references) through the session. provided: the fixture's
    /// text rows, and references prepared once per case outside iteration timing.
    let profile: BenchmarkProfile
    let residency: BenchmarkResidency
    /// Gate on the first measured iteration of a provided-profile case: SSIM against the fixture image.
    let minimumProvidedSSIM: Double
    let cases: [Case]

    func validate() throws {
        guard schema == 1, benchmarkIsHash(bundleFingerprint), (1...10).contains(warmupIterations),
              (3...100).contains(measuredIterations), (1...16).contains(cases.count), Set(cases.map(\.id)).count == cases.count,
              (0...1).contains(minimumProvidedSSIM), !textToImageFixture.isEmpty else {
            throw AetherError.invalidRequest("FLUX.2 benchmark needs schema 1, unique cases, 1...10 warmups, 3...100 measurements and a pinned bundle")
        }
        for item in cases {
            guard !item.id.isEmpty, !item.fixtureCase.isEmpty, ["t2i", "reference"].contains(item.fixture),
                  item.fixture == "t2i" || referenceFixture != nil else {
                throw AetherError.invalidRequest("FLUX.2 benchmark case \(item.id) needs a fixture case from a pinned fixture")
            }
        }
    }
}

/// One fixture case as the benchmark needs it; every file is checked against the fixture document.
@available(macOS 27, iOS 27, *)
struct Flux2BenchmarkWorkload: Sendable {
    let id: String
    let fixtureCase: String
    let prompt: String
    let width: Int
    let height: Int
    let seed: UInt64
    let steps: Int
    let tokenIDs: [Int32]
    let noise: FloatTensor
    let text: FloatTensor
    /// `[H, W, 3]` on [0, 1], the float32 Diffusers image.
    let image: [Float]
    let references: [Flux2ReferenceImage]
    var imageTokens: Int { (width / 16) * (height / 16) }

    /// Reads `cases.jsonl` and the files a case names from a Forge-format fixture directory.
    static func load(fixture root: URL, expectedID: String, caseID: String, benchmarkID: String) throws -> Self {
        struct Document: Decodable { let id: String; let sha256: [String: String] }
        struct Line: Decodable {
            struct Files: Decodable { let ids: String; let text: String; let noise: String; let image: String; let referencePixels: [String]? }
            struct Ref: Decodable { let width: Int; let height: Int }
            let id: String, prompt: String, width: Int, height: Int, seed: Int, steps: Int
            let files: Files, references: [Ref]?
        }
        let document = try JSONDecoder().decode(Document.self, from: Data(contentsOf: root.appendingPathComponent("fixture.json")))
        guard document.id == expectedID else {
            throw AetherError.invalidRequest("fixture \(root.lastPathComponent) is \(document.id), the plan pins \(expectedID)")
        }
        func data(_ name: String) throws -> Data {
            let bytes = try Data(contentsOf: root.appendingPathComponent(name))
            guard document.sha256[name] == benchmarkHash(bytes) else { throw AetherError.invalidRequest("fixture file hash mismatch: \(name)") }
            return bytes
        }
        func floats(_ name: String) throws -> [Float] {
            try data(name).withUnsafeBytes { raw in
                (0..<(raw.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
            }
        }
        let lines = String(decoding: try data("cases.jsonl"), as: UTF8.self).split(separator: "\n")
        guard let line = try lines.map({ try JSONDecoder().decode(Line.self, from: Data($0.utf8)) }).first(where: { $0.id == caseID }) else {
            throw AetherError.invalidRequest("fixture \(expectedID) has no case \(caseID)")
        }
        let ids = try data(line.files.ids).withUnsafeBytes { raw in
            (0..<(raw.count / 4)).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) }
        }
        let tokens = (line.width / 16) * (line.height / 16)
        let references = try zip(line.references ?? [], line.files.referencePixels ?? []).map {
            try Flux2ReferenceImage(pixels: FloatTensor(shape: [3, $0.height, $0.width], values: try floats($1)))
        }
        return Self(id: benchmarkID, fixtureCase: caseID, prompt: line.prompt, width: line.width, height: line.height,
                    seed: UInt64(line.seed), steps: line.steps, tokenIDs: ids,
                    noise: try FloatTensor(shape: [tokens, 128], values: try floats(line.files.noise)),
                    text: try FloatTensor(shape: [Flux2Prompt.length, Flux2Conditioning.width], values: try floats(line.files.text)),
                    image: try floats(line.files.image), references: references)
    }
}

/// Per-iteration stage clock over `Flux2GenerationEvent`s. Observer time (hashing, checks) is excluded from stages and
/// included in the total, as in the Krea harness.
@available(macOS 27, iOS 27, *)
final class Flux2BenchmarkJournal: Sendable {
    struct State: Sendable {
        var row: BenchmarkIteration
        var anchor: ContinuousClock.Instant
        var phase = "conditioning"
        var nextStep = 0
    }
    private let state: Mutex<State>
    private let steps: Int
    private let imageTokens: Int
    private let referenceTokens: Int
    private let memory: BenchmarkMemorySampler

    init(caseID: String, role: String, iteration: Int, profile: BenchmarkProfile, steps: Int, imageTokens: Int,
         referenceTokens: Int, memory: BenchmarkMemorySampler, started: ContinuousClock.Instant) {
        var row = BenchmarkIteration(caseID: caseID, role: role, iteration: iteration)
        row.profile = profile
        state = Mutex(State(row: row, anchor: started))
        self.steps = steps; self.imageTokens = imageTokens; self.referenceTokens = referenceTokens; self.memory = memory
    }
    var row: BenchmarkIteration { state.withLock { $0.row } }

    func accept(_ event: Flux2GenerationEvent) throws {
        let entered = ContinuousClock.now
        try state.withLock { value in
            defer {
                let ended = ContinuousClock.now
                value.row.observerSeconds += AetherProbe.seconds(entered.duration(to: ended))
                value.anchor = ended
            }
            let elapsed = AetherProbe.seconds(value.anchor.duration(to: entered))
            switch event {
            case .conditioningReady(let conditioning):
                guard value.phase == "conditioning" else { throw AetherError.invalidTensor("benchmark conditioning order") }
                value.row.stages["conditioning"] = elapsed
                value.row.conditioningSHA256 = floatHash(conditioning.tensor.values.map(Float.init))
                value.phase = referenceTokens > 0 ? "references" : "denoiserSetup"
            case .referencesReady(let prepared):
                guard value.phase == "references", prepared.tokens == referenceTokens else {
                    throw AetherError.invalidTensor("benchmark reference order")
                }
                value.row.stages["references"] = elapsed
                value.phase = "denoiserSetup"
            case .denoisersReady(let tokens):
                guard value.phase == "denoiserSetup", tokens == imageTokens + referenceTokens else {
                    throw AetherError.invalidTensor("benchmark denoiser setup")
                }
                value.row.stages["denoiserSetup"] = elapsed
                value.phase = "step"
            case .step(let index, let total, let latents):
                guard value.phase == "step", index == value.nextStep, total == steps, latents.shape == [imageTokens, 128] else {
                    throw AetherError.invalidTensor("benchmark sampling order")
                }
                value.row.stages["step\(index)"] = elapsed
                value.row.stepLatentsSHA256.append(floatHash(latents.values))
                value.nextStep += 1
                if value.nextStep == steps { value.phase = "decode" }
            case .decoded:
                guard value.phase == "decode" else { throw AetherError.invalidTensor("benchmark decode order") }
                value.row.stages["decode"] = elapsed
                value.phase = "completion"
            }
            memory.sample()
        }
    }

    /// Completes the row; `expectedPixels` (the case's first iteration) must match exactly: generation is deterministic.
    func finish(_ image: GeneratedImage, ended: ContinuousClock.Instant, total: Double, expectedPixels: String?) throws -> String {
        try state.withLock { value in
            value.row.totalSeconds = total
            guard value.phase == "completion" else { throw AetherError.invalidTensor("benchmark generation incomplete") }
            value.row.stages["completion"] = AetherProbe.seconds(value.anchor.duration(to: ended))
            let hash = floatHash(image.pixelsCHW.values)
            value.row.pixelsSHA256 = hash
            if let expectedPixels, expectedPixels != hash { throw AetherError.invalidTensor("benchmark pixels differ between iterations") }
            let sample = memory.snapshot
            value.row.memorySamples = sample.samples
            value.row.sampledPeakMemory = sample.peak
            guard sample.samples > 0, sample.error == nil else {
                throw AetherError.unsupported("benchmark memory sampling failed: \(sample.error ?? "no samples")")
            }
            value.row.passed = true
            return hash
        }
    }

    func fail(_ error: any Error, total: Double) -> BenchmarkIteration {
        state.withLock { value in
            value.row.passed = false
            value.row.error = String(describing: error)
            value.row.totalSeconds = total
            let sample = memory.snapshot
            value.row.memorySamples = sample.samples
            value.row.sampledPeakMemory = sample.peak
            return value.row
        }
    }
}

@available(macOS 27, iOS 27, *)
extension BenchmarkStatistics {
    /// FLUX.2 measured-iteration statistics: every stage, total, observer time, pipeline time and the summed steps.
    static func flux2(_ rows: [BenchmarkIteration], caseID: String, steps: Int) -> [String: Self] {
        let rows = rows.filter { $0.caseID == caseID && $0.role == "measured" && $0.passed }
        guard !rows.isEmpty else { return [:] }
        var metrics: [String: [Double]] = [:]
        for row in rows {
            if let total = row.totalSeconds { metrics["totalIncludingObservers", default: []].append(total) }
            metrics["observer", default: []].append(row.observerSeconds)
            for (name, duration) in row.stages { metrics[name, default: []].append(duration) }
            metrics["pipelineExcludingObservers", default: []].append(row.stages.values.reduce(0, +))
            metrics["sampling", default: []].append((0..<steps).reduce(0) { $0 + (row.stages["step\($1)"] ?? 0) })
        }
        return metrics.compactMapValues(summarize)
    }
}

/// SSIM on BT.601 luma of two `[H,W,3]` images on [0, 1]: 11×11 Gaussian window (σ 1.5), K = 0.01/0.03, valid region
/// only. The same metric as the native parity suite's (the prototype's `diffusion_fixtures.ssim`).
@available(macOS 27, iOS 27, *)
func flux2SSIM(_ a: [Float], _ b: [Float], width: Int, height: Int) -> Double {
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

/// `[3,H,W]` on [−1, 1] → `[H,W,3]` on [0, 1], clamped as the reference pipeline post-processes.
@available(macOS 27, iOS 27, *)
func flux2UnitHWC(_ pixels: FloatTensor) -> [Float] {
    let h = pixels.shape[1], w = pixels.shape[2], plane = h * w
    var out = [Float](repeating: 0, count: plane * 3)
    for p in 0..<plane { for c in 0..<3 { out[p * 3 + c] = min(max(pixels.values[c * plane + p] / 2 + 0.5, 0), 1) } }
    return out
}
