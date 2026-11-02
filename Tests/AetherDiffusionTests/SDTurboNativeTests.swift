import AetherCoreAI
import AetherDiffusion
import AetherDiffusionCoreAI
import AetherSDTurbo
import AetherKit
import Foundation
import Testing

private struct Fixture: Decodable { let id: String; let prompt: String; let steps: Int; let ids: [Int32]; let timesteps: [Float]; let sigmas: [Float]; let initialNoiseSigma: Float }
private let env = ProcessInfo.processInfo.environment

@Test(.enabled(if: env["AETHER_SDTURBO_QUALIFY"] == "1"), .timeLimit(.minutes(60)))
@available(macOS 27, iOS 27, *)
func sdTurboNativeParity() async throws {
    let root = URL(filePath: try #require(env["AETHER_SDTURBO_QUALIFICATION"]))
    let bundle = try SDTurboBundle(at: root.appendingPathComponent("bundle"))
    let tokenizer = try SDTurboTokenizer.open(bundle: bundle)
    struct TokenCase: Decodable { let prompt: String; let ids: [Int32] }
    let tokenCases = try JSONDecoder().decode([TokenCase].self, from: Data(contentsOf: root.appendingPathComponent("reference/tokenizer-cases.json")))
    for c in tokenCases { #expect(try tokenizer.prepare(c.prompt).values == c.ids) }
    let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: root.appendingPathComponent("reference/cases.json")))
    let lease = try await SharedGPULease.acquire(); defer { lease.release() }
    let backend = try CoreAIEpsilonBackend(textEncoder: bundle.root.appendingPathComponent("textEncoder.aimodel"),
        denoiser: bundle.root.appendingPathComponent("denoiser.aimodel"), decoder: bundle.root.appendingPathComponent("decoder.aimodel"),
        width: bundle.width, height: bundle.height, latentChannels: 4, latentScale: 8, textLength: 77, textWidth: 1024, vocabularySize: 49408)
    let session = DiffusionSession(backend: backend)
    var report: [[String: Any]] = []
    let latentShape = [1, 4, bundle.height / 8, bundle.width / 8]
    func read(_ dir: URL, _ name: String, _ shape: [Int]) throws -> FloatTensor {
        let data = try Data(contentsOf: dir.appendingPathComponent(name))
        let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return try FloatTensor(shape: shape, values: values)
    }
    func compare(_ a: [Float], _ b: [Float]) -> (cosine: Double, rmse: Double) {
        var dot = 0.0, aa = 0.0, bb = 0.0, se = 0.0
        for (x, y) in zip(a, b) { let x = Double(x), y = Double(y); dot += x*y; aa += x*x; bb += y*y; se += (x-y)*(x-y) }
        return (dot / sqrt(max(aa*bb, 1e-30)), sqrt(se / Double(a.count)))
    }
    for c in fixtures {
        let dir = root.appendingPathComponent("reference/\(c.id)")
        let ids = try tokenizer.prepare(c.prompt)
        #expect(ids.values == c.ids)
        let s = try bundle.schedule(steps: c.steps)
        #expect(s.timesteps == c.timesteps && s.sigmas == c.sigmas && s.initialNoiseSigma == c.initialNoiseSigma)
        let text = try read(dir, "text.f32", [1, 77, 1024])
        let nativeText = try await backend.encode(ids)
        let textMetric = compare(text.values, nativeText.values)
        #expect(textMetric.cosine >= 0.99999)
        for k in s.timesteps.indices {
            let x = try read(dir, "latent-\(k).f32", latentShape)
            let epsilon = try read(dir, "epsilon-\(k).f32", latentShape)
            let native = try await backend.denoise(sample: s.scaledInput(x, step: k), timestep: s.timesteps[k], conditioning: text)
            let metric = compare(epsilon.values, native.values)
            #expect(metric.cosine >= 0.9999)
            report.append(["case": c.id, "stage": "epsilon", "step": k, "cosine": metric.cosine, "rmse": metric.rmse])
        }
        let final = try read(dir, "final.f32", latentShape)
        let image = try read(dir, "image.f32", [3, bundle.height, bundle.width])
        let decoded = try await backend.decode(final)
        let decoderMetric = compare(image.values, decoded.pixelsCHW.values)
        #expect(decoderMetric.rmse <= 0.01)
        let noise = try read(dir, "noise.f32", latentShape)
        if c.steps == 4 {
            await #expect(throws: CancellationError.self) {
                try await session.generate(tokenIDs: ids, noise: noise, schedule: s) { step, _ in
                    if step == 0 { withUnsafeCurrentTask { $0?.cancel() } }
                }
            }
            report.append(["case": c.id, "stage": "cancellation", "afterStep": 0, "recovery": "following full generation"])
        }
        let result = try await session.generate(tokenIDs: ids, noise: noise, schedule: s)
        let metric = compare(image.values, result.pixelsCHW.values)
        #expect(metric.rmse <= 0.03)
        if c.steps == 1 {
            let repeated = try await session.generate(tokenIDs: ids, noise: noise, schedule: s)
            #expect(repeated.pixelsCHW == result.pixelsCHW)
            report.append(["case": c.id, "stage": "repeat", "exactPixels": repeated.pixelsCHW == result.pixelsCHW])
        }
        try result.savePNG(to: root.appendingPathComponent("native-\(c.id).png"))
        report.append(["case": c.id, "stage": "pipeline", "textCosine": textMetric.cosine, "decoderRMSE": decoderMetric.rmse, "cosine": metric.cosine, "rmse": metric.rmse])
    }
    await session.unload()
    await session.unload()
    let ids = try tokenizer.prepare("Closed session")
    let schedule = try bundle.schedule(steps: 1), noise = try bundle.noise(seed: 0)
    await #expect(throws: AetherError.sessionClosed) { try await session.generate(tokenIDs: ids, noise: noise, schedule: schedule) }
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: root.appendingPathComponent("native-report.json"))
}
