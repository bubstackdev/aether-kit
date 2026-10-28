import AetherCoreAI
import AetherKit
import AetherSpeech
import Foundation

/// Decoder-only mode (`Assets/decoder-only.json`, staged by `stage_device_assets.py --decoder-only`): runs a staged
/// reduced-precision decoder and the float32 decoder on the same synthetic inputs at one bucket, through
/// `ModelSession` with `preferGPU`. AetherSpeech refuses non-float32 bundles by design, so this bypasses the session.
/// It asks one question: does the float16-compute decoder still abort on the GPU (MPSGraph placing its float16 regions
/// on the Neural Engine, as on macOS 27.0.1), and if not, how far its output is from float32. An abort kills the
/// process; the last `KOKORO_DEVICE_PROGRESS` line shows where.
struct DecoderOnlyResult: Encodable {
    let frames: Int
    let bucket: Int
    let float32Seconds: [Double]
    let reducedSeconds: [Double]
    /// Reduced-precision output against float32 on the same inputs, over the real frames' samples.
    let relativeL2: Double
    let maxAbs: Double
    let reducedFinite: Bool
}

private struct DecoderOnlyPlan: Decodable { let frames: Int; let bucket: Int }

func runKokoroDecoderOnlyProbe(assetsRoot: URL, report: inout DeviceReport) async throws {
    let plan = try JSONDecoder().decode(DecoderOnlyPlan.self, from: Data(contentsOf: assetsRoot.appendingPathComponent("decoder-only.json")))
    let (count, shape) = (plan.frames, plan.bucket)
    let style = try FloatTensor(shape: [1, 256], values: (0..<256).map { Float(($0 % 17) - 8) / 80 })
    let inputs: [String: TensorValue] = [
        "prosody": .float32(try FloatTensor(shape: [1, 640, shape], values: (0..<(640 * shape)).map { i in i % shape < count ? Float(i % 23) / 23 - 0.5 : 0 })),
        "text": .float32(try FloatTensor(shape: [1, 512, shape], values: (0..<(512 * shape)).map { i in i % shape < count ? Float(i % 19) / 19 - 0.5 : 0 })),
        "style": .float32(style),
        "noise": .float32(try FloatTensor(shape: [1, shape, 600, 9], values: KokoroNoise.generate(seed: UInt64(count), count: shape * 5400))),
        "frames": .int32(try Int32Tensor(shape: [1], values: [Int32(count)])),
    ]
    func run(_ name: String) async throws -> ([Float], [Double]) {
        progressLine("opening \(name) decoder")
        let session = try await ModelSession.open(at: assetsRoot.appendingPathComponent("decoder-only/\(name).aimodel"), compute: .preferGPU)
        var seconds: [Double] = [], audio: [Float] = []
        for attempt in 0..<3 {
            progressLine("\(name) decode \(attempt)")
            let started = ContinuousClock.now
            let outputs = try await session.runTensors(function: "decode", inputs: inputs)
            let elapsed = ContinuousClock.now - started
            seconds.append(Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18)
            guard outputs.count == 1, case .float32(let tensor)? = outputs.values.first else {
                throw AetherError.invalidTensor("decoder must return one float32 tensor")
            }
            audio = Array(tensor.values.prefix(count * 600))
        }
        await session.unload()
        return (audio, seconds)
    }
    let (reference, float32Seconds) = try await run("float32")
    let (reduced, reducedSeconds) = try await run("reduced")
    var difference = 0.0, norm = 0.0, maxAbs = 0.0
    for (a, b) in zip(reduced, reference) {
        let d = Double(a) - Double(b)
        difference += d * d; norm += Double(b) * Double(b); maxAbs = max(maxAbs, abs(d))
    }
    report.decoderOnly = DecoderOnlyResult(frames: count, bucket: shape, float32Seconds: float32Seconds, reducedSeconds: reducedSeconds,
                                           relativeL2: (difference / max(norm, .leastNonzeroMagnitude)).squareRoot(), maxAbs: maxAbs,
                                           reducedFinite: reduced.allSatisfy(\.isFinite))
    progressLine("decoder-only complete")
}
