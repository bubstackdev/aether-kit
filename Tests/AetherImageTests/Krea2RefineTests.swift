import AetherCore
import AetherHost
import Foundation
import Synchronization
import Testing
@testable import AetherImage
@testable import AetherRuntime

/// Refine (image-to-image) on the Krea 2 fakes, with a fake VAE encoder added to the fixture manifest: the encoder's
/// packed latents, the flow-matching start at a step or an off-grid sigma, caller noise, conditioning reuse across
/// tiles, and the request checks. The real encoder and a refine tile are the Forge gate's.
@Suite("Krea 2 refine")
struct Krea2RefineTests {
    typealias Calls = LatentDiffusionTests.Calls

    /// The fixture manifest with a `vaeEncoder` (pixels `image` → `latents`), in the prompt phase.
    static func manifest() throws -> Manifest {
        var manifest = try Krea2Tests.manifest()
        var encoder = manifest.graphs["vaeDecoder"]!
        encoder.file = .single(AssetPath(asset: "main", path: "krea2_turbo-vae-encoder.aimodel"))
        encoder.function = "encode"
        encoder.inputs = ["pixels": "image"]
        encoder.outputs = ["latents": "latents"]
        encoder.residency = .perPrompt
        manifest.graphs["vaeEncoder"] = encoder
        return manifest
    }

    /// Encoder: latents[r, c, ch] = pixel (channel ch % 3, y 16r, x 16c) / 4 + ch / 1000 (fp16 in, fp32 out).
    final class FakeEncoder: GraphBackend {
        let signature = GraphSignature(
            function: "encode",
            inputs: [TensorSpec(name: "image", scalarType: .float16, shape: [3, TensorSpec.dynamic, TensorSpec.dynamic])],
            states: [], outputs: [TensorSpec(name: "latents", scalarType: .float32, shape: [TensorSpec.dynamic, TensorSpec.dynamic, 64])])
        let calls: Calls
        init(_ calls: Calls) { self.calls = calls }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            calls.append("encoder")
            let shape = bindings.boundShape(of: "image")!, (h, w) = (shape[1], shape[2])
            let pixels = try bindings.input("image")!.readFloats()
            let (rows, columns) = (h / 16, w / 16)
            try bindings.output("latents")!.write(Float.self, count: rows * columns * 64) { i in
                let t = i / 64, ch = i % 64, r = t / columns, c = t % columns
                return pixels[((ch % 3) * h + 16 * r) * w + 16 * c] / 4 + Float(ch) / 1000
            }
        }
    }

    static func pipeline(
        calls: Calls, filter: any ContentFilter = Krea2Tests.RecordingFilter(), manifest: Manifest? = nil
    ) throws -> (Krea2Pipeline, Calls) {
        try Krea2Tests.pipeline(
            calls: calls, filter: filter, manifest: try manifest ?? Self.manifest(),
            wrap: { role, graph in role == "vaeEncoder" ? FakeEncoder(calls) : graph })
    }

    /// A deterministic 48x32 picture on [-1, 1].
    static let picture: GeneratedImage = {
        let (w, h) = (48, 32)
        let values = (0..<(3 * w * h)).map { i in Float((i * 37) % 200) / 100 - 1 }
        return try! GeneratedImage(width: w, height: h, planarRGB: values)
    }()

    static func request(seed: UInt64 = 7, steps: Int = 4) -> ImageRequest {
        ImageRequest(prompt: "a cat", seed: seed, steps: steps, width: 48, height: 32)
    }

    /// The final latents the fakes reach from `start` on the host: the flow-matching start, then the remaining steps
    /// (Krea2Tests' velocity: 0.5 · (fp16 latents + sigma) on the first 4 channels, plus the prompt's bias).
    static func expected(x0: [Float], noise: [Float], schedule: FlowMatchSchedule, from step: Int) throws -> [Float] {
        let ids = try Krea2Tests.encode("a cat")
        let bias = Float(Float16(Float(ids[Krea2Tests.drop]) / 100_000 + Float(Krea2Tests.drop) / 1000))
        var x = try schedule.refineStart(x0: x0, noise: noise, step: step)
        for s in step..<schedule.steps {
            let sigma = schedule.timesteps[s]
            let v = (0..<x.count).map { i -> Float in
                let image = i % 64 < Krea2Tests.carry ? Float(Float16(x[i])) + sigma : 0
                return Float(Float16(image * 0.5 + bias))
            }
            try schedule.advance(step: s, latents: &x, velocity: v)
        }
        return x
    }

    @Test func theEncoderPacksAnImageIntoTheDecodersLatents() async throws {
        let calls = Calls(), filter = Krea2Tests.RecordingFilter()
        let (pipeline, loads) = try Self.pipeline(calls: calls, filter: filter)
        #expect(pipeline.supportsRefine)
        let latents = try await pipeline.encode(image: Self.picture)
        #expect(latents.rows == 2 && latents.columns == 3 && latents.channels == 64)
        let p = Self.picture.planarRGB
        // Token (1, 2), channel 4: pixel channel 1 at (y 16, x 32), through fp16.
        let pixel: Float = Float(Float16(p[(1 * 32 + 16) * 48 + 32]))
        let expected: Float = pixel / 4 + Float(4) / 1000
        #expect(latents.values[(1 * 3 + 2) * 64 + 4] == expected)
        #expect(loads.log == ["vaeEncoder"] && calls.log == ["encoder"] && filter.log.isEmpty)
    }

    @Test func aRefineRunsTheRemainingStepsFromTheEncodedImage() async throws {
        let calls = Calls()
        let (pipeline, _) = try Self.pipeline(calls: calls)
        let x0 = try await pipeline.encode(image: Self.picture)
        let generation = try await pipeline.generate(Self.request(), refining: try Krea2RefineStart(latents: x0, start: .step(2)))
        let schedule = try FlowMatchSchedule(try Self.manifest().schedule!, steps: 4, imageTokens: 6)
        #expect(calls.log.filter { $0.hasPrefix("segment0") } == schedule.timesteps[2...].map { "segment0 \($0)" })
        #expect(generation.metrics.stepSeconds.count == 2)
        let noise = try SeededGaussianNoise.generate(seed: 7, count: 6 * 64)
        let x = try Self.expected(x0: x0.values, noise: noise, schedule: schedule, from: 2)
        let pixel: Int = (2 * 32 + 20) * 48 + 40, latent: Int = (1 * 3 + 2) * 64 + 2
        #expect(generation.image.planarRGB[pixel] == Float(Float16(x[latent])))
    }

    @Test func callerNoiseReplacesTheSeedsAndTilesShareOneEncode() async throws {
        let calls = Calls()
        let (pipeline, _) = try Self.pipeline(calls: calls)
        let start = try Krea2RefineStart(latents: try await pipeline.encode(image: Self.picture), start: .step(3))
        let seeded = try await pipeline.generate(Self.request(seed: 7), refining: start).image
        let conditioning = try await pipeline.encode(prompt: "a cat")
        let textRuns = calls.log.filter { $0 == "text" }.count
        let noise = try SeededGaussianNoise.generate(seed: 7, count: 6 * 64)
        // The seed's noise passed in renders the seeded image, whatever the request's seed; other noise doesn't.
        let supplied = try await pipeline.generate(Self.request(seed: 99), refining: start, reusing: conditioning, noise: noise).image
        let other = try await pipeline.generate(
            Self.request(seed: 7), refining: start, reusing: conditioning, noise: noise.map { -$0 }).image
        #expect(supplied == seeded && other != seeded)
        #expect(calls.log.filter { $0 == "text" }.count == textRuns)
    }

    @Test func anOffGridStartRunsOneStepFromItsSigma() async throws {
        let calls = Calls()
        let (pipeline, _) = try Self.pipeline(calls: calls)
        let x0 = try await pipeline.encode(image: Self.picture)
        _ = try await pipeline.generate(Self.request(), refining: try Krea2RefineStart(latents: x0, start: .lastStepFrom(sigma: 0.2)))
        #expect(calls.log.filter { $0.hasPrefix("segment0") } == ["segment0 0.2"])
        let schedule = try FlowMatchSchedule(try Self.manifest().schedule!, steps: 4, imageTokens: 6)
        let replaced = try schedule.replacingLastSigma(0.2)
        #expect(replaced.sigmas.dropLast(2) == schedule.sigmas.dropLast(2) && replaced.sigmas[3] == 0.2 && replaced.sigmas[4] == 0)
    }

    @Test func requestsThatDontFitTheRefineAreRefused() async throws {
        let (pipeline, loads) = try Self.pipeline(calls: Calls())
        let x0 = try await pipeline.encode(image: Self.picture)
        #expect(throws: AetherError.self) { _ = try Krea2RefineStart(latents: x0, start: .step(0)) }
        #expect(throws: AetherError.self) { _ = try Krea2RefineStart(latents: x0, start: .lastStepFrom(sigma: 1)) }
        let loaded = loads.log.count
        func refused(_ request: ImageRequest, _ start: Krea2RefineStart.Start, noise: [Float]? = nil) async -> Bool {
            await Krea2ConditioningTests.isInvalidParameters {
                _ = try await pipeline.generate(request, refining: try Krea2RefineStart(latents: x0, start: start), noise: noise)
            }
        }
        #expect(await refused(Self.request(steps: 4), .step(4)))
        #expect(await refused(ImageRequest(prompt: "a cat", seed: 7, steps: 4, width: 32, height: 32), .step(2)))
        #expect(await refused(Self.request(steps: 4), .lastStepFrom(sigma: 0.9)))
        #expect(await refused(Self.request(steps: 4), .step(2), noise: [0, 1]))
        #expect(loads.log.count == loaded)
    }

    @Test func anExportWithoutTheEncoderHasNoRefine() async throws {
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: Calls())
        #expect(!pipeline.supportsRefine)
        await #expect(throws: AetherError.self) { _ = try await pipeline.encode(image: Self.picture) }
        #expect(loads.log.isEmpty)
    }
}
