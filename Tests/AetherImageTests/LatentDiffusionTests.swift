import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing
@testable import AetherImage

/// `latent-diffusion@1` wiring with fake graphs (DESIGN.md §12): phase order, the host recipe (seeded NCHW noise, the
/// table's timesteps, scaled inputs, the Euler epsilon update), residency plans, refusals and cancellation. The real
/// graphs are the gate's (`LatentDiffusionGateTests`).
@Suite("Latent diffusion pipeline")
struct LatentDiffusionTests {
    static let manifestURL = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "AetherCoreTests/Fixtures/sd-turbo/aether.json")
    static let mac = Device(platform: .macOS, architecture: "test")
    static let latentCount = 4 * 64 * 64

    static func manifest() throws -> Manifest { try Manifest.decode(Data(contentsOf: manifestURL)) }

    /// A shared log of graph calls, in order.
    final class Calls: Sendable {
        private let state = Mutex<[String]>([])
        var log: [String] { state.withLock { $0 } }
        func append(_ entry: String) { state.withLock { $0.append(entry) } }
    }

    /// Text encoder: writes the ids (as floats) into the first 77 conditioning values.
    final class FakeText: GraphBackend {
        let signature = GraphSignature(
            function: "encode", inputs: [TensorSpec(name: "input_ids", scalarType: .int32, shape: [1, 77])], states: [],
            outputs: [TensorSpec(name: "text_embeds", scalarType: .float32, shape: [1, 77, 1024])])
        let calls: Calls
        init(_ calls: Calls) { self.calls = calls }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            calls.append("text")
            let ids = try bindings.input("input_ids")!.read(Int32.self)
            try bindings.output("text_embeds")!.write(ids.map(Float.init))
        }
    }

    /// U-Net: epsilon = 0.25 · input + conditioning[1] / 1e5 (the first prompt piece), recording each timestep.
    final class FakeUNet: GraphBackend {
        let signature = GraphSignature(
            function: "denoise",
            inputs: [
                TensorSpec(name: "sample", scalarType: .float32, shape: [1, 4, 64, 64]),
                TensorSpec(name: "timestep", scalarType: .float32, shape: [1]),
                TensorSpec(name: "text_embeds", scalarType: .float32, shape: [1, 77, 1024]),
            ], states: [], outputs: [TensorSpec(name: "epsilon", scalarType: .float32, shape: [1, 4, 64, 64])])
        let calls: Calls
        let onRun: @Sendable (Int) -> Void
        private var runs = 0
        init(_ calls: Calls, onRun: @escaping @Sendable (Int) -> Void = { _ in }) {
            self.calls = calls
            self.onRun = onRun
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            onRun(runs)
            runs += 1
            let t = try bindings.input("timestep")!.read(Float.self)[0]
            calls.append("unet \(Int(t))")
            let bias = try bindings.input("text_embeds")!.read(Float.self, 1..<2)[0] / 100_000
            let sample = try bindings.input("sample")!.readFloats()
            try bindings.output("epsilon")!.write(sample.map { $0 * 0.25 + bias })
        }
    }

    /// VAE: image[c, y, x] = latents[c, y / 8, x / 8] / 10 for c < 3.
    final class FakeVAE: GraphBackend {
        let signature = GraphSignature(
            function: "decode", inputs: [TensorSpec(name: "latents", scalarType: .float32, shape: [1, 4, 64, 64])], states: [],
            outputs: [TensorSpec(name: "image", scalarType: .float32, shape: [3, 512, 512])])
        let calls: Calls
        init(_ calls: Calls) { self.calls = calls }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            calls.append("vae")
            let latents = try bindings.input("latents")!.readFloats()
            var image = [Float](repeating: 0, count: 3 * 512 * 512)
            for c in 0..<3 { for y in 0..<512 { for x in 0..<512 {
                image[(c * 512 + y) * 512 + x] = latents[(c * 64 + y / 8) * 64 + x / 8] / 10
            } } }
            try bindings.output("image")!.write(image)
        }
    }

    /// A tokenizer over a tiny CLIP-shaped vocabulary (`TokenizerTests` builds it).
    static func tokenizer() throws -> CLIPTokenizer { try TokenizerTests.tiny() }

    static func pipeline(
        calls: Calls, budget: Int64 = 1 << 40, files: [String: Int64] = ["textEncoder": 10, "denoiser": 100, "vaeDecoder": 10],
        coordinator: ResourceCoordinator? = nil, onRun: @escaping @Sendable (Int) -> Void = { _ in }
    ) throws -> (LatentDiffusionPipeline, Calls) {
        let loads = Calls()
        let pipeline = try LatentDiffusionPipeline(
            manifest: try manifest(), tokenizer: try tokenizer(), device: mac,
            options: .init(memory: .bytes(budget), coordinator: coordinator, gpu: .unadmitted, probe: Flux2KleinTests.still),
            load: { role throws(AetherError) in
                loads.append(role)
                switch role {
                case "textEncoder": return FakeText(calls)
                case "denoiser": return FakeUNet(calls, onRun: onRun)
                default: return FakeVAE(calls)
                }
            },
            fileBytes: { files[$0] ?? 0 })
        return (pipeline, loads)
    }

    /// What the pipeline must compute, on the host, for the fakes above.
    static func expected(prompt: String, seed: UInt64, steps: Int) throws -> [Float] {
        let schedule = try EpsilonEulerSchedule(try manifest().schedule!, steps: steps)
        var x = try SeededGaussianNoise.generate(seed: seed, count: latentCount)
        x.withUnsafeMutableBufferPointer { schedule.scaleInitialNoise($0) }
        let bias = Float(try tokenizer().encode(prompt)[1]) / 100_000
        for step in 0..<steps {
            let input = try schedule.modelInput(step: step, latents: x)
            try schedule.advance(step: step, latents: &x, epsilon: input.map { $0 * 0.25 + bias })
        }
        return x
    }

    @Test func generatesThroughThePhasesWithTheHostRecipe() async throws {
        let calls = Calls()
        let (pipeline, loads) = try Self.pipeline(calls: calls)
        let image = try await pipeline.generate(.init(prompt: "a cat", seed: 42, steps: 4))
        #expect(calls.log == ["text", "unet 999", "unet 749", "unet 499", "unet 249", "vae"])
        #expect(loads.log == ["textEncoder", "denoiser", "vaeDecoder"])
        #expect(image.width == 512 && image.height == 512)
        let x = try Self.expected(prompt: "a cat", seed: 42, steps: 4)
        let pixel: Int = (1 * 512 + 100) * 512 + 300, latent: Int = (1 * 64 + 12) * 64 + 37
        let first: Float = x[0] / 10, inner: Float = x[latent] / 10
        #expect(image.planarRGB[0] == first)
        #expect(image.planarRGB[pixel] == inner)
        // Everything fit: the graphs stay loaded, and a second generation loads nothing.
        #expect(await pipeline.residentRoles == ["denoiser", "textEncoder", "vaeDecoder"])
        let again = try await pipeline.generate(.init(prompt: "a cat", seed: 42, steps: 4))
        #expect(again == image)
        #expect(loads.log.count == 3)
        await pipeline.unload()
        #expect(await pipeline.residentRoles.isEmpty)
    }

    @Test func defaultsComeFromTheManifest() async throws {
        let calls = Calls()
        let (pipeline, _) = try Self.pipeline(calls: calls)
        _ = try await pipeline.generate(.init(prompt: "x"))
        #expect(calls.log == ["text", "unet 999", "vae"])   // steps 1, 512x512
    }

    @Test func perPhaseReleasesEachPhasesGraphs() async throws {
        // Files 10 / 100 / 10: all resident needs 120; per phase, at most 10 held + 100 = 110.
        let calls = Calls()
        let (pipeline, loads) = try Self.pipeline(calls: calls, budget: 115)
        _ = try await pipeline.generate(.init(prompt: "a cat", seed: 1, steps: 2))
        #expect(await pipeline.residentRoles.isEmpty)
        #expect(loads.log == ["textEncoder", "denoiser", "vaeDecoder"])
    }

    @Test func refusesBeforeLoadingWhatCannotFit() async throws {
        let calls = Calls()
        let (pipeline, loads) = try Self.pipeline(calls: calls, budget: 50)
        await #expect(throws: AetherError.self) { try await pipeline.generate(.init(prompt: "a cat")) }
        #expect(loads.log.isEmpty && calls.log.isEmpty)
        // A shared coordinator: the plan's peak is reserved for the generation, then released.
        let coordinator = try ResourceCoordinator(capacityBytes: 1000)
        let (shared, _) = try Self.pipeline(calls: Calls(), coordinator: coordinator)
        _ = try await shared.generate(.init(prompt: "a cat"))
        #expect(coordinator.snapshot.reservedBytes == 0)
        // A fresh pipeline plans from its files (120); `shared` has recorded its (zero) measured phases.
        let (fresh, _) = try Self.pipeline(calls: Calls(), coordinator: coordinator)
        let held = try coordinator.reserve(bytes: 950)
        await #expect(throws: AetherError.self) { try await fresh.generate(.init(prompt: "a cat")) }
        held.release()
    }

    @Test func invalidRequestsAreRefusedBeforeAnyWork() async throws {
        let calls = Calls()
        let (pipeline, loads) = try Self.pipeline(calls: calls)
        for request in [LatentDiffusionPipeline.Request(prompt: "x", steps: 5), .init(prompt: "x", steps: 0),
                        .init(prompt: "x", width: 256, height: 256), .init(prompt: "x", width: 512, height: 520)] {
            await #expect(throws: AetherError.self) { try await pipeline.generate(request) }
        }
        #expect(loads.log.isEmpty)
    }

    @Test func cancellationStopsBetweenCallsAndReleases() async throws {
        let calls = Calls()
        let (pipeline, _) = try Self.pipeline(calls: calls) { run in
            if run == 0 { withUnsafeCurrentTask { $0?.cancel() } }
        }
        await #expect(throws: CancellationError.self) {
            try await pipeline.generate(.init(prompt: "a cat", steps: 4))
        }
        #expect(calls.log == ["text", "unet 999"])
        #expect(await pipeline.residentRoles.isEmpty)
        // Still usable.
        _ = try await pipeline.generate(.init(prompt: "a cat", steps: 1))
    }

    @Test func manifestsOutsideTheContractAreRefused() throws {
        var manifest = try Self.manifest()
        manifest.schedule?.guidance = .init(kind: .cfg, scale: 2)
        #expect(throws: AetherError.self) {
            try LatentDiffusionPipeline(manifest: manifest, tokenizer: try Self.tokenizer(), device: Self.mac,
                                        options: .init(gpu: .unadmitted), load: { _ throws(AetherError) in FakeText(Calls()) }, fileBytes: { _ in 0 })
        }
        manifest = try Self.manifest()
        manifest.graphs["denoiser"]?.inputs["conditioning"]?.source = nil
        #expect(throws: AetherError.self) {
            try LatentDiffusionPipeline(manifest: manifest, tokenizer: try Self.tokenizer(), device: Self.mac,
                                        options: .init(gpu: .unadmitted), load: { _ throws(AetherError) in FakeText(Calls()) }, fileBytes: { _ in 0 })
        }
        #expect(Registry.image.supports(.latentDiffusion))
    }
}

@Suite("CLIP tokenizer")
struct TokenizerTests {
    /// A CLIP-shaped vocabulary: the specials at their real ids, single bytes, and the merges `c a`, `ca t</w>`. With
    /// `sd2`, a `tokenizer_config.json` as SD 2.x's adds `!` (id 0) and makes it the pad token; without, there are no
    /// config files, as in a bare vocab/merges pair.
    static func tiny(sd2: Bool = true, length: Int = 8) throws -> CLIPTokenizer {
        let directory = FileManager.default.temporaryDirectory.appending(path: "aether-clip-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var vocab: [String: Int] = ["!": 0, "!</w>": 256, "<|startoftext|>": 49406, "<|endoftext|>": 49407]
        for (i, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocab[String(letter)] = 100 + i
            vocab[String(letter) + "</w>"] = 200 + i
        }
        vocab["ca"] = 300
        vocab["cat</w>"] = 301
        vocab[","] = 302
        vocab[",</w>"] = 303
        try JSONSerialization.data(withJSONObject: vocab).write(to: directory.appending(path: "vocab.json"))
        try "#version: 0.2\nc a\nca t</w>\n".write(to: directory.appending(path: "merges.txt"), atomically: true, encoding: .utf8)
        if sd2 {
            let config: [String: Any] = [
                "added_tokens_decoder": ["0": ["content": "!", "special": true], "49406": ["content": "<|startoftext|>"],
                                         "49407": ["content": "<|endoftext|>"]],
                "bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>", "pad_token": "!", "unk_token": "<|endoftext|>",
            ]
            try JSONSerialization.data(withJSONObject: config).write(to: directory.appending(path: "tokenizer_config.json"))
        }
        return try CLIPTokenizer(directory: directory, length: length)
    }

    @Test func mergesByRankWithTheEndOfWordSuffix() throws {
        let t = try Self.tiny()
        #expect(t.pad == 0 && t.addedTokens.contains("!"))
        #expect(try t.encode("Cat") == [49406, 301, 49407, 0, 0, 0, 0, 0])
        // "cab": c+a merge, b keeps its suffix; "!" is an added token; whitespace is dropped.
        #expect(try t.encode("cab  !\tcat") == [49406, 300, 201, 0, 301, 49407, 0, 0])
        #expect(try t.encode("") == [49406, 49407, 0, 0, 0, 0, 0, 0])
    }

    /// OpenAI CLIP's tokenizer adds only BOS and EOS: `!` is an ordinary piece (`!</w>`, 256), and the pad defaults to
    /// EOS, as transformers' `CLIPTokenizer` does.
    @Test func addedTokensComeFromTheTokenizerFiles() throws {
        let t = try Self.tiny(sd2: false)
        #expect(t.pad == 49407 && t.addedTokens == ["<|startoftext|>", "<|endoftext|>"])
        #expect(try t.encode("cat!") == [49406, 301, 256, 49407, 49407, 49407, 49407, 49407])
        // The text's own special-token strings are still split out.
        #expect(try t.encode("cat<|endoftext|>") == [49406, 301, 49407, 49407, 49407, 49407, 49407, 49407])
    }

    @Test func truncatesToTheContextAndEndsWithEOS() throws {
        let t = try Self.tiny()
        let ids = try t.encode("cat cat cat cat cat cat cat cat")
        #expect(ids == [49406, 301, 301, 301, 301, 301, 301, 49407])
    }

    @Test func rejectsAVocabularyThatIsNotCLIPs() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "aether-clip-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: ["a": 0]).write(to: directory.appending(path: "vocab.json"))
        try "".write(to: directory.appending(path: "merges.txt"), atomically: true, encoding: .utf8)
        #expect(throws: AetherError.self) { try CLIPTokenizer(directory: directory) }
    }
}
