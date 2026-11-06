import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Testing
@testable import AetherImage

/// `flux2-klein@1` wiring with fake graphs (DESIGN.md §12): the prompt rule, the segment chain (carries, the shared
/// condition, rotary tables), the host recipe (packed seeded noise, the empirical-shift schedule, fp16 latents, the
/// Euler update), the decoder layout, residency, refusals and cancellation. The real graphs are the gate's
/// (`Flux2KleinGateTests`). The exporter's manifest is shrunk to an 8-row prompt of width 16 so the fakes stay small.
@Suite("FLUX.2 [klein] pipeline")
struct Flux2KleinTests {
    typealias Calls = LatentDiffusionTests.Calls
    static let manifestURL = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "AetherCoreTests/Fixtures/flux2/flux2-klein-4b.json")
    static let mac = LatentDiffusionTests.mac
    static let rows = 8, width = 16, carry = 4
    /// A probe that never moves: plans come from the files, whatever else the process allocates.
    static let still = GPUMemoryProbe { 0 }

    static func manifest() throws -> Manifest {
        var manifest = try Manifest.decode(Data(contentsOf: manifestURL))
        manifest.flux2?.prompt.maxTokens = rows
        manifest.flux2?.conditioning.rows = rows
        manifest.flux2?.conditioning.width = width
        return manifest
    }

    /// Prompt ids: 100, the scalars, 200.
    static let encode: Flux2KleinPipeline.PromptEncoder = { text in [100] + text.unicodeScalars.map { Int32($0.value) } + [200] }

    /// Text encoder: row r = ids[r] · mask[r] / 1e5 + positions[r] / 1e3 in every column (fp16 out).
    final class FakeText: GraphBackend {
        let signature = GraphSignature(
            function: "encode",
            inputs: ["input_ids", "position_ids", "attention_mask"].map { TensorSpec(name: $0, scalarType: .int32, shape: [1, TensorSpec.dynamic]) },
            states: [], outputs: [TensorSpec(name: "text_embeds", scalarType: .float16, shape: [TensorSpec.dynamic, Flux2KleinTests.width])])
        let calls: Calls
        init(_ calls: Calls) { self.calls = calls }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            calls.append("text")
            let ids = try bindings.input("input_ids")!.read(Int32.self), mask = try bindings.input("attention_mask")!.read(Int32.self)
            let positions = try bindings.input("position_ids")!.read(Int32.self)
            try bindings.output("text_embeds")!.write(Float16.self, count: ids.count * Flux2KleinTests.width) { i in
                let r = i / Flux2KleinTests.width
                return Float16(Float(ids[r] * mask[r]) / 100_000 + Float(positions[r]) / 1000)
            }
        }
    }

    /// Segment 0: carryImage = the first 4 latent channels, carryText = the first 4 text columns, condition = sigma.
    /// Segment 1: carryImage + condition[0]. Segment 2: velocity = 0.5 · carryImage (first 4 channels) + carryText[1, 0].
    /// Each checks its rotary tables' row counts.
    final class FakeSegment: GraphBackend {
        let index: Int
        let signature: GraphSignature
        let calls: Calls
        let onRun: @Sendable () -> Void
        init(_ index: Int, _ calls: Calls, onRun: @escaping @Sendable () -> Void = {}) {
            self.index = index
            self.calls = calls
            self.onRun = onRun
            let rope = [TensorSpec(name: "rope_text", scalarType: .float32, shape: [TensorSpec.dynamic, 2, 128]),
                        TensorSpec(name: "rope_image", scalarType: .float32, shape: [TensorSpec.dynamic, 2, 128])]
            let carries = { (n: Int) in [TensorSpec(name: "image_\(n)", scalarType: .float32, shape: [TensorSpec.dynamic, Flux2KleinTests.carry]),
                                         TensorSpec(name: "text_\(n)", scalarType: .float32, shape: [TensorSpec.dynamic, Flux2KleinTests.carry])] }
            let cond = TensorSpec(name: "cond", scalarType: .float32, shape: [17, Flux2KleinTests.carry])
            switch index {
            case 0:
                signature = GraphSignature(
                    function: "segment",
                    inputs: [TensorSpec(name: "latents", scalarType: .float16, shape: [TensorSpec.dynamic, 128]),
                             TensorSpec(name: "text_embeds", scalarType: .float16, shape: [TensorSpec.dynamic, Flux2KleinTests.width]),
                             TensorSpec(name: "sigma", scalarType: .float32, shape: [1])] + rope,
                    states: [], outputs: carries(1) + [cond])
            case 1:
                signature = GraphSignature(function: "segment", inputs: carries(1) + [cond] + rope, states: [], outputs: carries(2))
            default:
                signature = GraphSignature(
                    function: "segment", inputs: carries(2) + [cond] + rope, states: [],
                    outputs: [TensorSpec(name: "velocity", scalarType: .float16, shape: [TensorSpec.dynamic, 128])])
            }
        }

        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            onRun()
            let tokens = bindings.boundShape(of: "rope_image")![0]
            guard bindings.boundShape(of: "rope_text")![0] == Flux2KleinTests.rows else { throw .tensor(name: "rope_text", reason: .unknownName) }
            let c = Flux2KleinTests.carry
            switch index {
            case 0:
                let sigma = try bindings.input("sigma")!.read(Float.self)[0]
                calls.append("segment0 \(sigma)")
                let latents = try bindings.input("latents")!.readFloats(), text = try bindings.input("text_embeds")!.readFloats()
                try bindings.output("image_1")!.write(Float.self, count: tokens * c) { latents[($0 / c) * 128 + $0 % c] }
                try bindings.output("text_1")!.write(Float.self, count: Flux2KleinTests.rows * c) { text[($0 / c) * Flux2KleinTests.width + $0 % c] }
                try bindings.output("cond")!.fill(sigma)
            case 1:
                calls.append("segment1")
                let image = try bindings.input("image_1")!.readFloats(), cond = try bindings.input("cond")!.readFloats()
                try bindings.output("image_2")!.write(image.map { $0 + cond[0] })
                try bindings.output("text_2")!.copy(from: bindings.input("text_1")!)
            default:
                calls.append("segment2")
                let image = try bindings.input("image_2")!.readFloats(), text = try bindings.input("text_2")!.readFloats()
                try bindings.output("velocity")!.write(Float16.self, count: tokens * 128) { i in
                    let t = i / 128, ch = i % 128
                    return Float16((ch < c ? image[t * c + ch] : 0) * 0.5 + text[1 * c + 0])
                }
            }
        }
    }

    /// VAE: image[ch, y, x] = latents[y / 16, x / 16, ch] for ch < 3.
    final class FakeVAE: GraphBackend {
        let signature = GraphSignature(
            function: "decode", inputs: [TensorSpec(name: "latents", scalarType: .float16, shape: [TensorSpec.dynamic, TensorSpec.dynamic, 128])], states: [],
            outputs: [TensorSpec(name: "image", scalarType: .float32, shape: [3, TensorSpec.dynamic, TensorSpec.dynamic])])
        let calls: Calls
        init(_ calls: Calls) { self.calls = calls }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            calls.append("vae")
            let shape = bindings.boundShape(of: "latents")!, (rows, columns) = (shape[0], shape[1])
            let latents = try bindings.input("latents")!.readFloats()
            let (h, w) = (rows * 16, columns * 16)
            try bindings.output("image")!.write(Float.self, count: 3 * h * w) { i in
                let ch = i / (h * w), y = (i / w) % h, x = i % w
                return latents[((y / 16) * columns + x / 16) * 128 + ch]
            }
        }
    }

    static func pipeline(
        calls: Calls, device: Device = mac, budget: Int64 = 1 << 40,
        files: [String: Int64] = ["textEncoder": 10, "denoiser0": 30, "denoiser1": 30, "denoiser2": 30, "vaeDecoder": 10],
        coordinator: ResourceCoordinator? = nil, onRun: @escaping @Sendable () -> Void = {}
    ) throws -> (Flux2KleinPipeline, Calls) {
        let loads = Calls()
        let pipeline = try Flux2KleinPipeline(
            manifest: try manifest(), device: device,
            options: .init(memory: .bytes(budget), coordinator: coordinator, probe: still), encodePrompt: encode,
            load: { role throws(AetherError) in
                loads.append(role)
                switch role {
                case "textEncoder": return FakeText(calls)
                case "denoiser0": return FakeSegment(0, calls, onRun: onRun)
                case "denoiser1": return FakeSegment(1, calls)
                case "denoiser2": return FakeSegment(2, calls)
                default: return FakeVAE(calls)
                }
            },
            fileBytes: { files[$0] ?? 0 })
        return (pipeline, loads)
    }

    /// The final packed latents the pipeline must reach, on the host, for the fakes above.
    static func expected(prompt: String, seed: UInt64, steps: Int, tokens: Int) throws -> [Float] {
        let schedule = try FlowMatchSchedule(try manifest().schedule!, steps: steps, imageTokens: tokens)
        let ids = try encode(prompt)
        // Row 1 of the text: id 1 is real, at position 1.
        let bias = Float(Float16(Float(ids[1]) / 100_000 + 1 / 1000))
        var x = try SeededGaussianNoise.generate(seed: seed, count: tokens * 128)
        for step in 0..<steps {
            let sigma = schedule.timesteps[step]
            let v = (0..<x.count).map { i -> Float in
                let ch = i % 128
                let image = ch < carry ? Float(Float16(x[i])) + sigma : 0
                return Float(Float16(image * 0.5 + bias))
            }
            try schedule.advance(step: step, latents: &x, velocity: v)
        }
        return x
    }

    @Test func generatesThroughTheSegmentsWithTheHostRecipe() async throws {
        let calls = Calls()
        let (pipeline, loads) = try Self.pipeline(calls: calls)
        let image = try await pipeline.generate(.init(prompt: "a cat", seed: 7, steps: 2, width: 48, height: 32))
        let schedule = try FlowMatchSchedule(try Self.manifest().schedule!, steps: 2, imageTokens: 6)
        #expect(calls.log == ["text"] + schedule.timesteps.flatMap { ["segment0 \($0)", "segment1", "segment2"] } + ["vae"])
        #expect(loads.log == ["textEncoder", "denoiser0", "denoiser1", "denoiser2", "vaeDecoder"])
        #expect(image.width == 48 && image.height == 32)
        // Pixel (channel 2, y 20, x 40) is token (1, 2), channel 2, rounded to fp16 for the decoder.
        let x = try Self.expected(prompt: "a cat", seed: 7, steps: 2, tokens: 6)
        let pixel: Int = (2 * 32 + 20) * 48 + 40, latent: Int = (1 * 3 + 2) * 128 + 2
        let inner = Float(Float16(x[latent])), first = Float(Float16(x[0]))
        #expect(image.planarRGB[pixel] == inner)
        #expect(image.planarRGB[0] == first)
        // Everything fit: a second generation loads nothing and repeats the image.
        #expect(try await pipeline.generate(.init(prompt: "a cat", seed: 7, steps: 2, width: 48, height: 32)) == image)
        #expect(loads.log.count == 5)
        await pipeline.unload()
        #expect(await pipeline.residentRoles.isEmpty)
    }

    @Test func thePromptIsPaddedOrKeepsItsPrefix() throws {
        let rule = try #require(try Self.manifest().flux2?.prompt)
        let short = try Flux2KleinPipeline.Prompt(rendered: [5, 6, 7], rule: rule)
        #expect(short.ids == [5, 6, 7] + Array(repeating: Int32(rule.padToken), count: 5))
        #expect(short.mask == [1, 1, 1, 0, 0, 0, 0, 0] && !short.truncated)
        let long = try Flux2KleinPipeline.Prompt(rendered: Array(1...12), rule: rule)
        #expect(long.ids == Array(1...8) && long.mask == Array(repeating: 1, count: 8) && long.truncated)
        #expect(throws: AetherError.self) { try Flux2KleinPipeline.Prompt(rendered: [], rule: rule) }
    }

    /// Split halves, 4 axes of 32, θ 2000: token i of the text has angle i on the last axis's first pair (index 48),
    /// and grid token (row 2, column 3) angle 2 on the second axis (index 16) and 3 on the third (index 32).
    @Test func rotaryTablesFollowTheManifest() throws {
        let rotary = try #require(try Self.manifest().flux2?.rotary)
        #expect(rotary.dimension == 128)
        let text = Flux2Rotary.table(ids: Flux2Rotary.textIDs(count: 3), rotary: rotary)
        #expect(text.count == 3 * 2 * 128)
        let base = 1 * 256
        #expect(text[base + 48] == Float(cos(1.0)) && text[base + 48 + 64] == Float(cos(1.0)))
        #expect(text[base + 128 + 48] == Float(sin(1.0)) && text[base + 128 + 48 + 64] == Float(sin(1.0)))
        #expect(text[base + 0] == 1 && text[base + 128] == 0)   // axis 0 is 0 for text
        let grid = Flux2Rotary.table(ids: Flux2Rotary.gridIDs(rows: 3, columns: 4), rotary: rotary)
        let token = (2 * 4 + 3) * 256
        #expect(grid[token + 16] == Float(cos(2.0)) && grid[token + 32] == Float(cos(3.0)) && grid[token + 48] == 1)
        // The second pair of an axis turns slower: θ^(2/32).
        #expect(grid[token + 17] == Float(cos(2 * (1 / pow(2000, 2.0 / 32)))))
    }

    @Test func perPhaseReleasesEachPhasesGraphs() async throws {
        // Files 10 / 90 / 10: all resident needs 110; per phase, at most 10 held + 90 = 100.
        let (pipeline, loads) = try Self.pipeline(calls: Calls(), budget: 105)
        _ = try await pipeline.generate(.init(prompt: "a cat", steps: 1, width: 32, height: 32))
        #expect(await pipeline.residentRoles.isEmpty)
        #expect(loads.log == ["textEncoder", "denoiser0", "denoiser1", "denoiser2", "vaeDecoder"])
    }

    @Test func refusalsComeBeforeAnyWork() async throws {
        let calls = Calls()
        // Too little memory, no size, a size off the grid, too many tokens, too many steps.
        let (small, smallLoads) = try Self.pipeline(calls: calls, budget: 50)
        await #expect(throws: AetherError.self) { try await small.generate(.init(prompt: "a cat", width: 32, height: 32)) }
        #expect(smallLoads.log.isEmpty)
        let (pipeline, loads) = try Self.pipeline(calls: calls)
        for request in [Flux2KleinPipeline.Request(prompt: "x"), .init(prompt: "x", width: 40, height: 32),
                        .init(prompt: "x", width: 2048, height: 2048), .init(prompt: "x", steps: 51, width: 32, height: 32)] {
            await #expect(throws: AetherError.self) { try await pipeline.generate(request) }
        }
        #expect(loads.log.isEmpty && calls.log.isEmpty)
        // iOS: refused when the pipeline opens.
        #expect {
            _ = try Self.pipeline(calls: calls, device: Device(platform: .iOS, architecture: "test"))
        } throws: { error in
            guard case .unsupportedDevice(let id, "ios", _)? = error as? AetherError else { return false }
            return id == "flux2-klein-4b"
        }
    }

    @Test func cancellationStopsBetweenCallsAndReleases() async throws {
        let calls = Calls()
        let (pipeline, _) = try Self.pipeline(calls: calls) { withUnsafeCurrentTask { $0?.cancel() } }
        await #expect(throws: CancellationError.self) {
            try await pipeline.generate(.init(prompt: "a cat", steps: 2, width: 32, height: 32))
        }
        #expect(calls.log.count == 2 && calls.log[0] == "text" && calls.log[1].hasPrefix("segment0"))
        #expect(await pipeline.residentRoles.isEmpty)
    }

    @Test func manifestsOutsideTheContractAreRefused() throws {
        func refuses(_ change: (inout Manifest) -> Void) throws {
            var manifest = try Self.manifest()
            change(&manifest)
            #expect(throws: AetherError.self) {
                try Flux2KleinPipeline(manifest: manifest, device: Self.mac, options: .init(), encodePrompt: Self.encode,
                                       load: { _ throws(AetherError) in FakeVAE(Calls()) }, fileBytes: { _ in 0 })
            }
        }
        // A carry from a later segment, a host-fed carry, guidance, no flux2 block, segments out of order.
        try refuses { $0.graphs["denoiser1"]?.inputs["carryImage"]?.source = .from("denoiser2.velocity") }
        try refuses { $0.graphs["denoiser1"]?.inputs["carryText"]?.source = nil }
        try refuses { $0.schedule?.guidance = .init(kind: .cfg, scale: 2) }
        try refuses { $0.flux2 = nil }
        try refuses { $0.flux2?.segments = ["denoiser1", "denoiser0", "denoiser2"] }
        // The optional reference encoder may be absent.
        var manifest = try Self.manifest()
        manifest.graphs["vaeEncoder"] = nil
        _ = try Flux2KleinPipeline(manifest: manifest, device: Self.mac, options: .init(), encodePrompt: Self.encode,
                                   load: { _ throws(AetherError) in FakeVAE(Calls()) }, fileBytes: { _ in 0 })
        #expect(Registry.image.supports(.flux2Klein) && Registry.image.supports(.latentDiffusion))
    }
}
