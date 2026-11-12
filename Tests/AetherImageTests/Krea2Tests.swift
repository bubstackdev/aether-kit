import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing
@testable import AetherImage

/// `krea2@1` wiring with fake graphs (DESIGN.md §12): the content filter (prompt before any work, image before it is
/// returned), the prompt rule, the four-segment chain (carries, the shared condition, the image rotary table), the host
/// recipe (packed seeded noise, the exponential-shift schedule, fp16 latents, the Euler update), the decoder layout,
/// residency, refusals and cancellation. The real graphs are the gate's (`Krea2GateTests`). The exporter's manifest is
/// shrunk to a 2-row prefix, 10 tokens and conditioning width 4 so the fakes stay small.
@Suite("Krea 2 pipeline")
struct Krea2Tests {
    typealias Calls = LatentDiffusionTests.Calls
    static let manifestURL = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "AetherCoreTests/Fixtures/krea2/vanilla.json")
    static let mac = LatentDiffusionTests.mac
    static let drop = 2, maxTokens = 10, width = 4, carry = 4
    static var columns: Int { 12 * width }
    /// A probe that never moves: plans come from the files, whatever else the process allocates.
    static let still = GPUMemoryProbe { 0 }

    static func manifest() throws -> Manifest {
        var manifest = try Manifest.decode(Data(contentsOf: manifestURL))
        manifest.krea2?.prompt.dropLeadingRows = drop
        manifest.krea2?.prompt.maxTokens = maxTokens
        manifest.krea2?.prompt.keepTrailingTokens = 2
        manifest.krea2?.conditioning.width = width
        manifest.krea2?.conditioning.rows = "1...\(maxTokens - drop)"
        return manifest
    }

    /// Rendered ids: 100, 101 (the "prefix"), the scalars, 200.
    static let encode: Krea2Pipeline.PromptEncoder = { text in [100, 101] + text.unicodeScalars.map { Int32($0.value) } + [200] }

    /// A filter that records what it saw and refuses the prompt or the image when told to.
    final class RecordingFilter: ContentFilter {
        let refusePrompt: Bool, refuseImage: Bool
        let seen = Mutex<[String]>([])
        init(refusePrompt: Bool = false, refuseImage: Bool = false) {
            self.refusePrompt = refusePrompt
            self.refuseImage = refuseImage
        }
        var log: [String] { seen.withLock { $0 } }
        func review(prompt: String) async throws -> ContentVerdict {
            seen.withLock { $0.append("prompt \(prompt)") }
            return refusePrompt ? .refuse(reason: "no prompts today") : .allow
        }
        func review(image: GeneratedImage, prompt: String) async throws -> ContentVerdict {
            seen.withLock { $0.append("image \(image.width)x\(image.height) \(prompt)") }
            return refuseImage ? .refuse(reason: "no images today") : .allow
        }
    }

    /// Text encoder: conditioning row r = ids[drop + r] / 1e5 + positions[drop + r] / 1e3 in every column (fp16 out).
    final class FakeText: GraphBackend {
        let signature = GraphSignature(
            function: "encode",
            inputs: ["input_ids", "position_ids"].map { TensorSpec(name: $0, scalarType: .int32, shape: [1, TensorSpec.dynamic]) },
            states: [], outputs: [TensorSpec(name: "text_embeds", scalarType: .float16, shape: [TensorSpec.dynamic, Krea2Tests.columns])])
        let calls: Calls
        init(_ calls: Calls) { self.calls = calls }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            calls.append("text")
            let ids = try bindings.input("input_ids")!.read(Int32.self), positions = try bindings.input("position_ids")!.read(Int32.self)
            let rows = ids.count - Krea2Tests.drop
            try bindings.output("text_embeds")!.write(Float16.self, count: rows * Krea2Tests.columns) { i in
                let r = i / Krea2Tests.columns + Krea2Tests.drop
                return Float16(Float(ids[r]) / 100_000 + Float(positions[r]) / 1000)
            }
        }
    }

    /// Segment 0: carryImage = the first 4 latent channels, carryText = the first 4 text columns, condition = sigma.
    /// Segment 1: carryImage + condition[0]. Segment 2: the carries unchanged. Segment 3: velocity =
    /// 0.5 · carryImage (first 4 channels) + carryText[0, 0]. Each checks its rotary table's row count.
    final class FakeSegment: GraphBackend {
        let index: Int
        let signature: GraphSignature
        let calls: Calls
        let onRun: @Sendable () -> Void
        init(_ index: Int, _ calls: Calls, onRun: @escaping @Sendable () -> Void = {}) {
            self.index = index
            self.calls = calls
            self.onRun = onRun
            let rope = TensorSpec(name: "rope_image", scalarType: .float32, shape: [TensorSpec.dynamic, 2, 128])
            let carries = { (n: Int) in [TensorSpec(name: "image_\(n)", scalarType: .float32, shape: [TensorSpec.dynamic, Krea2Tests.carry]),
                                         TensorSpec(name: "text_\(n)", scalarType: .float32, shape: [TensorSpec.dynamic, Krea2Tests.carry])] }
            let cond = TensorSpec(name: "cond", scalarType: .float32, shape: [8, Krea2Tests.carry])
            switch index {
            case 0:
                signature = GraphSignature(
                    function: "segment",
                    inputs: [TensorSpec(name: "latents", scalarType: .float16, shape: [TensorSpec.dynamic, 64]),
                             TensorSpec(name: "text_embeds", scalarType: .float16, shape: [TensorSpec.dynamic, Krea2Tests.columns]),
                             TensorSpec(name: "sigma", scalarType: .float32, shape: [1]), rope],
                    states: [], outputs: carries(1) + [cond])
            case 3:
                signature = GraphSignature(
                    function: "segment", inputs: carries(3) + [cond, rope], states: [],
                    outputs: [TensorSpec(name: "velocity", scalarType: .float16, shape: [TensorSpec.dynamic, 64])])
            default:
                signature = GraphSignature(function: "segment", inputs: carries(index) + [cond, rope], states: [], outputs: carries(index + 1))
            }
        }

        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            onRun()
            let tokens = bindings.boundShape(of: "rope_image")![0]
            let c = Krea2Tests.carry
            switch index {
            case 0:
                let sigma = try bindings.input("sigma")!.read(Float.self)[0]
                calls.append("segment0 \(sigma)")
                let latents = try bindings.input("latents")!.readFloats(), text = try bindings.input("text_embeds")!.readFloats()
                let rows = bindings.boundShape(of: "text_embeds")![0]
                try bindings.output("image_1")!.write(Float.self, count: tokens * c) { latents[($0 / c) * 64 + $0 % c] }
                try bindings.output("text_1")!.write(Float.self, count: rows * c) { text[($0 / c) * Krea2Tests.columns + $0 % c] }
                try bindings.output("cond")!.fill(sigma)
            case 1:
                calls.append("segment1")
                let image = try bindings.input("image_1")!.readFloats(), cond = try bindings.input("cond")!.readFloats()
                try bindings.output("image_2")!.write(image.map { $0 + cond[0] })
                try bindings.output("text_2")!.copy(from: bindings.input("text_1")!)
            case 2:
                calls.append("segment2")
                try bindings.output("image_3")!.copy(from: bindings.input("image_2")!)
                try bindings.output("text_3")!.copy(from: bindings.input("text_2")!)
            default:
                calls.append("segment3")
                let image = try bindings.input("image_3")!.readFloats(), text = try bindings.input("text_3")!.readFloats()
                try bindings.output("velocity")!.write(Float16.self, count: tokens * 64) { i in
                    let t = i / 64, ch = i % 64
                    return Float16((ch < c ? image[t * c + ch] : 0) * 0.5 + text[0])
                }
            }
        }
    }

    /// VAE: image[ch, y, x] = latents[y / 16, x / 16, ch] for ch < 3.
    final class FakeVAE: GraphBackend {
        let signature = GraphSignature(
            function: "decode", inputs: [TensorSpec(name: "latents", scalarType: .float16, shape: [TensorSpec.dynamic, TensorSpec.dynamic, 64])], states: [],
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
                return latents[((y / 16) * columns + x / 16) * 64 + ch]
            }
        }
    }

    static let files: [String: Int64] = [
        "textEncoder": 10, "denoiser0": 20, "denoiser1": 20, "denoiser2": 20, "denoiser3": 20, "vaeDecoder": 10,
    ]

    static func pipeline(
        calls: Calls, filter: any ContentFilter = RecordingFilter(), device: Device = mac, budget: Int64 = 1 << 40,
        manifest: Manifest? = nil, onRun: @escaping @Sendable () -> Void = {},
        wrap: @escaping @Sendable (String, any GraphBackend) -> any GraphBackend = { $1 }
    ) throws -> (Krea2Pipeline, Calls) {
        let loads = Calls()
        let pipeline = try Krea2Pipeline(
            manifest: try manifest ?? Self.manifest(), device: device, filter: filter,
            options: .init(memory: .bytes(budget), gpu: .unadmitted, probe: still), encodePrompt: encode,
            load: { role throws(AetherError) in
                loads.append(role)
                let graph: any GraphBackend = switch role {
                case "textEncoder": FakeText(calls)
                case "denoiser0": FakeSegment(0, calls, onRun: onRun)
                case "denoiser1": FakeSegment(1, calls)
                case "denoiser2": FakeSegment(2, calls)
                case "denoiser3": FakeSegment(3, calls)
                default: FakeVAE(calls)
                }
                return wrap(role, graph)
            },
            fileBytes: { files[$0] ?? 0 })
        return (pipeline, loads)
    }

    /// The final packed latents the pipeline must reach, on the host, for the fakes above.
    static func expected(prompt: String, seed: UInt64, steps: Int, tokens: Int) throws -> [Float] {
        let schedule = try FlowMatchSchedule(try manifest().schedule!, steps: steps, imageTokens: tokens)
        let ids = try encode(prompt)
        // Conditioning row 0 is rendered id `drop`, at position `drop`.
        let bias = Float(Float16(Float(ids[drop]) / 100_000 + Float(drop) / 1000))
        var x = try SeededGaussianNoise.generate(seed: seed, count: tokens * 64)
        for step in 0..<steps {
            let sigma = schedule.timesteps[step]
            let v = (0..<x.count).map { i -> Float in
                let ch = i % 64
                let image = ch < carry ? Float(Float16(x[i])) + sigma : 0
                return Float(Float16(image * 0.5 + bias))
            }
            try schedule.advance(step: step, latents: &x, velocity: v)
        }
        return x
    }

    /// Check live storage at the graph boundary, rather than relying on process-wide allocator reclamation.
    final class MemoryChecked: GraphBackend {
        let base: any GraphBackend
        let role: String
        let checks: Calls
        init(_ base: any GraphBackend, role: String, checks: Calls) {
            self.base = base
            self.role = role
            self.checks = checks
        }
        var signature: GraphSignature { base.signature }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            if role == "denoiser0" {
                let bound = signature.inputs.reduce(0) { $0 + bindings.input($1.name)!.storageByteCount }
                    + signature.outputs.reduce(0) { $0 + bindings.output($1.name)!.storageByteCount }
                // In addition to the bound tensors: the second carry pair and final velocity. The third carry
                // pair shares the first's storage, and the text encoder's token/position buffers are gone.
                let extra = bindings.output("image_1")!.storageByteCount + bindings.output("text_1")!.storageByteCount
                    + bindings.input("latents")!.storageByteCount
                #expect(bindings.arena.byteCount == bound + extra)
                checks.append(role)
            }
            if role.hasPrefix("denoiser") {
                for input in signature.inputs {
                    for output in signature.outputs {
                        #expect(!bindings.input(input.name)!.isSameStorage(as: bindings.output(output.name)!))
                    }
                }
            }
            if role == "vaeDecoder" {
                let bytes = bindings.input("latents")!.storageByteCount + bindings.output("image")!.storageByteCount
                #expect(bindings.arena.byteCount == bytes, "only the decoder's own tensors survive into decode")
                checks.append(role)
            }
            try await base.run(&bindings)
        }
    }

    @Test func reusesDeadCarriesAndReleasesStepStorageBeforeDecode() async throws {
        let checks = Calls()
        let (pipeline, _) = try Self.pipeline(calls: Calls(), wrap: { role, graph in
            MemoryChecked(graph, role: role, checks: checks)
        })
        let request = ImageRequest(prompt: "a cat", seed: 7, steps: 2, width: 48, height: 32)
        let image = try await pipeline.generate(request)
        let expected = try Self.expected(prompt: request.prompt, seed: 7, steps: 2, tokens: 6)
        for ch in 0..<3 {
            #expect(image.planarRGB[ch * 48 * 32] == Float(Float16(expected[ch])))
        }
        #expect(try await pipeline.generate(request) == image)
        #expect(checks.log == ["denoiser0", "denoiser0", "vaeDecoder", "denoiser0", "denoiser0", "vaeDecoder"])
        await pipeline.unload()
    }

    /// Review finding 6: Krea exports list no fixed sizes, so `maxImageTokens` is required.
    @Test func theTokenBoundIsRequired() throws {
        var unbounded = try Self.manifest()
        unbounded.latent?.maxImageTokens = nil
        #expect(throws: AetherError.self) { _ = try Self.pipeline(calls: Calls(), manifest: unbounded) }
    }

    @Test func generatesThroughTheSegmentsWithTheHostRecipe() async throws {
        let calls = Calls(), filter = RecordingFilter()
        let (pipeline, loads) = try Self.pipeline(calls: calls, filter: filter)
        let image = try await pipeline.generate(.init(prompt: "a cat", seed: 7, steps: 2, width: 48, height: 32))
        let schedule = try FlowMatchSchedule(try Self.manifest().schedule!, steps: 2, imageTokens: 6)
        #expect(calls.log == ["text"] + schedule.timesteps.flatMap { ["segment0 \($0)", "segment1", "segment2", "segment3"] } + ["vae"])
        #expect(loads.log == ["textEncoder", "denoiser0", "denoiser1", "denoiser2", "denoiser3", "vaeDecoder"])
        #expect(filter.log == ["prompt a cat", "image 48x32 a cat"])
        #expect(image.width == 48 && image.height == 32)
        // Pixel (channel 2, y 20, x 40) is token (1, 2), channel 2, rounded to fp16 for the decoder.
        let x = try Self.expected(prompt: "a cat", seed: 7, steps: 2, tokens: 6)
        let pixel: Int = (2 * 32 + 20) * 48 + 40, latent: Int = (1 * 3 + 2) * 64 + 2
        #expect(image.planarRGB[pixel] == Float(Float16(x[latent])))
        #expect(image.planarRGB[0] == Float(Float16(x[0])))
        // Everything fit: a second generation loads nothing and repeats the image.
        #expect(try await pipeline.generate(.init(prompt: "a cat", seed: 7, steps: 2, width: 48, height: 32)) == image)
        #expect(loads.log.count == 6)
        await pipeline.unload()
        #expect(await pipeline.residentRoles.isEmpty)
    }

    /// The stage a generation's `contentRefused` names, or nil when it succeeds or fails otherwise.
    static func refusalStage(of pipeline: Krea2Pipeline, _ request: ImageRequest) async -> AetherError.GenerationFailure.ContentStage? {
        let error: (any Error)?
        do {
            _ = try await pipeline.generate(request)
            error = nil
        } catch let thrown {
            error = thrown
        }
        guard case .generation(.contentRefused(let stage, _))? = error as? AetherError else { return nil }
        return stage
    }

    /// A refused prompt stops before any graph loads; a refused image is discarded after the run.
    @Test func theContentFilterGuardsThePromptAndTheImage() async throws {
        let request = ImageRequest(prompt: "a cat", steps: 1, width: 32, height: 32)
        func refusal(_ pipeline: Krea2Pipeline) async -> AetherError.GenerationFailure.ContentStage? {
            await Self.refusalStage(of: pipeline, request)
        }
        let promptCalls = Calls()
        let (strictPrompt, promptLoads) = try Self.pipeline(calls: promptCalls, filter: RecordingFilter(refusePrompt: true))
        #expect(await refusal(strictPrompt) == .prompt)
        #expect(promptLoads.log.isEmpty && promptCalls.log.isEmpty)

        let imageCalls = Calls(), imageFilter = RecordingFilter(refuseImage: true)
        let (strictImage, _) = try Self.pipeline(calls: imageCalls, filter: imageFilter)
        #expect(await refusal(strictImage) == .output)
        #expect(imageCalls.log.last == "vae" && imageFilter.log == ["prompt a cat", "image 32x32 a cat"])

        // A filter that throws fails the generation with its error (an unreachable service blocks output).
        struct Unreachable: Error {}
        struct Throwing: ContentFilter {
            func review(prompt: String) async throws -> ContentVerdict { throw Unreachable() }
            func review(image: GeneratedImage, prompt: String) async throws -> ContentVerdict { .allow }
        }
        let (throwing, throwingLoads) = try Self.pipeline(calls: Calls(), filter: Throwing())
        await #expect(throws: Unreachable.self) { try await throwing.generate(request) }
        #expect(throwingLoads.log.isEmpty)

        // The registry's driver loads a model, not a pipeline: a pipeline is only made with a filter.
        #expect(Krea2Driver.Loaded.self == Krea2Model.self)
        #expect(Registry.image.supports(.krea2))
    }

    @Test func thePromptKeepsItsTailAndBoundsItsRows() throws {
        let manifest = try Self.manifest()
        let rule = try #require(manifest.krea2?.prompt), rows = try #require(manifest.krea2?.conditioning.rowRange)
        let short = try Krea2Pipeline.Prompt(rendered: [1, 2, 3, 4], rule: rule, rows: rows)
        #expect(short.ids == [1, 2, 3, 4] && !short.truncated && short.conditioningRows == 2)
        // 14 ids, at most 10: the first 8 and the last 2.
        let long = try Krea2Pipeline.Prompt(rendered: Array(1...14), rule: rule, rows: rows)
        #expect(long.ids == Array(1...8) + [13, 14] && long.truncated && long.conditioningRows == 8)
        // No row after the prefix, or a negative id.
        #expect(throws: AetherError.self) { try Krea2Pipeline.Prompt(rendered: [1, 2], rule: rule, rows: rows) }
        #expect(throws: AetherError.self) { try Krea2Pipeline.Prompt(rendered: [1, 2, -3], rule: rule, rows: rows) }
    }

    /// Split halves over axes 32, 48, 48 (θ 1000): token (row 2, column 3) has angle 0 on axis 0 (index 0), 2 on axis 1
    /// (index 16) and 3 on axis 2 (index 40); each cos repeats 64 later, and the sin rows follow at 128.
    @Test func theRotaryTableFollowsTheManifest() throws {
        let rotary = try #require(try Self.manifest().krea2?.rotary)
        let table = Krea2Rotary.image(rows: 3, columns: 4, rotary: rotary)
        #expect(table.count == 12 * 2 * 128)
        let token = (2 * 4 + 3) * 256
        #expect(table[token + 0] == 1 && table[token + 128] == 0)
        #expect(table[token + 16] == Float(cos(2.0)) && table[token + 16 + 64] == Float(cos(2.0)))
        #expect(table[token + 40] == Float(cos(3.0)) && table[token + 128 + 40] == Float(sin(3.0)))
        // The second pair of axis 1 turns slower: θ^(2/48).
        #expect(table[token + 17] == Float(cos(2 * (1 / pow(1000, 2.0 / 48)))))
    }

    @Test func perPhaseReleasesEachPhasesGraphs() async throws {
        // Files 10 / 80 / 10: all resident needs 100; per phase, at most 10 held + 80 = 90.
        let (pipeline, loads) = try Self.pipeline(calls: Calls(), budget: 95)
        _ = try await pipeline.generate(.init(prompt: "a cat", steps: 1, width: 32, height: 32))
        #expect(await pipeline.residentRoles.isEmpty)
        #expect(loads.log == ["textEncoder", "denoiser0", "denoiser1", "denoiser2", "denoiser3", "vaeDecoder"])
    }

    /// Third review, P1-3: per phase, a phase's graphs are destroyed when the phase ends, before the next phase's
    /// graphs run (before: the pipeline's locals kept every graph alive until the generation returned).
    @Test func perPhaseDestroysEachPhasesGraphsAtItsEnd() async throws {
        let calls = Calls()
        let (pipeline, _) = try Self.pipeline(calls: calls, budget: 95, wrap: { role, graph in
            LatentDiffusionTests.Tracked(graph, role: role, log: calls)
        })
        _ = try await pipeline.generate(.init(prompt: "a cat", steps: 1, width: 32, height: 32))
        let log = calls.log
        #expect(Array(log.prefix(2)) == ["text", "~textEncoder"])
        let vae = try #require(log.firstIndex(of: "vae"))
        let released = log[2..<vae].filter { $0.hasPrefix("~") }
        #expect(released.sorted() == ["denoiser0", "denoiser1", "denoiser2", "denoiser3"].map { "~\($0)" })
        #expect(log[2..<vae].last?.hasPrefix("~") == true, "every denoiser is destroyed before the decoder runs")
        #expect(Array(log[vae...]) == ["vae", "~vaeDecoder"])
    }

    /// Third review, P2-9: a cancellation that lands during the decoder's call (the last one) used to be ignored and
    /// the image returned; now the request fails with `CancellationError` and the residency is released.
    @Test func cancellationDuringTheLastCallIsNotASuccess() async throws {
        let (pipeline, _) = try Self.pipeline(calls: Calls(), wrap: { role, graph in
            role == "vaeDecoder"
                ? LatentDiffusionTests.Tracked(graph, role: role, log: Calls(), afterRun: LatentDiffusionTests.cancelCurrentTask)
                : graph
        })
        await #expect(throws: CancellationError.self) {
            try await pipeline.generate(.init(prompt: "a cat", steps: 1, width: 32, height: 32))
        }
        #expect(await pipeline.residentRoles.isEmpty)
    }

    @Test func refusalsComeBeforeAnyWork() async throws {
        let calls = Calls()
        // Too little memory, no size, a size off the grid, too many tokens, too many steps, a prompt with no rows.
        let (small, smallLoads) = try Self.pipeline(calls: calls, budget: 50)
        await #expect(throws: AetherError.self) { try await small.generate(.init(prompt: "a cat", width: 32, height: 32)) }
        #expect(smallLoads.log.isEmpty)
        let filter = RecordingFilter()
        let (pipeline, loads) = try Self.pipeline(calls: calls, filter: filter)
        for request in [ImageRequest(prompt: "x"), .init(prompt: "x", width: 40, height: 32),
                        .init(prompt: "x", width: 2048, height: 2048), .init(prompt: "x", steps: 51, width: 32, height: 32)] {
            await #expect(throws: AetherError.self) { try await pipeline.generate(request) }
        }
        #expect(loads.log.isEmpty && calls.log.isEmpty && filter.log.isEmpty)
        // iOS: refused when the pipeline opens.
        #expect {
            _ = try Self.pipeline(calls: calls, device: Device(platform: .iOS, architecture: "test"))
        } throws: { error in
            guard case .unsupportedDevice(let id, "ios", _)? = error as? AetherError else { return false }
            return id == "krea-2-turbo"
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
                try Krea2Pipeline(manifest: manifest, device: Self.mac, filter: RecordingFilter(), options: .init(gpu: .unadmitted),
                                  encodePrompt: Self.encode, load: { _ throws(AetherError) in FakeVAE(Calls()) }, fileBytes: { _ in 0 })
            }
        }
        // A carry from a later segment, a host-fed carry, guidance, no krea2 block, segments out of order.
        try refuses { $0.graphs["denoiser1"]?.inputs["carryImage"]?.source = .from("denoiser3.velocity") }
        try refuses { $0.graphs["denoiser2"]?.inputs["carryText"]?.source = nil }
        try refuses { $0.schedule?.guidance = .init(kind: .cfg, scale: 2) }
        try refuses { $0.krea2 = nil }
        try refuses { $0.krea2?.segments = ["denoiser1", "denoiser0", "denoiser2", "denoiser3"] }
        _ = try Krea2Pipeline(manifest: try Self.manifest(), device: Self.mac, filter: RecordingFilter(), options: .init(gpu: .unadmitted),
                              encodePrompt: Self.encode, load: { _ throws(AetherError) in FakeVAE(Calls()) }, fileBytes: { _ in 0 })
    }
}
