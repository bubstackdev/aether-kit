import AetherCore
@_spi(Testing) import AetherHost
import Foundation
import Synchronization
import Testing
@testable import AetherImage
@testable import AetherRuntime

/// Reference images through the pipeline on fakes: a static-capacity vision graph and a two-function text encoder
/// (`step` and `references`), on the Krea 2 fixture manifest with a small `krea2.references` block. Checks what the host
/// feeds the references function (extension ids, M-RoPE, slot-indexed embeddings and deepstack levels), the
/// conditioning it returns, and that it renders through `generate(_:reusing:)`. The real graphs are the gate's.
@Suite("Krea 2 reference images")
struct Krea2ReferenceTests {
    typealias Calls = LatentDiffusionTests.Calls
    static let pad: Int32 = 900, start: Int32 = 901, end: Int32 = 902, base = 1000
    static let capacity = 16, slots = 8, hidden = 3, levels = 2

    /// Like Krea2Tests' encoder, with the vision markers as single ids.
    static let encode: Krea2Pipeline.PromptEncoder = { text in
        var ids: [Int32] = [100, 101]
        var rest = Substring(text)
        let specials: [(String, Int32)] = [("<|image_pad|>", pad), ("<|vision_start|>", start), ("<|vision_end|>", end)]
        while let c = rest.first {
            if let (marker, id) = specials.first(where: { rest.hasPrefix($0.0) }) {
                ids.append(id)
                rest = rest.dropFirst(marker.count)
            } else {
                ids.append(Int32(c.unicodeScalars.first!.value))
                rest = rest.dropFirst()
            }
        }
        return ids + [200]
    }

    static func manifest(references: Bool = true) throws -> Manifest {
        var manifest = try Krea2Tests.manifest()
        manifest.krea2?.prompt.maxTokens = 40
        manifest.krea2?.conditioning.rows = "1...38"
        guard references else { return try Manifest.decode(manifest.encoded()) }
        var text = manifest.graphs["textEncoder"]!
        text.function = nil
        text.functions = ["step": "encode", "references": "encodeWithImages"]
        text.inputsByFunction = ["references": [
            "tokens": "ref_ids", "ropePositions": "rope", "imageEmbeds": "image_embeds", "deepstackEmbeds": "deepstack"]]
        manifest.graphs["textEncoder"] = text
        var vision = manifest.graphs["vaeDecoder"]!
        vision.file = .single(AssetPath(asset: "main", path: "krea2_turbo-vision.aimodel"))
        vision.function = "encode"
        vision.inputs = ["patches": "patches", "grid": "grid"]
        vision.outputs = ["imageEmbeds": "image_embeds", "deepstackEmbeds": "deepstack"]
        vision.residency = .perPrompt
        manifest.graphs["vision"] = vision
        manifest.krea2?.references = try JSONDecoder().decode(Manifest.Krea2.References.self, from: Data("""
            {"maxImages": 2, "maxSide": 8, "maxImageTokens": \(slots), "label": "P{n}:",
             "tokens": {"pad": \(pad), "start": \(start), "end": \(end)}, "extensionBase": \(base), "rows": "1...60",
             "positions": "mrope", "deepstackLevels": \(levels), "vision": "vision", "function": "references",
             "preprocess": {"fit": "longSideRounded", "maxSide": 8, "factor": 4, "minPixels": 16, "maxPixels": 4096,
                            "maxAspect": 200, "filter": "keysBicubic-0.5-antialias", "mean": 127.5, "std": 127.5,
                            "patch": 2, "temporalPatch": 2, "merge": 2, "order": "mergeWindow C,T,py,px"}}
            """.utf8))
        return try Manifest.decode(manifest.encoded())
    }

    /// Vision: per merged token t, embeds[t][k] = the first value of its window's first patch + k / 10 + rows / 100;
    /// deepstack[l][t][k] = embeds[t][k] + 100 · (l + 1). Rows past the image's tokens are -7 (never read).
    final class FakeVision: GraphBackend {
        let signature = GraphSignature(
            function: "encode",
            inputs: [TensorSpec(name: "patches", scalarType: .float16, shape: [capacity, 24]),
                     TensorSpec(name: "grid", scalarType: .int32, shape: [2])],
            states: [],
            outputs: [TensorSpec(name: "image_embeds", scalarType: .float32, shape: [capacity / 4, hidden]),
                      TensorSpec(name: "deepstack", scalarType: .float32, shape: [levels, capacity / 4, hidden])])
        let calls: Calls
        init(_ calls: Calls) { self.calls = calls }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            let grid = try bindings.input("grid")!.read(Int32.self), patches = try bindings.input("patches")!.readFloats()
            calls.append("vision \(grid[0])x\(grid[1])")
            let tokens = Int(grid[0] / 2 * (grid[1] / 2)), rows = capacity / 4
            let embeds = (0..<(rows * hidden)).map { i -> Float in
                let t = i / hidden, k = i % hidden
                return t < tokens ? patches[t * 4 * 24] + Float(k) / 10 + Float(grid[0]) / 100 : -7
            }
            try bindings.output("image_embeds")!.write(embeds)
            try bindings.output("deepstack")!.write((0..<(levels * rows * hidden)).map { embeds[$0 % (rows * hidden)] + Float(100 * ($0 / (rows * hidden) + 1)) })
        }
    }

    /// References: conditioning row r (sequence row i = drop + r) in every column, fp16: for an image row (id ≥ base)
    /// imageEmbeds[slot][1] + deepstack[1][slot][2] / 1000, else id / 1e5; plus rope[1][i] / 1e3 + rope[2][i] / 1e4.
    final class FakeReferenceText: GraphBackend {
        let signature = GraphSignature(
            function: "encodeWithImages",
            inputs: [TensorSpec(name: "ref_ids", scalarType: .int32, shape: [1, TensorSpec.dynamic]),
                     TensorSpec(name: "rope", scalarType: .int32, shape: [3, TensorSpec.dynamic]),
                     TensorSpec(name: "image_embeds", scalarType: .float32, shape: [slots, hidden]),
                     TensorSpec(name: "deepstack", scalarType: .float32, shape: [levels, slots, hidden])],
            states: [], outputs: [TensorSpec(name: "text_embeds", scalarType: .float16, shape: [TensorSpec.dynamic, Krea2Tests.columns])])
        let calls: Calls
        init(_ calls: Calls) { self.calls = calls }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            calls.append("references")
            let ids = try bindings.input("ref_ids")!.read(Int32.self), rope = try bindings.input("rope")!.read(Int32.self)
            let embeds = try bindings.input("image_embeds")!.readFloats(), deep = try bindings.input("deepstack")!.readFloats()
            let n = ids.count, rows = n - Krea2Tests.drop
            try bindings.output("text_embeds")!.write(Float16.self, count: rows * Krea2Tests.columns) { j in
                let i = j / Krea2Tests.columns + Krea2Tests.drop
                return Float16(Krea2ReferenceTests.value(id: ids[i], rope1: rope[n + i], rope2: rope[2 * n + i], embeds: embeds, deepstack: deep))
            }
        }
    }

    static func value(id: Int32, rope1: Int32, rope2: Int32, embeds: [Float], deepstack: [Float]) -> Float {
        let base: Float
        if Int(id) >= Self.base {
            let slot = Int(id) - Self.base
            base = embeds[slot * hidden + 1] + deepstack[(slots + slot) * hidden + 2] / 1000
        } else {
            base = Float(id) / 100_000
        }
        return base + Float(rope1) / 1000 + Float(rope2) / 10_000
    }

    final class PhasedText: PhasedGraphBackend {
        let step: any GraphBackend, references: any GraphBackend
        init(_ calls: Calls) {
            step = Krea2Tests.FakeText(calls)
            references = FakeReferenceText(calls)
        }
        var signature: GraphSignature { step.signature }
        func run(_ bindings: inout Bindings) async throws(AetherError) { try await step.run(&bindings) }
        func function(forPhase phase: String) -> (any GraphBackend)? { phase == "references" ? references : nil }
    }

    static func pipeline(
        calls: Calls, filter: any ContentFilter = Krea2Tests.RecordingFilter(), manifest: Manifest? = nil
    ) throws -> (Krea2Pipeline, Calls) {
        let loads = Calls()
        let pipeline = try Krea2Pipeline(
            manifest: try manifest ?? Self.manifest(), device: Krea2Tests.mac, filter: filter,
            options: .init(memory: .bytes(1 << 40), gpu: .unadmitted, probe: Krea2Tests.still), encodePrompt: encode,
            load: { role throws(AetherError) in
                loads.append(role)
                return switch role {
                case "textEncoder": PhasedText(calls)
                case "vision": FakeVision(calls)
                case "denoiser0": Krea2Tests.FakeSegment(0, calls)
                case "denoiser1": Krea2Tests.FakeSegment(1, calls)
                case "denoiser2": Krea2Tests.FakeSegment(2, calls)
                case "denoiser3": Krea2Tests.FakeSegment(3, calls)
                default: Krea2Tests.FakeVAE(calls)
                }
            },
            fileBytes: { Krea2Tests.files[$0] ?? 0 })
        return (pipeline, loads)
    }

    /// An 8x8 image (4 tokens) and an 8x4 one (2 tokens), with distinct bytes.
    static let square = try! Krea2ReferenceImage(rgb: (0..<(8 * 8 * 3)).map { UInt8($0 % 251) }, width: 8, height: 8)
    static let wide = try! Krea2ReferenceImage(rgb: (0..<(8 * 4 * 3)).map { UInt8(($0 * 7) % 251) }, width: 8, height: 4)

    @Test func theReferencesFunctionGetsSlotsRopeAndDeepstack() async throws {
        let calls = Calls(), filter = Krea2Tests.RecordingFilter()
        let (pipeline, loads) = try Self.pipeline(calls: calls, filter: filter)
        #expect(pipeline.supportsReferences)
        let prepared = try pipeline.prepare(prompt: "a cat", references: [Self.square, Self.wide])
        let conditioning = try await pipeline.encode(prompt: "a cat", references: [Self.square, Self.wide])
        #expect(conditioning.rows == prepared.conditioningRows && conditioning.referenceDigests == [Self.square.digest, Self.wide.digest])
        #expect(calls.log == ["vision 4x4", "vision 2x4", "references"] && loads.log == ["vision", "textEncoder"])
        #expect(filter.log == ["prompt a cat"])

        // The host's view: the same prompt, patches and fake vision outputs, through the fake's formula.
        let built = try pipeline.referencePrompt("a cat", references: [Self.square, Self.wide])
        #expect(built.prompt.spans.map(\.range.count) == [4, 2])
        var embeds = [Float](repeating: 0, count: Self.slots * Self.hidden)
        var deep = [Float](repeating: 0, count: Self.levels * Self.slots * Self.hidden)
        var slot = 0
        for patches in built.patches {
            for t in 0..<patches.tokens {
                for k in 0..<Self.hidden {
                    let e = Float(Float16(patches.values[t * 4 * 24])) + Float(k) / 10 + Float(patches.rows) / 100
                    embeds[slot * Self.hidden + k] = e
                    for l in 0..<Self.levels { deep[(l * Self.slots + slot) * Self.hidden + k] = e + Float(100 * (l + 1)) }
                }
                slot += 1
            }
        }
        let n = built.prompt.encoderIDs.count, rope = built.prompt.ropePositions
        let expected = (Krea2Tests.drop..<n).map { i in
            Float(Float16(Self.value(id: built.prompt.encoderIDs[i], rope1: rope[n + i], rope2: rope[2 * n + i], embeds: embeds, deepstack: deep)))
        }
        let got = conditioning.bytes.withUnsafeBytes { raw in
            (0..<conditioning.rows).map { r in Float(raw.loadUnaligned(fromByteOffset: 2 * r * Krea2Tests.columns, as: Float16.self)) }
        }
        #expect(got == expected)
    }

    @Test func referenceConditioningRendersThroughReuse() async throws {
        let calls = Calls()
        let (pipeline, loads) = try Self.pipeline(calls: calls)
        let conditioning = try await pipeline.encode(prompt: "a cat", references: [Self.square])
        let generation = try await pipeline.generate(
            ImageRequest(prompt: "a cat", seed: 3, steps: 2, width: 32, height: 32), reusing: conditioning)
        #expect(generation.metrics.conditioningRows == conditioning.rows)
        #expect(!loads.log.dropFirst(2).contains("textEncoder") && !loads.log.dropFirst(2).contains("vision"))
        #expect(calls.log.filter { $0 == "text" }.isEmpty)
    }

    @Test func requestsOutsideTheContractAreRefusedBeforeAnyGraph() async throws {
        let (pipeline, loads) = try Self.pipeline(calls: Calls())
        await #expect(throws: AetherError.self) { _ = try await pipeline.encode(prompt: "a cat", references: []) }
        await #expect(throws: AetherError.self) {
            _ = try await pipeline.encode(prompt: "a cat", references: [Self.square, Self.square, Self.wide])
        }
        await #expect(throws: AetherError.self) {  // 4 + 4 + … tokens over 8
            _ = try await pipeline.encode(prompt: "a cat", references: [Self.square, Self.square, Self.square])
        }
        #expect(loads.log.isEmpty)

        let (plain, plainLoads) = try Self.pipeline(calls: Calls(), manifest: try Self.manifest(references: false))
        #expect(!plain.supportsReferences)
        #expect(throws: AetherError.self) { _ = try plain.prepare(prompt: "a cat", references: [Self.square]) }
        await #expect(throws: AetherError.self) { _ = try await plain.encode(prompt: "a cat", references: [Self.square]) }
        #expect(plainLoads.log.isEmpty)
    }
}
