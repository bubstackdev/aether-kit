import AetherCore
@_spi(Testing) import AetherHost
import Foundation
import Synchronization
import Testing
@testable import AetherImage
@testable import AetherRuntime

/// A masked `krea2@1` export on fakes: denoisers with one static text length (the reference rows' upper bound) and a
/// `textMask` input (MANIFEST.md §5.5), for text-only and reference prompts alike. The fakes reduce over the text rows
/// the way attention does (a mean over the rows the mask keeps), and refuse padding that isn't zero or a mask that
/// doesn't mark the real rows, so a padded + masked render must equal the exact-length export's render bit for bit.
/// Also: reference conditioning at the same shape, conditioning reuse, the one shape admission keys on, refusals, and
/// `prepare(prompt:)`.
@Suite("Krea 2 text mask")
struct Krea2TextMaskTests {
    typealias Calls = LatentDiffusionTests.Calls
    /// The fixture's text-only rows are 1...38 and its reference rows 1...60 (Krea2ReferenceTests.manifest): the
    /// segments' one text length is 60.
    static let textRows = 38, bucket = 60

    /// Krea2ReferenceTests' manifest (text rows 1...38, reference rows 1...60), its segments taking `textMask`.
    static func manifest(references: Bool = true) throws -> Manifest {
        var manifest = try Krea2ReferenceTests.manifest(references: references)
        for i in 0..<4 { manifest.graphs["denoiser\(i)"]?.inputs["textMask"] = "text_mask" }
        return try Manifest.decode(manifest.encoded())
    }

    /// A segment computing the same thing at exact length (`length` nil: a dynamic text axis, no mask) or at a static
    /// `length` with `text_mask`: segment 0 carries the first 4 latent and text channels and emits sigma as the
    /// condition; segment 1 adds condition[0] plus the mean of text column 0 over the kept rows to the image carry;
    /// segment 2 passes the carries; segment 3 emits 0.5 · image + the mean of text column 1 over the kept rows. The
    /// mean sums the kept rows in order, so the two forms agree exactly. A masked segment throws unless the mask is
    /// 1 for a prefix and 0 after it and (segment 0) the conditioning past that prefix is zero.
    final class Segment: GraphBackend {
        let index: Int
        let length: Int?
        let signature: GraphSignature
        let calls: Calls

        init(_ index: Int, length: Int?, function: String = "segment", _ calls: Calls) {
            self.index = index
            self.length = length
            self.calls = calls
            let text = length ?? TensorSpec.dynamic, c = Krea2Tests.carry
            let rope = TensorSpec(name: "rope_image", scalarType: .float32, shape: [TensorSpec.dynamic, 2, 128])
            let mask = length.map { [TensorSpec(name: "text_mask", scalarType: .float32, shape: [$0])] } ?? []
            let carries = { (n: Int) in [TensorSpec(name: "image_\(n)", scalarType: .float32, shape: [TensorSpec.dynamic, c]),
                                         TensorSpec(name: "text_\(n)", scalarType: .float32, shape: [text, c])] }
            let cond = TensorSpec(name: "cond", scalarType: .float32, shape: [8, c])
            switch index {
            case 0:
                signature = GraphSignature(
                    function: function,
                    inputs: [TensorSpec(name: "latents", scalarType: .float16, shape: [TensorSpec.dynamic, 64]),
                             TensorSpec(name: "text_embeds", scalarType: .float16, shape: [text, Krea2Tests.columns]),
                             TensorSpec(name: "sigma", scalarType: .float32, shape: [1]), rope] + mask,
                    states: [], outputs: carries(1) + [cond])
            case 3:
                signature = GraphSignature(
                    function: function, inputs: carries(3) + [cond, rope] + mask, states: [],
                    outputs: [TensorSpec(name: "velocity", scalarType: .float16, shape: [TensorSpec.dynamic, 64])])
            default:
                signature = GraphSignature(function: function, inputs: carries(index) + [cond, rope] + mask, states: [],
                                           outputs: carries(index + 1))
            }
        }

        /// The rows the mask keeps (all of them without one); throws on a mask that isn't a 1-prefix then 0s.
        func keptRows(_ bindings: Bindings, total: Int) throws -> Int {
            guard length != nil else { return total }
            let mask = try bindings.input("text_mask")!.read(Float.self)
            let kept = mask.prefix { $0 == 1 }.count
            guard kept >= 1, mask.dropFirst(kept).allSatisfy({ $0 == 0 }) else {
                throw AetherError.tensor(name: "text_mask", reason: .unsupported("not a prefix of ones: \(mask)"))
            }
            return kept
        }

        static func mean(_ values: [Float], rows: Int, column: Int) -> Float {
            var sum: Float = 0
            for r in 0..<rows { sum += values[r * Krea2Tests.carry + column] }
            return sum / Float(rows)
        }

        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            do { try body(bindings) } catch let error as AetherError { throw error } catch { throw .runtime(underlying: error) }
        }

        private func body(_ bindings: Bindings) throws {
            let tokens = bindings.boundShape(of: "rope_image")![0], c = Krea2Tests.carry
            let textName = index == 0 ? "text_embeds" : "text_\(index)"
            let total = bindings.boundShape(of: textName)![0]
            let rows = try keptRows(bindings, total: total)
            calls.append("segment\(index) \(signature.function) text \(total) kept \(rows)")
            switch index {
            case 0:
                let sigma = try bindings.input("sigma")!.read(Float.self)[0]
                let latents = try bindings.input("latents")!.readFloats(), text = try bindings.input("text_embeds")!.readFloats()
                guard text[(rows * Krea2Tests.columns)...].allSatisfy({ $0 == 0 }) else {
                    throw AetherError.tensor(name: "text_embeds", reason: .unsupported("padding rows are not zero"))
                }
                try bindings.output("image_1")!.write(Float.self, count: tokens * c) { latents[($0 / c) * 64 + $0 % c] }
                try bindings.output("text_1")!.write(Float.self, count: total * c) { text[($0 / c) * Krea2Tests.columns + $0 % c] }
                try bindings.output("cond")!.fill(sigma)
            case 1:
                let image = try bindings.input("image_1")!.readFloats(), cond = try bindings.input("cond")!.readFloats()
                let bias = cond[0] + Self.mean(try bindings.input("text_1")!.readFloats(), rows: rows, column: 0)
                try bindings.output("image_2")!.write(image.map { $0 + bias })
                try bindings.output("text_2")!.copy(from: bindings.input("text_1")!)
            case 2:
                try bindings.output("image_3")!.copy(from: bindings.input("image_2")!)
                try bindings.output("text_3")!.copy(from: bindings.input("text_2")!)
            default:
                let image = try bindings.input("image_3")!.readFloats()
                let bias = Self.mean(try bindings.input("text_3")!.readFloats(), rows: rows, column: 1)
                try bindings.output("velocity")!.write(Float16.self, count: tokens * 64) { i in
                    let t = i / 64, ch = i % 64
                    return Float16((ch < c ? image[t * c + ch] : 0) * 0.5 + bias)
                }
            }
        }
    }

    /// `masked`: the masked manifest and segments at text length `length`; otherwise the exact-length ones.
    static func pipeline(
        masked: Bool, calls: Calls, manifest: Manifest? = nil, length: Int = bucket
    ) throws -> (Krea2Pipeline, Calls) {
        let loads = Calls()
        let manifest = try manifest ?? (masked ? Self.manifest() : Krea2ReferenceTests.manifest())
        let pipeline = try Krea2Pipeline(
            manifest: manifest, device: Krea2Tests.mac, filter: Krea2Tests.RecordingFilter(),
            options: .init(memory: .bytes(1 << 40), gpu: .unadmitted, probe: Krea2Tests.still),
            encodePrompt: Krea2ReferenceTests.encode,
            load: { role throws(AetherError) in
                loads.append(role)
                if role.hasPrefix("denoiser"), let index = Int(role.dropFirst("denoiser".count)) {
                    return Segment(index, length: masked ? length : nil, calls)
                }
                return switch role {
                case "textEncoder": Krea2ReferenceTests.PhasedText(calls)
                case "vision": Krea2ReferenceTests.FakeVision(calls)
                default: Krea2Tests.FakeVAE(calls)
                }
            },
            fileBytes: { Krea2Tests.files[$0] ?? 0 })
        return (pipeline, loads)
    }

    /// Prompts of 3, 13 and 33 conditioning rows (the encoder adds a 2-id prefix and one closing id).
    static let prompts = ["ab", String(repeating: "x", count: 12), String(repeating: "y", count: 32)]

    @Test func paddedAndMaskedRendersTheExactLengthImage() async throws {
        let maskedCalls = Calls(), exactCalls = Calls()
        let (masked, maskedLoads) = try Self.pipeline(masked: true, calls: maskedCalls)
        let (exact, _) = try Self.pipeline(masked: false, calls: exactCalls)
        for prompt in Self.prompts {
            let request = ImageRequest(prompt: prompt, seed: 11, steps: 2, width: 48, height: 32)
            let a = try await masked.generate(request, observing: nil), b = try await exact.generate(request)
            #expect(a.image == b, "\(prompt.count) characters: the masked render differs")
            let rows = try masked.prepare(prompt: prompt).conditioningRows
            #expect(a.metrics.conditioningRows == rows, "metrics report the real rows")
        }
        // Every masked call ran `segment` at the bucket, keeping each prompt's real rows; the exact export ran each
        // prompt at its own length.
        let rows = try Self.prompts.map { try masked.prepare(prompt: $0).conditioningRows }
        #expect(rows == [3, 13, 33])
        #expect(Set(maskedCalls.log.filter { $0.hasPrefix("segment") }.map { $0.split(separator: " ")[3] }) == ["\(Self.bucket)"])
        #expect(maskedCalls.log.filter { $0.hasPrefix("segment") }.allSatisfy { $0.contains(" segment ") })
        for count in rows { #expect(maskedCalls.log.contains("segment3 segment text \(Self.bucket) kept \(count)")) }
        #expect(Set(exactCalls.log.filter { $0.hasPrefix("segment0") }.map { $0.split(separator: " ")[3] }) == Set(rows.map { "\($0)" }))
        // One shape for every prompt: nothing reloads.
        #expect(maskedLoads.log == ["textEncoder", "denoiser0", "denoiser1", "denoiser2", "denoiser3", "vaeDecoder"])
    }

    /// Admission keys on the bucket: past `maximumShapes` (4) distinct row counts the exact export releases and
    /// reloads its denoisers, while the masked export keeps one shape and loads once.
    @Test func everyPromptLengthSharesOneShape() async throws {
        let prompts = (1...6).map { String(repeating: "z", count: $0) }
        let (masked, maskedLoads) = try Self.pipeline(masked: true, calls: Calls())
        let (exact, exactLoads) = try Self.pipeline(masked: false, calls: Calls())
        for prompt in prompts {
            let request = ImageRequest(prompt: prompt, seed: 5, steps: 1, width: 32, height: 32)
            #expect(try await masked.generate(request) == exact.generate(request))
        }
        #expect(maskedLoads.log.filter { $0 == "denoiser0" }.count == 1)
        #expect(exactLoads.log.filter { $0 == "denoiser0" }.count > 1)
        #expect(masked.denoiserShape(rows: 1) == Self.bucket && masked.denoiserShape(rows: 38) == Self.bucket)
        #expect(masked.denoiserShape(rows: 60) == Self.bucket && exact.denoiserShape(rows: 39) == 39)
    }

    /// Reference conditioning, short and past the text-only rows, runs at the same shape as text-only prompts (one
    /// function, nothing reloads) and renders the exact-length export's image.
    @Test func referenceRowsRunAtTheSameShape() async throws {
        let maskedCalls = Calls()
        let (masked, maskedLoads) = try Self.pipeline(masked: true, calls: maskedCalls)
        let (exact, _) = try Self.pipeline(masked: false, calls: Calls())
        _ = try await masked.generate(ImageRequest(prompt: "ab", seed: 9, steps: 2, width: 32, height: 32))
        for text in ["a cat", String(repeating: "w", count: 30)] {
            let a = try await masked.encode(prompt: text, references: [Krea2ReferenceTests.square])
            let b = try await exact.encode(prompt: text, references: [Krea2ReferenceTests.square])
            #expect(a.rows == b.rows && a.bytes == b.bytes)
            let request = ImageRequest(prompt: text, seed: 9, steps: 2, width: 32, height: 32)
            let image = try await masked.generate(request, reusing: a).image
            #expect(try await image == exact.generate(request, reusing: b).image)
            #expect(maskedCalls.log.contains("segment3 segment text \(Self.bucket) kept \(a.rows)"), "\(a.rows) rows")
        }
        #expect(try await masked.encode(prompt: String(repeating: "w", count: 30), references: [Krea2ReferenceTests.square]).rows
                > Self.textRows, "the long reference prompt is past the text-only rows")
        #expect(maskedLoads.log.filter { $0 == "denoiser0" }.count == 1, "text-only and reference renders share one load")
    }

    @Test func conditioningReuseRendersTheEncodedImage() async throws {
        let calls = Calls()
        let (pipeline, _) = try Self.pipeline(masked: true, calls: calls)
        let prompt = Self.prompts[1]
        let conditioning = try await pipeline.encode(prompt: prompt)
        #expect(conditioning.rows == 13, "conditioning holds the real rows, not the padding")
        let request = ImageRequest(prompt: prompt, seed: 4, steps: 2, width: 48, height: 32)
        let reused = try await pipeline.generate(request, reusing: conditioning)
        #expect(try await reused.image == pipeline.generate(request))
        #expect(reused.metrics.conditioningRows == 13)
    }

    /// `prepare(prompt:)` reports the real rows and truncation, as on an exact-length export.
    @Test func preparationReportsTheRealRows() throws {
        let (masked, _) = try Self.pipeline(masked: true, calls: Calls())
        let (exact, _) = try Self.pipeline(masked: false, calls: Calls())
        for prompt in Self.prompts + [String(repeating: "q", count: 60)] {
            #expect(try masked.prepare(prompt: prompt) == exact.prepare(prompt: prompt))
        }
        let long = try masked.prepare(prompt: String(repeating: "q", count: 60))
        #expect(long.truncated && long.conditioningRows == Self.textRows)
    }

    /// Rows past every function's text length (here a `step` shorter than the manifest's rows, with no `references`)
    /// fail with `invalidParameters` and release the graphs; manifests that mask some segments only, or pair the
    /// references function and the mask wrongly, are refused when the pipeline opens.
    @Test func whatDoesNotFitIsRefused() async throws {
        let manifest = try Self.manifest(references: false)
        let (short, _) = try Self.pipeline(masked: true, calls: Calls(), manifest: manifest, length: 10)
        _ = try await short.generate(ImageRequest(prompt: "abc", seed: 1, steps: 1, width: 32, height: 32))
        await #expect {
            try await short.generate(ImageRequest(prompt: String(repeating: "x", count: 20), seed: 1, steps: 1, width: 32, height: 32))
        } throws: { error in
            guard case .generation(.invalidParameters(let reason))? = error as? AetherError else { return false }
            return reason.contains("21 conditioning rows exceed this export's denoiser text length 10")
        }
        #expect(await short.residentRoles.isEmpty)

        func refuses(_ manifest: Manifest) {
            #expect(throws: AetherError.self) {
                try Krea2Pipeline(manifest: manifest, device: Krea2Tests.mac, filter: Krea2Tests.RecordingFilter(),
                                  options: .init(gpu: .unadmitted), encodePrompt: Krea2ReferenceTests.encode,
                                  load: { _ throws(AetherError) in Krea2Tests.FakeVAE(Calls()) }, fileBytes: { _ in 0 })
            }
        }
        var partial = try Self.manifest()
        partial.graphs["denoiser2"]?.inputs["textMask"] = nil
        refuses(partial)
        // A second function on a segment: the contract has no segment phase but `step`.
        var phased = try Self.manifest()
        phased.graphs["denoiser1"]?.function = nil
        phased.graphs["denoiser1"]?.functions = ["step": "segment", "references": "segmentReferences"]
        refuses(phased)
    }

    /// The exporter's masked fixture (its defaults: no reference bucket) decodes and passes the contract: `textMask` on
    /// every segment, one function each.
    @Test func theMaskedFixtureIsAContractManifest() throws {
        let url = Krea2Tests.manifestURL.deletingLastPathComponent().appending(path: "realism-masked.json")
        let manifest = try Manifest.decode(Data(contentsOf: url))
        try DriverContract.krea2.validate(manifest)
        for i in 0..<4 {
            let graph = try #require(manifest.graphs["denoiser\(i)"])
            #expect(graph.inputs["textMask"]?.name == "text_mask")
            #expect(graph.phases == ["step": "segment"])
        }
        _ = try Krea2Pipeline(manifest: manifest, device: Krea2Tests.mac, filter: Krea2Tests.RecordingFilter(),
                              options: .init(gpu: .unadmitted), encodePrompt: Krea2Tests.encode,
                              load: { _ throws(AetherError) in Krea2Tests.FakeVAE(Calls()) }, fileBytes: { _ in 0 })
    }

    // MARK: reference conditioning beside resident denoisers

    /// Metal as Core AI spends it: a graph's bytes from its load until it's released; a function of a multi-function
    /// file takes its own copy at its first run.
    final class Metal: @unchecked Sendable {
        let level = Mutex<Int64>(0), peak = Mutex<Int64>(0)
        func add(_ bytes: Int64) {
            let now = level.withLock { $0 += bytes; return $0 }
            peak.withLock { $0 = max($0, now) }
        }
        var probe: GPUMemoryProbe { GPUMemoryProbe { [self] in level.withLock { $0 } } }
    }

    /// `inner` holding `bytes` while loaded.
    final class Charged: PhasedGraphBackend {
        let inner: any GraphBackend, metal: Metal, bytes: Int64
        init(_ inner: any GraphBackend, metal: Metal, bytes: Int64) {
            self.inner = inner; self.metal = metal; self.bytes = bytes
            metal.add(bytes)
        }
        deinit { metal.add(-bytes) }
        var signature: GraphSignature { inner.signature }
        func run(_ bindings: inout Bindings) async throws(AetherError) { try await inner.run(&bindings) }
        func function(forPhase phase: String) -> (any GraphBackend)? { (inner as? any PhasedGraphBackend)?.function(forPhase: phase) }
    }

    /// The text encoder file: `encode` resident at load, `encodeWithImages` taking another copy at its first run.
    final class ChargedText: PhasedGraphBackend {
        let step: Charged, metal: Metal, copy: Int64
        let references: any GraphBackend
        private var copied = false
        init(_ calls: Calls, metal: Metal, bytes: Int64) {
            let text = Krea2ReferenceTests.PhasedText(calls)
            step = Charged(text.step, metal: metal, bytes: bytes)
            references = text.references
            self.metal = metal
            copy = bytes
        }
        deinit { if copied { metal.add(-copy) } }
        var signature: GraphSignature { step.signature }
        func run(_ bindings: inout Bindings) async throws(AetherError) { try await step.run(&bindings) }
        func function(forPhase phase: String) -> (any GraphBackend)? {
            guard phase == "references" else { return nil }
            if !copied { copied = true; metal.add(copy) }
            return references
        }
    }

    /// Text-only, then reference, then text-only renders on a budget that holds every graph with one text-encoder copy
    /// but not with the references function's copy beside it (text encoder 10, its references copy 10, vision 2,
    /// segments 20 each, decoder 10; budget 115). The reference encode releases the text-only encoder first and its own
    /// graphs after, so it fits; the denoisers and the decoder stay loaded throughout, and only the text encoder and the
    /// vision tower load again.
    @Test func referenceConditioningLoadsBesideTheResidentDenoisers() async throws {
        let metal = Metal(), calls = Calls(), loads = Calls()
        let budget: Int64 = 115
        let pipeline = try Krea2Pipeline(
            manifest: try Self.manifest(), device: Krea2Tests.mac, filter: Krea2Tests.RecordingFilter(),
            options: .init(memory: .bytes(budget), gpu: .unadmitted, probe: metal.probe),
            encodePrompt: Krea2ReferenceTests.encode,
            load: { role throws(AetherError) in
                loads.append(role)
                if role.hasPrefix("denoiser"), let index = Int(role.dropFirst("denoiser".count)) {
                    return Charged(Segment(index, length: Self.bucket, calls), metal: metal, bytes: 20)
                }
                switch role {
                case "textEncoder": return ChargedText(calls, metal: metal, bytes: 10)
                case "vision": return Charged(Krea2ReferenceTests.FakeVision(calls), metal: metal, bytes: 2)
                default: return Charged(Krea2Tests.FakeVAE(calls), metal: metal, bytes: 10)
                }
            },
            fileBytes: { ["textEncoder": 10, "vision": 2, "vaeDecoder": 10, "denoiser0": 20, "denoiser1": 20,
                          "denoiser2": 20, "denoiser3": 20][$0] ?? 0 })
        func text(_ prompt: String) async throws {
            _ = try await pipeline.generate(ImageRequest(prompt: prompt, seed: 2, steps: 1, width: 32, height: 32))
        }
        let long = String(repeating: "w", count: 30)
        try await text("ab")
        try await text("abcdefg")
        let conditioning = try await pipeline.encode(prompt: long, references: [Krea2ReferenceTests.square])
        let afterEncode = await pipeline.residentRoles
        #expect(!afterEncode.contains("vision") && !afterEncode.contains("textEncoder"),
                "the reference conditioning graphs are released after the encode: \(afterEncode)")
        _ = try await pipeline.generate(ImageRequest(prompt: long, seed: 2, steps: 1, width: 32, height: 32), reusing: conditioning)
        try await text("abc")
        let count = { (role: String) in loads.log.filter { $0 == role }.count }
        #expect(count("denoiser0") == 1 && count("denoiser3") == 1 && count("vaeDecoder") == 1, "\(loads.log)")
        #expect(count("textEncoder") == 3 && count("vision") == 1, "\(loads.log)")
        #expect(metal.peak.withLock { $0 } <= budget, "peak \(metal.peak.withLock { $0 }) over \(budget)")
    }
}
