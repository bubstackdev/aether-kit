import AetherCore
@_spi(Testing) import AetherHost
import Foundation
import Synchronization
import Testing
@testable import AetherImage
@testable import AetherRuntime

/// A masked `krea2@1` export on fakes: denoisers with a static text length and a `textMask` input (MANIFEST.md §5.5),
/// `step` at the text-only bucket and `references` at the reference bucket. The fakes reduce over the text rows the
/// way attention does (a mean over the rows the mask keeps), and refuse padding that isn't zero or a mask that doesn't
/// mark the real rows, so a padded + masked render must equal the exact-length export's render bit for bit. Also: the
/// reference bucket, conditioning reuse, the shape admission keys on, refusals, and `prepare(prompt:)`.
@Suite("Krea 2 text mask")
struct Krea2TextMaskTests {
    typealias Calls = LatentDiffusionTests.Calls
    /// The fixture's text-only rows are 1...38 and its reference rows 1...60 (Krea2ReferenceTests.manifest).
    static let textBucket = 38, referenceBucket = 60

    /// Krea2ReferenceTests' manifest (text rows 1...38, reference rows 1...60), its segments taking `textMask` and,
    /// with `phased`, a `references` function each.
    static func manifest(phased: Bool = true, references: Bool = true) throws -> Manifest {
        var manifest = try Krea2ReferenceTests.manifest(references: references)
        for i in 0..<4 {
            var graph = manifest.graphs["denoiser\(i)"]!
            graph.inputs["textMask"] = "text_mask"
            if phased {
                graph.function = nil
                graph.functions = ["step": "segment", "references": "segmentReferences"]
            }
            manifest.graphs["denoiser\(i)"] = graph
        }
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

    /// A masked segment's two functions, as `GraphResidency` loads a multi-function file.
    static func phased(_ index: Int, step: Int, references: Int?, _ calls: Calls) throws(AetherError) -> any GraphBackend {
        var functions: [String: any GraphBackend] = ["step": Segment(index, length: step, calls)]
        if let references { functions["references"] = Segment(index, length: references, function: "segmentReferences", calls) }
        return try GraphFunctionSet(functions: functions)
    }

    /// `masked`: the masked manifest and segments (`step` and `references` lengths); otherwise the exact-length ones.
    static func pipeline(
        masked: Bool, calls: Calls, manifest: Manifest? = nil, step: Int = textBucket, references: Int? = referenceBucket
    ) throws -> (Krea2Pipeline, Calls) {
        let loads = Calls()
        let manifest = try manifest ?? (masked ? Self.manifest(phased: references != nil) : Krea2ReferenceTests.manifest())
        let pipeline = try Krea2Pipeline(
            manifest: manifest, device: Krea2Tests.mac, filter: Krea2Tests.RecordingFilter(),
            options: .init(memory: .bytes(1 << 40), gpu: .unadmitted, probe: Krea2Tests.still),
            encodePrompt: Krea2ReferenceTests.encode,
            load: { role throws(AetherError) in
                loads.append(role)
                if role.hasPrefix("denoiser"), let index = Int(role.dropFirst("denoiser".count)) {
                    return masked ? try phased(index, step: step, references: references, calls) : Segment(index, length: nil, calls)
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
        #expect(Set(maskedCalls.log.filter { $0.hasPrefix("segment") }.map { $0.split(separator: " ")[3] }) == ["\(Self.textBucket)"])
        #expect(maskedCalls.log.filter { $0.hasPrefix("segment") }.allSatisfy { $0.contains(" segment ") })
        for count in rows { #expect(maskedCalls.log.contains("segment3 segment text \(Self.textBucket) kept \(count)")) }
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
        #expect(masked.denoiserShape(rows: 1) == Self.textBucket && masked.denoiserShape(rows: 38) == Self.textBucket)
        #expect(masked.denoiserShape(rows: 39) == Self.referenceBucket && exact.denoiserShape(rows: 39) == 39)
    }

    /// Reference conditioning past the text-only bucket runs every segment's `references` function at its length;
    /// shorter reference conditioning stays on `step`. Both render the exact-length export's image.
    @Test func referenceRowsRunTheReferencesBucket() async throws {
        let maskedCalls = Calls()
        let (masked, _) = try Self.pipeline(masked: true, calls: maskedCalls)
        let (exact, _) = try Self.pipeline(masked: false, calls: Calls())
        for text in ["a cat", String(repeating: "w", count: 30)] {
            let a = try await masked.encode(prompt: text, references: [Krea2ReferenceTests.square])
            let b = try await exact.encode(prompt: text, references: [Krea2ReferenceTests.square])
            #expect(a.rows == b.rows && a.bytes == b.bytes)
            let request = ImageRequest(prompt: text, seed: 9, steps: 2, width: 32, height: 32)
            let image = try await masked.generate(request, reusing: a).image
            #expect(try await image == exact.generate(request, reusing: b).image)
            let function = a.rows > Self.textBucket ? "segmentReferences text \(Self.referenceBucket)" : "segment text \(Self.textBucket)"
            #expect(maskedCalls.log.contains("segment3 \(function) kept \(a.rows)"), "\(a.rows) rows: \(maskedCalls.log.suffix(4))")
        }
        #expect(maskedCalls.log.contains { $0.hasPrefix("segment0 segmentReferences") })
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
        #expect(long.truncated && long.conditioningRows == Self.textBucket)
    }

    /// Rows past every function's text length (here a `step` shorter than the manifest's rows, with no `references`)
    /// fail with `invalidParameters` and release the graphs; manifests that mask some segments only, or pair the
    /// references function and the mask wrongly, are refused when the pipeline opens.
    @Test func whatDoesNotFitIsRefused() async throws {
        let manifest = try Self.manifest(phased: false, references: false)
        let (short, _) = try Self.pipeline(masked: true, calls: Calls(), manifest: manifest, step: 10, references: nil)
        _ = try await short.generate(ImageRequest(prompt: "abc", seed: 1, steps: 1, width: 32, height: 32))
        await #expect {
            try await short.generate(ImageRequest(prompt: String(repeating: "x", count: 20), seed: 1, steps: 1, width: 32, height: 32))
        } throws: { error in
            guard case .generation(.invalidParameters(let reason))? = error as? AetherError else { return false }
            return reason.contains("21 conditioning rows")
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
        // A references function without the mask.
        var unmasked = try Self.manifest()
        for i in 0..<4 { unmasked.graphs["denoiser\(i)"]?.inputs["textMask"] = nil }
        refuses(unmasked)
        // A references function without krea2.references.
        refuses(try Self.manifest(phased: true, references: false))
        // One segment with other functions.
        var mixed = try Self.manifest()
        mixed.graphs["denoiser1"]?.functions = nil
        mixed.graphs["denoiser1"]?.function = "segment"
        refuses(mixed)
    }

    /// A masked export with reference images but no `references` function (one bucket): reference conditioning that
    /// fits the text bucket renders; longer reference conditioning is refused with `invalidParameters`.
    @Test func withoutTheReferenceBucketLongReferenceRowsAreRefused() async throws {
        let manifest = try Self.manifest(phased: false, references: true)
        let (pipeline, _) = try Self.pipeline(masked: true, calls: Calls(), manifest: manifest, references: nil)
        let short = try await pipeline.encode(prompt: "a cat", references: [Krea2ReferenceTests.square])
        _ = try await pipeline.generate(ImageRequest(prompt: "a cat", seed: 1, steps: 1, width: 32, height: 32), reusing: short)
        let long = String(repeating: "w", count: 30)
        let conditioning = try await pipeline.encode(prompt: long, references: [Krea2ReferenceTests.square])
        await #expect {
            try await pipeline.generate(ImageRequest(prompt: long, seed: 1, steps: 1, width: 32, height: 32), reusing: conditioning)
        } throws: { error in
            guard case .generation(.invalidParameters(let reason))? = error as? AetherError else { return false }
            return reason.contains("\(conditioning.rows) conditioning rows")
        }
    }

    /// The exporter's masked fixture decodes and passes the contract: `textMask` on every segment, a `references`
    /// function each.
    @Test func theMaskedFixtureIsAContractManifest() throws {
        let url = Krea2Tests.manifestURL.deletingLastPathComponent().appending(path: "realism-masked.json")
        let manifest = try Manifest.decode(Data(contentsOf: url))
        try DriverContract.krea2.validate(manifest)
        for i in 0..<4 {
            let graph = try #require(manifest.graphs["denoiser\(i)"])
            #expect(graph.inputs["textMask"]?.name == "text_mask")
            #expect(graph.phases == ["step": "segment", "references": "segmentReferences"])
        }
        _ = try Krea2Pipeline(manifest: manifest, device: Krea2Tests.mac, filter: Krea2Tests.RecordingFilter(),
                              options: .init(gpu: .unadmitted), encodePrompt: Krea2Tests.encode,
                              load: { _ throws(AetherError) in Krea2Tests.FakeVAE(Calls()) }, fileBytes: { _ in 0 })
    }

    // MARK: the references bucket's own footprint

    /// Metal as Core AI spends it on a two-function file: nothing at load, each function's own copy of the weights at
    /// its first run (measured 2026-11-09: 7.08 GB per segment after `segment`, 13.85 GB after `segmentReferences` too).
    final class Metal: @unchecked Sendable {
        let level = Mutex<Int64>(0)
        let peak = Mutex<Int64>(0)
        func add(_ bytes: Int64) {
            let now = level.withLock { $0 += bytes; return $0 }
            peak.withLock { $0 = max($0, now) }
        }
        var probe: GPUMemoryProbe { GPUMemoryProbe { [self] in level.withLock { $0 } } }
    }

    /// A segment function that takes `bytes` at its first run and gives them back when released.
    final class Charged: GraphBackend {
        let inner: Segment, metal: Metal, bytes: Int64
        private(set) var charged = false
        init(_ inner: Segment, metal: Metal, bytes: Int64) { self.inner = inner; self.metal = metal; self.bytes = bytes }
        deinit { if charged { metal.add(-bytes) } }
        var signature: GraphSignature { inner.signature }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            if !charged { charged = true; metal.add(bytes) }
            try await inner.run(&bindings)
        }
    }

    /// A loaded segment file: both functions, recording which ever ran.
    final class ChargedSet: PhasedGraphBackend {
        let step: Charged, references: Charged
        init(step: Charged, references: Charged) { self.step = step; self.references = references }
        var signature: GraphSignature { step.signature }
        func run(_ bindings: inout Bindings) async throws(AetherError) { try await step.run(&bindings) }
        func function(forPhase phase: String) -> (any GraphBackend)? { phase == "references" ? references : nil }
        var ranBoth: Bool { step.charged && references.charged }
    }

    /// Text-only and reference renders alternating on a budget that holds every graph with one function's weights but
    /// not both: each switch releases the graphs and loads them for the other bucket, never holding both functions'
    /// weights, and renders in a row at one bucket load nothing.
    @Test func switchingBucketsSwapsTheFunctionsAndNeverHoldsBoth() async throws {
        let metal = Metal(), calls = Calls(), loads = Calls()
        /// Weak: a released segment must give its memory back. Each entry notes whether that set ran both functions.
        final class Sets: @unchecked Sendable {
            struct Weak { weak var set: ChargedSet? }
            var all: [Weak] = []
            var ranBoth = false
            func note() { ranBoth = ranBoth || all.contains { $0.set?.ranBoth == true } }
        }
        let sets = Sets()
        // Weights: text encoder 10 and decoder 10 at load, each segment function 20 at its first run. One bucket with
        // every graph: 100; both buckets' segments: 180. Budget 150.
        let budget: Int64 = 150
        let pipeline = try Krea2Pipeline(
            manifest: try Self.manifest(), device: Krea2Tests.mac, filter: Krea2Tests.RecordingFilter(),
            options: .init(memory: .bytes(budget), gpu: .unadmitted, probe: metal.probe),
            encodePrompt: Krea2ReferenceTests.encode,
            load: { role throws(AetherError) in
                loads.append(role)
                if role.hasPrefix("denoiser"), let index = Int(role.dropFirst("denoiser".count)) {
                    let set = ChargedSet(
                        step: Charged(Segment(index, length: Self.textBucket, calls), metal: metal, bytes: 20),
                        references: Charged(Segment(index, length: Self.referenceBucket, function: "segmentReferences", calls),
                                            metal: metal, bytes: 20))
                    sets.all.append(.init(set: set))
                    return set
                }
                switch role {
                case "textEncoder": metal.add(10); return Released(Krea2ReferenceTests.PhasedText(calls), metal: metal, bytes: 10)
                case "vision": return Krea2ReferenceTests.FakeVision(calls)
                default: metal.add(10); return Released(Krea2Tests.FakeVAE(calls), metal: metal, bytes: 10)
                }
            },
            fileBytes: { ["textEncoder": 10, "vaeDecoder": 10, "denoiser0": 20, "denoiser1": 20, "denoiser2": 20, "denoiser3": 20][$0] ?? 0 })
        let long = String(repeating: "w", count: 30)
        func text(_ prompt: String) async throws {
            _ = try await pipeline.generate(ImageRequest(prompt: prompt, seed: 2, steps: 1, width: 32, height: 32))
            sets.note()
        }
        func reference() async throws {
            let conditioning = try await pipeline.encode(prompt: long, references: [Krea2ReferenceTests.square])
            #expect(conditioning.rows > Self.textBucket)
            _ = try await pipeline.generate(ImageRequest(prompt: long, seed: 2, steps: 1, width: 32, height: 32), reusing: conditioning)
            sets.note()
        }
        func denoiserLoads() -> Int { loads.log.filter { $0 == "denoiser0" }.count }
        try await text("ab")
        try await text("abcdefg")
        #expect(denoiserLoads() == 1, "text-only renders in a row keep the denoisers")
        try await reference()
        #expect(denoiserLoads() == 2, "the reference bucket loads its own copy")
        try await reference()
        #expect(denoiserLoads() == 2, "reference renders in a row keep it")
        try await text("abc")
        #expect(denoiserLoads() == 3, "back to the text bucket")
        #expect(!sets.ranBoth, "no loaded segment ran both functions")
        #expect(metal.peak.withLock { $0 } <= budget, "peak \(metal.peak.withLock { $0 }) over \(budget)")
    }

    /// A graph that gives back `bytes` when released.
    final class Released: PhasedGraphBackend {
        let inner: any GraphBackend, metal: Metal, bytes: Int64
        init(_ inner: any GraphBackend, metal: Metal, bytes: Int64) { self.inner = inner; self.metal = metal; self.bytes = bytes }
        deinit { metal.add(-bytes) }
        var signature: GraphSignature { inner.signature }
        func run(_ bindings: inout Bindings) async throws(AetherError) { try await inner.run(&bindings) }
        func function(forPhase phase: String) -> (any GraphBackend)? { (inner as? any PhasedGraphBackend)?.function(forPhase: phase) }
    }
}
