public import AetherCore
public import AetherRuntime
public import AetherText
public import CoreGraphics
import Foundation
import Synchronization
@_spi(Testing) import AetherHost

/// The rotary positions a ``FakeVLMDecoder`` computed for every position it fed, shared with the test.
public final class RopeLog: Sendable {
    private let storage = Mutex<[Int: [Int32]]>([:])
    public init() {}
    func record(_ position: Int, _ thw: [Int32]) { storage.withLock { $0[position] = thw } }
    /// (t, h, w) per position, for positions 0..<count.
    public func positions(_ count: Int) -> [[Int32]] { storage.withLock { s in (0..<count).map { s[$0] ?? [] } } }
}

/// Holds the fake tower's first `holds` calls until ``open()`` (``entered`` yields as each one waits), and fails
/// every call while ``failing`` is set.
public final class TowerGate: Sendable {
    private let holds: Mutex<Int>
    private let fail = Mutex(false)
    public let entered: AsyncStream<Void>
    private let enteredContinuation: AsyncStream<Void>.Continuation
    private let released: AsyncStream<Void>
    private let releaseContinuation: AsyncStream<Void>.Continuation

    public init(holds: Int = 0) {
        self.holds = Mutex(holds)
        (entered, enteredContinuation) = AsyncStream.makeStream()
        (released, releaseContinuation) = AsyncStream.makeStream()
    }

    public var failing: Bool {
        get { fail.withLock { $0 } }
        set { fail.withLock { $0 = newValue } }
    }

    /// Lets one held call through.
    public func open() { releaseContinuation.yield() }

    func pass() async throws(AetherError) {
        let hold = holds.withLock { n in
            guard n > 0 else { return false }
            n -= 1
            return true
        }
        if hold {
            enteredContinuation.yield()
            for await _ in released { break }
        }
        if failing { throw .unsupportedOperation("the fake tower failed (TowerGate.failing)") }
    }
}

/// A toy `vlm-static-inject@1` model behind ``GraphBackend``: a vision tower and a decoder that reads its static
/// image inputs by the `qwen3vl-mrope` rule (docs/VLM.md §3.1), small enough for CPU tests.
///
/// Tokens: ``vocab`` head rows; image tokens are `vocab + slot`. Per fed token at position p the decoder computes a
/// feature: a text id's is (id, its rope position p − Σ amount_j [p ≥ start_j]); an image token's is its slot's rows
/// (``imageEmbeds``, every DeepStack level) and `imagePositions[slot]`. `keys[p]` holds the feature's hash, and the
/// row's logits peak at a hash of `keys[0...p]`. So wrong image rows, slots, positions or stale state change the output.
public enum FakeVLM {
    public static let vocab = 272
    public static let open: Int32 = 262, pad: Int32 = 263, close: Int32 = 264
    /// Patch 2, temporal 1, merge 2: rows of 12 values, the factor 4.
    public static let patch = 2, merge = 2, rowLength = 12
    public static let hidden = 4, levels = 3

    /// `preprocess.image` for the fake: smart resize (factor 4, 16…`patchRows`·4 pixels), block-major patches.
    public static func preprocess(patchRows: Int) -> Manifest.ImagePreprocess {
        var resize = Manifest.ImagePreprocess.Resize(mode: .smartResize, filter: .torchUint8Bicubic)
        resize.factor = patch * merge
        resize.minPixels = 16
        resize.maxPixels = patchRows * patch * patch
        return Manifest.ImagePreprocess(
            resize: resize, scale: 1.0 / 255, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], layout: .patches,
            patchify: .init(patch: patch, temporal: 1, merge: merge, rowOrder: .blockMajor, withinPatch: .channelFramePixel))
    }

    public static func imageBlock(slots: Int, images: Int) throws -> Manifest.ImageBlock {
        let json = """
        {"feed": [{"from": "vision.imageEmbeds", "to": "decoder.imageEmbeds"}],
         "tokens": {"pad": "<|image_pad|>", "padId": \(pad), "open": "<|vision_start|>", "openId": \(open),
                    "close": "<|vision_end|>", "closeId": \(close), "count": "gridH*gridW", "idBase": \(vocab)},
         "rope": {"rule": "qwen3vl-mrope"}, "maxImages": \(images), "slots": \(slots)}
        """
        return try JSONDecoder().decode(Manifest.ImageBlock.self, from: Data(json.utf8))
    }

    public static func visionGraph() -> Manifest.Graph {
        var graph = Manifest.Graph(
            file: .single(AssetPath(asset: "main", path: "tower.aimodel")), function: "main",
            inputs: ["patches": "patches", "grid": "grid"],
            outputs: ["imageEmbeds": "image_embeds", "deepstackEmbeds": "deepstack_embeds"], compute: .gpu,
            expectFrequentReshapes: false)
        graph.kind = .visionTower
        return graph
    }

    public static func decoderGraph(slots: Int, maxContext: Int = 4096, prefillChunk: Int = 512) -> Manifest.Graph {
        func s(_ name: String) -> Manifest.Binding {
            var b = Manifest.Binding(stringLiteral: name)
            b.binding = .static
            return b
        }
        let encodings = try! JSONDecoder().decode(
            Manifest.InputEncoding.self, from: Data(#"{"extensionIds": {"base": \#(vocab), "count": \#(slots)}}"#.utf8))
        return Manifest.Graph(
            file: .single(AssetPath(asset: "main", path: "decoder.aimodel")), function: "main",
            inputs: ["tokens": "input_ids", "positions": "position_ids", "imageEmbeds": s("image_embeds"),
                     "deepstackEmbeds": s("deepstack_embeds"), "imagePositions": s("image_positions"),
                     "ropeShiftStart": s("rope_shift_start"), "ropeShiftAmount": s("rope_shift_amount")],
            inputEncodings: ["tokens": encodings],
            outputs: ["logits": "logits"],
            states: Manifest.States(sequence: [Manifest.SequenceState(tensors: ["keys", "values"], seqAxis: 1)]),
            limits: Manifest.Limits(maxContext: maxContext, prefillChunk: prefillChunk), compute: .gpu,
            expectFrequentReshapes: true)
    }

    /// A VLM engine over the fakes: `patchRows` tower rows (`patchRows / 4` tokens an image), `slots` and `images`.
    public static func engine(
        patchRows: Int = 16, slots: Int = 24, images: Int = 4, prefillChunk: Int = 512, capacity: Int = 1024,
        towerLog: CallLog? = nil, rope: RopeLog? = nil, towerGate: TowerGate? = nil, options: DecoderEngine.Options? = nil,
        probe: GPUMemoryProbe? = nil, modelBytes: Int64 = 0
    ) async throws -> DecoderEngine {
        let decoder = FakeVLMDecoder(slots: slots, images: images, rope: rope)
        let tower = FakeVisionTower(rows: patchRows, log: towerLog, gate: towerGate)
        let device = Device(platform: .macOS, architecture: "fake9", physicalMemory: 64 << 30)
        let options = options ?? DecoderEngine.Options(capacity: capacity, memoryBudget: .bytes(1 << 30))
        guard let probe else {
            return try await DecoderEngine(
                step: decoder, graph: decoderGraph(slots: slots, prefillChunk: prefillChunk), vision: tower,
                visionGraph: visionGraph(), image: try imageBlock(slots: slots, images: images),
                preprocess: preprocess(patchRows: patchRows), device: device, options: options)
        }
        return try await DecoderEngine(
            step: decoder, graph: decoderGraph(slots: slots, prefillChunk: prefillChunk), vision: tower,
            visionGraph: visionGraph(), image: try imageBlock(slots: slots, images: images),
            preprocess: preprocess(patchRows: patchRows), device: device, options: options, probe: probe,
            modelBytes: modelBytes)
    }

    /// A deterministic `width` × `height` RGB image, different for each `seed`.
    public static func picture(width: Int, height: Int, seed: Int) -> CGImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                rgba[i] = UInt8((x * 37 + y * 11 + seed * 53) % 256)
                rgba[i + 1] = UInt8((x * 7 + y * 29 + seed * 101) % 256)
                rgba[i + 2] = UInt8((x * y + seed * 17) % 256)
            }
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }
}

/// The fake tower: `patches` f32 [rows, 12], `grid` i32 [2] → `image_embeds` f16 [rows/4, 4], `deepstack_embeds` f16
/// [3·rows/4, 4] (level-major). Token i's values mix its four patch rows; rows past the image's tokens are zero.
public final class FakeVisionTower: GraphBackend {
    public let signature: GraphSignature
    let rows: Int
    let log: CallLog?
    let gate: TowerGate?

    public init(rows: Int, log: CallLog? = nil, gate: TowerGate? = nil) {
        self.rows = rows
        self.log = log
        self.gate = gate
        let t = rows / (FakeVLM.merge * FakeVLM.merge)
        signature = GraphSignature(
            function: "main",
            inputs: [TensorSpec(name: "patches", scalarType: .float32, shape: [rows, FakeVLM.rowLength]),
                     TensorSpec(name: "grid", scalarType: .int32, shape: [2])],
            states: [],
            outputs: [TensorSpec(name: "image_embeds", scalarType: .float16, shape: [t, FakeVLM.hidden]),
                      TensorSpec(name: "deepstack_embeds", scalarType: .float16, shape: [FakeVLM.levels * t, FakeVLM.hidden])])
    }

    public func run(_ bindings: inout Bindings) async throws(AetherError) {
        try bindings.validate(against: signature)
        guard let patches = bindings.input("patches"), let grid = bindings.input("grid"),
              let embeds = bindings.output("image_embeds"), let deep = bindings.output("deepstack_embeds") else {
            throw .tensor(name: "patches", reason: .unbound)
        }
        try await gate?.pass()
        log?.append(1)
        let g = try grid.read(Int32.self)
        let tokens = Int(g[0]) * Int(g[1]) / 4, t = rows / 4, h = FakeVLM.hidden
        let p = try patches.read(Float.self)
        var e = [Float16](repeating: 0, count: t * h), d = [Float16](repeating: 0, count: FakeVLM.levels * t * h)
        for i in 0..<tokens {
            for j in 0..<h {
                var v: Float = 0
                for r in 0..<4 { for c in 0..<FakeVLM.rowLength { v += p[(4 * i + r) * FakeVLM.rowLength + c] * Float((c + j * 3 + r) % 5 + 1) } }
                e[i * h + j] = Float16((v * 8).rounded() / 8)
                for level in 0..<FakeVLM.levels { d[(level * t + i) * h + j] = Float16(((v + Float(level + 1)) * 4).rounded() / 4) }
            }
        }
        try embeds.write(e)
        try deep.write(d)
    }
}

/// The fake decoder (see ``FakeVLM``): `input_ids` [1, S], `position_ids` [1, T], the five static inputs, KV-like
/// states `keys`/`values` [1, C], logits [1, S, vocab].
public final class FakeVLMDecoder: GraphBackend {
    public let signature: GraphSignature
    let slots: Int
    let rope: RopeLog?

    public init(slots: Int, images: Int, rope: RopeLog? = nil) {
        self.slots = slots
        self.rope = rope
        signature = GraphSignature(
            function: "main",
            inputs: [
                TensorSpec(name: "input_ids", scalarType: .int32, shape: [1, -1]),
                TensorSpec(name: "position_ids", scalarType: .int32, shape: [1, -1]),
                TensorSpec(name: "image_embeds", scalarType: .float16, shape: [slots, FakeVLM.hidden]),
                TensorSpec(name: "deepstack_embeds", scalarType: .float16, shape: [FakeVLM.levels * slots, FakeVLM.hidden]),
                TensorSpec(name: "image_positions", scalarType: .int32, shape: [slots, 3]),
                TensorSpec(name: "rope_shift_start", scalarType: .int32, shape: [images]),
                TensorSpec(name: "rope_shift_amount", scalarType: .int32, shape: [images]),
            ],
            states: [TensorSpec(name: "keys", scalarType: .float32, shape: [1, -1]),
                     TensorSpec(name: "values", scalarType: .float32, shape: [1, -1])],
            outputs: [TensorSpec(name: "logits", scalarType: .float32, shape: [1, -1, FakeVLM.vocab])])
    }

    static func mix(_ values: some Sequence<Int>) -> Int {
        var h: UInt64 = 1469598103934665603
        for v in values { h = (h ^ UInt64(bitPattern: Int64(v))) &* 1099511628211 }
        return Int(h % 65521)
    }

    public func run(_ bindings: inout Bindings) async throws(AetherError) {
        try bindings.validate(against: signature)
        guard let tokens = bindings.input("input_ids"), let positions = bindings.input("position_ids"),
              let keys = bindings.state("keys"), let values = bindings.state("values"),
              let embeds = bindings.input("image_embeds"), let deep = bindings.input("deepstack_embeds"),
              let imagePositions = bindings.input("image_positions"), let start = bindings.input("rope_shift_start"),
              let amount = bindings.input("rope_shift_amount") else { throw .tensor(name: "input_ids", reason: .unbound) }
        let s = bindings.boundShape(of: "input_ids")![1], p = bindings.boundShape(of: "position_ids")![1]
        let origin = bindings.inputOrigin("input_ids")![1]
        let ids = try tokens.read(Int32.self, origin..<(origin + s))
        let pos = try positions.read(Int32.self, 0..<p)
        let e = try embeds.read(Float16.self), d = try deep.read(Float16.self)
        let ip = try imagePositions.read(Int32.self), st = try start.read(Int32.self), am = try amount.read(Int32.self)
        var cache = try keys.read(Float.self)
        let h = FakeVLM.hidden, v = FakeVLM.vocab
        for i in 0..<s {
            let position = Int(pos[p - s + i])
            guard position < cache.count else { throw .capacity(what: "fake kv", needed: Int64(position + 1), available: Int64(cache.count)) }
            let id = Int(ids[i])
            let feature: Int
            if id >= v {
                let g = id - v
                let rows = (0..<h).map { Int(Float(e[g * h + $0]) * 8) }
                    + (0..<FakeVLM.levels).flatMap { level in (0..<h).map { Int(Float(d[(level * slots + g) * h + $0]) * 4) } }
                let thw = Array(ip[(g * 3)..<(g * 3 + 3)])
                rope?.record(position, thw)
                feature = Self.mix([1] + rows + thw.map(Int.init))
            } else {
                var r = position
                for j in 0..<st.count where position >= Int(st[j]) { r -= Int(am[j]) }
                rope?.record(position, [Int32(r), Int32(r), Int32(r)])
                feature = Self.mix([2, id, r])
            }
            cache[position] = Float(feature)
            try values.write([Float(position)], startingAt: position)
            // Printable bytes only, so the byte-level test tokenizer decodes every reply.
            let target = 32 + Self.mix(cache[0...position].map { Int($0) }) % 95
            try bindings.output("logits")?.write((0..<v).map { $0 == target ? Float(10) : -Float(abs($0 - target)) * 0.01 },
                                                 startingAt: i * v)
        }
        try keys.write(cache)
    }
}

/// ``ByteRenderer`` with images: a user turn's images render as `[open, pad, close]` each, before its text.
public struct VLMByteRenderer: PromptRenderer {
    public init() {}
    public func render(_ messages: [ChatMessage], options: RenderOptions) throws -> RenderedPrompt {
        var out: [Int32] = []
        for message in messages {
            let images = message.images.flatMap { _ in [FakeVLM.open, FakeVLM.pad, FakeVLM.close] }
            out += [ByteVocab.imStart] + ByteVocab.tokens("\(message.role.rawValue)\n") + images + ByteVocab.tokens(message.content)
                + [ByteVocab.imEnd] + ByteVocab.tokens("\n")
        }
        if options.addGenerationPrompt { out += [ByteVocab.imStart] + ByteVocab.tokens("assistant\n") }
        return RenderedPrompt(tokens: out)
    }
}

extension FakeVLM {
    /// A Generator over a fake VLM engine with the byte-level test tokenizer.
    public static func generator(
        towerLog: CallLog? = nil, slots: Int = 64, images: Int = 4, towerGate: TowerGate? = nil
    ) async throws -> Generator {
        let engine = try await engine(slots: slots, images: images, towerLog: towerLog, towerGate: towerGate)
        return Generator(engine: engine, decoder: ByteDecoding(), renderer: VLMByteRenderer(), stopTokens: [ByteVocab.imEnd],
                         profile: ByteVocab.profile, gpu: .unadmitted)
    }
}
