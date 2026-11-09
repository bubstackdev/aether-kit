public import AetherCore
import Foundation

/// The image half of a `vlm-static-inject@1` ``DecoderEngine`` (docs/VLM.md §3.3): the vision tower, its buffers, and
/// what the decoder's static inputs hold. Non-`Sendable`: the engine actor owns it, as it owns ``DecoderCore``.
///
/// One image per tower call: its patch rows are written into the tower's `patches` buffer (zero-padded to the graph's
/// rows), `grid` gets its patch grid, and the tower's outputs (`imageEmbeds` `[T, H]`, `deepstackEmbeds` `[L·T, H]`,
/// level-major) are copied into the decoder's statics at the image's slots (`[N, H]`, `[L·N, H]`). Then the rotary
/// inputs are rewritten for the whole prompt.
final class ImageBinder {
    struct Roles {
        var patches: String
        var grid: String?
        var embeds: String
        var deepstack: String?
    }

    let spec: ImageInputSpec
    let tower: any GraphBackend
    let towerRoles: Roles
    let arena: TensorArena
    let patches: TensorBuffer
    let grid: TensorBuffer?
    let embeds: TensorBuffer
    let deepstack: TensorBuffer?
    /// The tower's token rows T and the decoder's slots N, hidden width H and DeepStack levels L.
    let towerTokens: Int
    let hidden: Int
    let levels: Int
    /// The bytes the tower's buffers hold (the engine's fixed memory).
    let bufferBytes: Int64
    private(set) var bound: [ImagePromptPlan.Image] = []
    /// Tower calls since load (diagnostics and tests).
    private(set) var encodes = 0

    /// Checks the tower and the decoder's statics against each other and the manifest. Errors name the field.
    init(
        tower: any GraphBackend, vision: Manifest.Graph, decoder: Manifest.Graph, decoderSignature: GraphSignature,
        image: Manifest.ImageBlock, preprocess: Manifest.ImagePreprocess
    ) throws(AetherError) {
        guard vision.kind == .visionTower else {
            throw .manifestMismatch(field: "graphs.vision.kind", expected: DriverID.visionTower.description,
                                    found: vision.kind?.description ?? "none")
        }
        guard let patchesName = vision.inputs["patches"]?.name, let embedsName = vision.outputs["imageEmbeds"]?.name else {
            throw .manifestInvalid(field: "graphs.vision", reason: "vision-tower@1 needs patches in and imageEmbeds out")
        }
        let roles = Roles(patches: patchesName, grid: vision.inputs["grid"]?.name, embeds: embedsName,
                          deepstack: vision.outputs["deepstackEmbeds"]?.name)
        let sig = tower.signature
        guard let p = sig.input(patchesName), p.rank == 2, !p.isDynamic, let e = sig.output(embedsName), e.rank == 2,
              !e.isDynamic else {
            throw .manifestMismatch(field: "graphs.vision", expected: "static patches [rows, dim] and imageEmbeds [tokens, H]",
                                    found: "\(sig.inputs.map(\.summary)) → \(sig.outputs.map(\.summary))")
        }
        let tokens = try image.typedTokens()
        guard let patchify = preprocess.patchify, preprocess.layout == .patches else {
            throw .manifestInvalid(field: "preprocess.image.layout", reason: "vlm-static-inject@1 takes patch rows")
        }
        let merge = patchify.merge ?? 1
        let rows = p.shape[0], towerTokens = e.shape[0], hidden = e.shape[1]
        guard rows == towerTokens * merge * merge else {
            throw .manifestMismatch(field: "graphs.vision.outputs.imageEmbeds", expected: "\(rows / (merge * merge)) token rows",
                                    found: e.summary)
        }
        let rowLength = 3 * (patchify.temporal ?? 1) * patchify.patch * patchify.patch
        guard p.shape[1] == rowLength else {
            throw .manifestMismatch(field: "graphs.vision.inputs.patches", expected: "rows of \(rowLength)", found: p.summary)
        }
        if let resize = preprocess.resize, resize.mode == .smartResize {
            guard resize.maxPixels == rows * patchify.patch * patchify.patch else {
                throw .manifestMismatch(field: "preprocess.image.resize.maxPixels", expected: "\(rows * patchify.patch * patchify.patch) (the tower's rows · patch²)",
                                        found: "\(resize.maxPixels ?? 0)")
            }
        }
        // The decoder's statics.
        let inputs = decoder.inputs(forPhase: "step")
        func spec(_ role: String) -> TensorSpec? { inputs[role].flatMap { decoderSignature.input($0.name) } }
        guard let slotsSpec = spec("imageEmbeds"), slotsSpec.rank == 2, slotsSpec.shape[1] == hidden else {
            throw .manifestMismatch(field: "graphs.decoder.inputs.imageEmbeds", expected: "[slots, \(hidden)]",
                                    found: spec("imageEmbeds")?.summary ?? "none")
        }
        let slots = slotsSpec.shape[0]
        var levels = 0
        if let deep = spec("deepstackEmbeds") {
            guard let towerDeep = roles.deepstack.flatMap(sig.output), towerDeep.rank == 2, towerDeep.shape[1] == hidden,
                  towerDeep.shape[0] % towerTokens == 0, deep.rank == 2, deep.shape[1] == hidden,
                  deep.shape[0] == slots * (towerDeep.shape[0] / towerTokens) else {
                throw .manifestMismatch(field: "graphs.decoder.inputs.deepstackEmbeds", expected: "[levels · \(slots), \(hidden)] fed by the tower's deepstackEmbeds",
                                        found: deep.summary)
            }
            levels = deep.shape[0] / slots
        }
        guard let positions = spec("imagePositions"), positions.shape == [slots, 3], positions.scalarType == .int32,
              let start = spec("ropeShiftStart"), let amount = spec("ropeShiftAmount"), start.rank == 1,
              start.shape == amount.shape, start.scalarType == .int32, amount.scalarType == .int32 else {
            throw .manifestMismatch(field: "graphs.decoder.inputs.imagePositions", expected: "imagePositions int32 [\(slots), 3], ropeShiftStart and ropeShiftAmount int32 [images] (the qwen3vl-mrope rule)",
                                    found: "\(spec("imagePositions")?.summary ?? "none"), \(spec("ropeShiftStart")?.summary ?? "none")")
        }
        let maxImages = start.shape[0]
        if let declared = image.slots, declared != slots {
            throw .manifestMismatch(field: "image.slots", expected: "\(slots) (the decoder's imageEmbeds rows)", found: "\(declared)")
        }
        if let declared = image.maxImages, declared != maxImages {
            throw .manifestMismatch(field: "image.maxImages", expected: "\(maxImages) (the decoder's rope-shift entries)", found: "\(declared)")
        }
        guard let ext = decoder.inputEncodings?["tokens"]?.extensionIds, ext.base == tokens.idBase, ext.count >= slots else {
            throw .manifestMismatch(field: "graphs.decoder.inputEncodings.tokens.extensionIds",
                                    expected: "base \(tokens.idBase) (image.tokens.idBase), count ≥ \(slots)",
                                    found: "\(decoder.inputEncodings?["tokens"]?.extensionIds.map { "base \($0.base), count \($0.count)" } ?? "none")")
        }
        guard let idBase = Int32(exactly: tokens.idBase), let padId = Int32(exactly: tokens.padId) else {
            throw .manifestInvalid(field: "image.tokens", reason: "ids must fit Int32")
        }
        self.spec = ImageInputSpec(
            padId: padId, openId: tokens.openId.flatMap { Int32(exactly: $0) }, closeId: tokens.closeId.flatMap { Int32(exactly: $0) },
            idBase: idBase, slots: slots, maxImages: maxImages, patchRows: rows, merge: merge,
            preprocessor: try ImagePreprocessor(preprocess))
        self.tower = tower
        towerRoles = roles
        // The tower runs on the host path (`run`), one image at a time, on its own arena.
        let arena = TensorArena()
        self.arena = arena
        patches = try tower.makeBuffer(for: patchesName, role: .input, dimensions: [], in: arena)
        if let name = roles.grid {
            grid = try tower.makeBuffer(for: name, role: .input, dimensions: [], in: arena)
        } else {
            grid = nil
        }
        embeds = try tower.makeBuffer(for: embedsName, role: .output, dimensions: [], in: arena)
        deepstack = levels > 0 ? try tower.makeBuffer(for: roles.deepstack!, role: .output, dimensions: [], in: arena) : nil
        self.towerTokens = towerTokens
        self.hidden = hidden
        self.levels = levels
        bufferBytes = [patches, embeds] .map { Int64($0.storageByteCount) }.reduce(0, +)
            + Int64(deepstack?.storageByteCount ?? 0) + Int64(grid?.storageByteCount ?? 0)
    }

    /// The text-only rotary inputs (positions zero, every shift entry unused); the image rows stay as they are (no
    /// text token reads them). Nothing is bound afterwards.
    func clear(_ statics: [String: TensorBuffer]) throws(AetherError) {
        bound = []
        try writeRotary(ImagePromptPlan.empty(spec), statics)
    }

    /// Keeps only the images that end at or before `position` (a rewind): a later or cut image's rows and shift no
    /// longer belong to the state, and text fed after the rewind must not take its shift.
    func truncate(to position: Int, _ statics: [String: TensorBuffer]) throws(AetherError) {
        let kept = bound.prefix { $0.end <= position }
        guard kept.count < bound.count else { return }
        let plan = ImagePromptPlan.retaining(Array(kept), spec: spec)
        bound = []
        try writeRotary(plan, statics)
        bound = Array(kept)
    }

    /// Encodes every image of `plan` the statics don't hold at its slots, copies its rows in, and writes the rotary
    /// inputs. On a failure nothing is bound.
    func bind(_ plan: ImagePromptPlan, prepared: [PreparedPromptImage], _ statics: [String: TensorBuffer]) async throws(AetherError) {
        guard prepared.count == plan.images.count, zip(prepared, plan.images).allSatisfy({ $0.digest == $1.digest }) else {
            throw .generation(.invalidParameters("the prepared images don't match the plan's"))
        }
        let previous = bound
        bound = []
        for (k, image) in plan.images.enumerated() {
            let held = k < previous.count && previous[k].digest == image.digest && previous[k].base == image.base
                && previous[k].count == image.count
            if held { continue }
            try await encode(prepared[k], into: image.base, statics)
        }
        try writeRotary(plan, statics)
        bound = plan.images
    }

    private func encode(_ image: PreparedPromptImage, into base: Int, _ statics: [String: TensorBuffer]) async throws(AetherError) {
        let rows = image.gridH * image.gridW
        guard image.rowLength == patches.shape[1], rows <= patches.shape[0], image.values.count == rows * image.rowLength else {
            throw .invalidInput("a prepared image of \(rows)x\(image.rowLength) for patches \(patches.shape)")
        }
        try patches.zero()
        try patches.write(image.values)
        try grid?.write([Int32(image.gridH), Int32(image.gridW)])
        var b = Bindings(arena: arena)
        try b.bindInput(towerRoles.patches, to: patches)
        if let name = towerRoles.grid, let grid { try b.bindInput(name, to: grid) }
        try b.bindOutput(towerRoles.embeds, to: embeds)
        if let name = towerRoles.deepstack, let deepstack { try b.bindOutput(name, to: deepstack) }
        try await tower.run(&b)
        encodes += 1
        let count = image.tokens
        guard let target = statics["imageEmbeds"] else { throw .tensor(name: "imageEmbeds", reason: .unbound) }
        try Self.copyRows(from: embeds, row: 0, to: target, row: base, rows: count, width: hidden)
        if let deepstack, let deep = statics["deepstackEmbeds"] {
            let slots = target.shape[0]
            for level in 0..<levels {
                try Self.copyRows(from: deepstack, row: level * towerTokens, to: deep, row: level * slots + base, rows: count, width: hidden)
            }
        }
    }

    private func writeRotary(_ plan: ImagePromptPlan, _ statics: [String: TensorBuffer]) throws(AetherError) {
        guard let positions = statics["imagePositions"], let start = statics["ropeShiftStart"],
              let amount = statics["ropeShiftAmount"] else { throw .tensor(name: "imagePositions", reason: .unbound) }
        try positions.write(plan.imagePositions(slots: spec.slots))
        let shifts = plan.ropeShifts(entries: spec.maxImages)
        try start.write(shifts.start)
        try amount.write(shifts.amount)
    }

    /// Copies `rows` rows of `width` elements between two contiguous buffers (any arenas), converting between float
    /// widths when the types differ.
    static func copyRows(
        from source: TensorBuffer, row sourceRow: Int, to target: TensorBuffer, row targetRow: Int, rows: Int, width: Int
    ) throws(AetherError) {
        guard rows > 0 else { return }
        let n = rows * width, from = sourceRow * width, to = targetRow * width
        guard from + n <= source.count, to + n <= target.count, source.isContiguous, target.isContiguous else {
            throw .tensor(name: target.name, reason: .outOfBounds(index: to + n, count: target.count))
        }
        if source.scalarType == target.scalarType, let size = source.scalarType.byteWidth, source.arena !== target.arena {
            try source.arena.withMutableBytes(source) { src throws(AetherError) in
                try target.arena.withMutableBytes(target) { dst throws(AetherError) in
                    (dst.baseAddress! + to * size).copyMemory(from: src.baseAddress! + from * size, byteCount: n * size)
                }
            }
            return
        }
        switch target.scalarType {
        case .float16:
            let values = try source.readFloats(from..<(from + n))
            try target.write(Float16.self, startingAt: to, count: n) { Float16(values[$0]) }
        default:
            try target.write(try source.readFloats(from..<(from + n)), startingAt: to)
        }
    }
}

extension ImagePromptPlan {
    /// A plan with no image: the text-only rotary inputs.
    static func empty(_ spec: ImageInputSpec) -> ImagePromptPlan {
        ImagePromptPlan(tokens: [], images: [], idBase: spec.idBase, padId: spec.padId)
    }

    /// A plan holding `images` as already placed (the images kept by a rewind).
    static func retaining(_ images: [Image], spec: ImageInputSpec) -> ImagePromptPlan {
        ImagePromptPlan(tokens: [], images: images, idBase: spec.idBase, padId: spec.padId)
    }
}

extension DecoderEngine: ImageConditionedEngine {
    /// The images the static inputs hold (empty for a text model).
    public var boundImages: [ImagePromptPlan.Image] { imageBinder?.bound ?? [] }

    /// Tower calls since load (diagnostics: a reused image isn't encoded again).
    public var imageEncodes: Int { imageBinder?.encodes ?? 0 }

    public func bindImages(_ plan: ImagePromptPlan, prepared: [PreparedPromptImage]) async throws(AetherError) {
        guard let binder = imageBinder else {
            guard plan.images.isEmpty else { throw .unsupportedOperation("this model takes no images") }
            return
        }
        // Held as the engine's operation across the tower calls: no prefill, decode, rewind or second bind runs while
        // a bind is suspended.
        let core = core
        try await core.withStatics(operation: "an image bind") { statics throws(AetherError) in
            let previous = binder.bound
            do throws(AetherError) {
                try await binder.bind(plan, prepared: prepared, statics)
            } catch {
                try? binder.clear(statics)
                // What the state fed after the images it held took their rotary shifts, which the statics no longer
                // hold: refuse to feed on it until a rewind before them or a reset.
                if let first = previous.first { core.markImageStateStale(from: first.start) }
                throw error
            }
        }
    }
}

/// The `vlm-static-inject@1` driver (MANIFEST.md §1.2): loads an installed bundle's decoder and vision tower into one
/// ``DecoderEngine``, an ``ImageConditionedEngine``.
public enum VLMStaticInjectDriver: ModelDriver {
    public static let id: DriverID = .vlmStaticInject

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> DecoderEngine {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        return try await DecoderEngine(bundle: bundle, manifest: manifest, device: options.device, options: options.decoder)
    }
}
