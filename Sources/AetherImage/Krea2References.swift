public import AetherCore
public import AetherHost
import AetherRuntime
import CryptoKit
public import Foundation
import ImageIO
import CoreGraphics

// Reference images for Krea 2 (MANIFEST.md §5.6 `krea2.references`): each reference becomes extra text-encoder rows.
// The prompt renders one labelled image part per reference before the text, each image pad expanded to the image's
// merged token count; the `vision` graph embeds each image; the text encoder's `references` function takes the token
// ids (image pads as extension ids), the three-axis positions and the embeddings, and returns the conditioning rows
// the unchanged denoisers take. The rules are the 0.0.x `visionTokens` contract's, so a reference prompt conditions
// exactly as it did there.

/// One reference image: 8-bit RGB, rows top to bottom. Built from memory only (no file path is read).
public struct Krea2ReferenceImage: Sendable, Hashable {
    /// Encoded bytes accepted for one reference, and the decoded sides accepted before fitting (bounds decode memory).
    public static let maximumEncodedBytes = 32 * 1024 * 1024
    public static let maximumSourceSide = 8192

    public let width: Int
    public let height: Int
    public let rgb: [UInt8]
    /// SHA-256 (lowercase hex) of the width and height (little-endian Int64) then the RGB bytes: the image's identity
    /// in a reference conditioning (``Krea2Conditioning/referenceDigests``), as 0.0.x's.
    public let digest: String

    /// Fails unless `rgb` holds `width · height · 3` bytes and each side is in 1…``maximumSourceSide``.
    public init(rgb: [UInt8], width: Int, height: Int) throws(AetherError) {
        guard width > 0, height > 0, width <= Self.maximumSourceSide, height <= Self.maximumSourceSide,
              rgb.count == width * height * 3 else {
            throw .generation(.invalidParameters("a reference's RGB bytes don't match \(width)x\(height)"))
        }
        self.width = width
        self.height = height
        self.rgb = rgb
        var hash = SHA256()
        withUnsafeBytes(of: Int64(width).littleEndian) { hash.update(bufferPointer: $0) }
        withUnsafeBytes(of: Int64(height).littleEndian) { hash.update(bufferPointer: $0) }
        rgb.withUnsafeBytes { hash.update(bufferPointer: $0) }
        digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Decodes PNG, JPEG or HEIC bytes with the EXIF orientation applied, drawn into 8-bit RGB in the image's own RGB
    /// colour space when it has one (an untagged or sRGB PNG keeps its exact bytes), else sRGB; alpha onto black.
    public init(encoded data: Data) throws(AetherError) {
        guard !data.isEmpty, data.count <= Self.maximumEncodedBytes,
              let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetCount(source) > 0 else {
            throw .generation(.invalidParameters("a reference is not a supported image"))
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let declaredWidth = (properties?[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
        let declaredHeight = (properties?[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
        guard (1...Self.maximumSourceSide).contains(declaredWidth), (1...Self.maximumSourceSide).contains(declaredHeight) else {
            throw .generation(.invalidParameters("a reference's sides must be 1...\(Self.maximumSourceSide)"))
        }
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        let image: CGImage?
        if orientation == 1 {
            image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        } else {
            // The thumbnail path applies the EXIF transform; the longest side keeps full resolution.
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: max(declaredWidth, declaredHeight)]
            image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        }
        guard let image else { throw .generation(.invalidParameters("a reference could not be decoded")) }
        let (w, h) = (image.width, image.height)
        guard w > 0, h > 0, w <= Self.maximumSourceSide, h <= Self.maximumSourceSide else {
            throw .generation(.invalidParameters("a reference's sides must be 1...\(Self.maximumSourceSide)"))
        }
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
            ?? CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        var rgbx = [UInt8](repeating: 0, count: w * h * 4)
        let drawn: Bool = rgbx.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                          bytesPerRow: w * 4, space: space,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            context.interpolationQuality = .none
            context.setBlendMode(.copy)
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw .generation(.invalidParameters("a reference could not be drawn as 8-bit RGB")) }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[3 * i] = rgbx[4 * i]
            rgb[3 * i + 1] = rgbx[4 * i + 1]
            rgb[3 * i + 2] = rgbx[4 * i + 2]
        }
        try self.init(rgb: rgb, width: w, height: h)
    }
}

/// A reference image as patch rows for the vision graph: `[patches, rowLength]` in merge-window order.
struct Krea2ReferencePatches: Sendable, Hashable {
    /// The patch grid (rows, columns); the image's tokens are `(rows / merge) · (columns / merge)`.
    var rows: Int
    var columns: Int
    var merge: Int
    var values: [Float]
    var digest: String

    var patches: Int { rows * columns }
    var tokenRows: Int { rows / merge }
    var tokenColumns: Int { columns / merge }
    var tokens: Int { tokenRows * tokenColumns }
}

/// The 0.0.x reference preprocessing, exactly: fit the longer side to `maxSide` (the other side rounded, never up),
/// the Qwen-VL smart resize to multiples of `factor`, one antialiased Keys cubic resample (a = −0.5) from the source,
/// then `(v − mean) / std` and patch rows over merge windows, each `[C][T][py][px]` with the still image repeated
/// over the temporal patch.
enum Krea2ReferencePreprocessing {
    typealias Rule = Manifest.Krea2.References.Preprocess

    static func fitted(width: Int, height: Int, maxSide: Int) -> (width: Int, height: Int) {
        let scale = min(1, Double(maxSide) / Double(max(width, height)))
        guard scale < 1 else { return (width, height) }
        return (max(1, Int((Double(width) * scale).rounded())), max(1, Int((Double(height) * scale).rounded())))
    }

    /// Sides round to multiples of `factor` (ties to even), then scale into `minPixels…maxPixels`.
    static func targetSize(width: Int, height: Int, rule: Rule) throws(AetherError) -> (width: Int, height: Int) {
        guard width > 0, height > 0 else { throw .generation(.invalidParameters("an empty reference image")) }
        guard Double(max(width, height)) / Double(min(width, height)) <= rule.maxAspect else {
            throw .generation(.invalidParameters("a reference's aspect ratio exceeds \(rule.maxAspect)"))
        }
        let f = Double(rule.factor)
        let maxPixels = Double(rule.maxPixels)
        let minPixels = min(Double(rule.minPixels), maxPixels)
        let h = Double(height), w = Double(width)
        var hb = max(f, (h / f).rounded(.toNearestOrEven) * f)
        var wb = max(f, (w / f).rounded(.toNearestOrEven) * f)
        if hb * wb > maxPixels {
            let beta = (h * w / maxPixels).squareRoot()
            hb = max(f, (h / beta / f).rounded(.down) * f)
            wb = max(f, (w / beta / f).rounded(.down) * f)
        } else if hb * wb < minPixels {
            let beta = (minPixels / (h * w)).squareRoot()
            hb = (h * beta / f).rounded(.up) * f
            wb = (w * beta / f).rounded(.up) * f
        }
        return (Int(wb), Int(hb))
    }

    static func prepare(_ image: Krea2ReferenceImage, rule: Rule) throws(AetherError) -> Krea2ReferencePatches {
        let fit = fitted(width: image.width, height: image.height, maxSide: rule.maxSide)
        let size = try targetSize(width: fit.width, height: fit.height, rule: rule)
        let rgb = (size.width, size.height) == (image.width, image.height)
            ? image.rgb : resizeBicubic(image.rgb, width: image.width, height: image.height, to: size.width, size.height)
        return patchify(rgb, width: size.width, height: size.height, rule: rule, digest: image.digest)
    }

    static func patchify(_ rgb: [UInt8], width: Int, height: Int, rule: Rule, digest: String) -> Krea2ReferencePatches {
        let p = rule.patch, m = rule.merge, tp = rule.temporalPatch
        let gh = height / p, gw = width / p, rowLength = rule.rowLength
        var values = [Float](repeating: 0, count: gh * gw * rowLength)
        // (v − mean) / std rounded once: v − 127.5 is exact in Float, so one division matches the reference processor's
        // float32 tensor bit for bit (an affine v · a + b does not).
        let mean = Float(rule.mean), std = Float(rule.std)
        let table = (0..<256).map { (Float($0) - mean) / std }
        values.withUnsafeMutableBufferPointer { out in
            rgb.withUnsafeBufferPointer { src in
                var row = 0
                for by in 0..<(gh / m) {
                    for bx in 0..<(gw / m) {
                        for dy in 0..<m {
                            for dx in 0..<m {
                                let py = by * m + dy, px = bx * m + dx
                                let base = row * rowLength
                                for c in 0..<3 {
                                    for y in 0..<p {
                                        let srcRow = ((py * p + y) * width + px * p) * 3 + c
                                        for x in 0..<p {
                                            let v = table[Int(src[srcRow + x * 3])]
                                            for t in 0..<tp { out[base + ((c * tp + t) * p + y) * p + x] = v }
                                        }
                                    }
                                }
                                row += 1
                            }
                        }
                    }
                }
            }
        }
        return Krea2ReferencePatches(rows: gh, columns: gw, merge: m, values: values, digest: digest)
    }

    struct Taps { var start: Int; var weights: [Float] }

    /// Separable antialiased Keys cubic (a = −0.5): kernels widen by the scale when downscaling, sample centres at pixel
    /// centres, each output rounded once after the vertical pass.
    static func resizeBicubic(_ rgb: [UInt8], width: Int, height: Int, to outW: Int, _ outH: Int) -> [UInt8] {
        let hTaps = taps(inSize: width, outSize: outW), vTaps = taps(inSize: height, outSize: outH)
        var mid = [Float](repeating: 0, count: outW * height * 3)
        for y in 0..<height {
            let row = y * width * 3
            for x in 0..<outW {
                let t = hTaps[x]
                var r: Float = 0, g: Float = 0, b: Float = 0
                for (k, w) in t.weights.enumerated() {
                    let i = row + (t.start + k) * 3
                    r += w * Float(rgb[i]); g += w * Float(rgb[i + 1]); b += w * Float(rgb[i + 2])
                }
                let o = (y * outW + x) * 3
                mid[o] = r; mid[o + 1] = g; mid[o + 2] = b
            }
        }
        var out = [UInt8](repeating: 0, count: outW * outH * 3)
        for y in 0..<outH {
            let t = vTaps[y]
            for x in 0..<outW {
                var r: Float = 0, g: Float = 0, b: Float = 0
                for (k, w) in t.weights.enumerated() {
                    let i = ((t.start + k) * outW + x) * 3
                    r += w * mid[i]; g += w * mid[i + 1]; b += w * mid[i + 2]
                }
                let o = (y * outW + x) * 3
                out[o] = clamp8(r); out[o + 1] = clamp8(g); out[o + 2] = clamp8(b)
            }
        }
        return out
    }

    static func taps(inSize: Int, outSize: Int) -> [Taps] {
        let scale = Double(inSize) / Double(outSize)
        let stretch = max(scale, 1)
        let radius = 2 * stretch
        return (0..<outSize).map { o in
            let centre = (Double(o) + 0.5) * scale
            let lo = max(Int((centre - radius + 0.5).rounded(.down)), 0)
            let hi = min(Int((centre + radius + 0.5).rounded(.down)), inSize)
            var w = (lo..<max(hi, lo + 1)).map { cubic((Double($0) + 0.5 - centre) / stretch) }
            let sum = w.reduce(0, +)
            if sum != 0 { w = w.map { $0 / sum } }
            return Taps(start: lo, weights: w.map(Float.init))
        }
    }

    static func cubic(_ x: Double) -> Double {
        let a = -0.5, t = abs(x)
        if t < 1 { return (a + 2) * t * t * t - (a + 3) * t * t + 1 }
        if t < 2 { return a * t * t * t - 5 * a * t * t + 8 * a * t - 4 * a }
        return 0
    }

    @inline(__always) static func clamp8(_ v: Float) -> UInt8 { UInt8(max(0, min(255, v.rounded()))) }
}

/// A rendered reference prompt (0.0.x's `KreaReferencePrompt`): the text encoder's ids with each image pad expanded
/// to its image's tokens, the M-RoPE positions, and where each image's rows are.
struct Krea2ReferencePrompt: Sendable, Hashable {
    struct Span: Sendable, Hashable {
        var range: Range<Int>
        var rows: Int
        var columns: Int
    }

    /// Image pads as the pad id (``prepared``'s view); ``encoderIDs`` has them as extension ids.
    var ids: [Int32]
    var encoderIDs: [Int32]
    /// `[3, L]` axis-major: time, height, width.
    var ropePositions: [Int32]
    var spans: [Span]
    var truncated: Bool
    var renderedTokenCount: Int
    var conditioningRows: Int

    /// - `rendered`: the ids of `systemPrefix + parts + text + suffix` (one pad per image).
    /// - `textOnlyCount`: the id count of the text-only rendering, which alone decides truncation (images and labels
    ///   are never cut; they count against `maxImageTokens`).
    init(
        rendered: [Int32], textOnlyCount: Int, images: [Krea2ReferencePatches], prompt: Manifest.Krea2.Prompt,
        references: Manifest.Krea2.References
    ) throws(AetherError) {
        guard (1...references.maxImages).contains(images.count) else {
            throw .generation(.invalidParameters("Krea 2 takes 1...\(references.maxImages) reference images"))
        }
        let total = images.reduce(0) { $0 + $1.tokens }
        guard images.allSatisfy({ $0.tokens > 0 }), total <= references.maxImageTokens else {
            throw .generation(.invalidParameters(
                "the reference images need \(total) tokens; the limit is \(references.maxImageTokens)"))
        }
        guard textOnlyCount > prompt.dropLeadingRows, rendered.allSatisfy({ $0 >= 0 }) else {
            throw .generation(.invalidParameters("the rendered Krea 2 reference prompt is invalid"))
        }
        var ids = rendered
        truncated = textOnlyCount > prompt.maxTokens
        if truncated {
            // Image parts precede the text, so the cut just before the kept suffix removes only text.
            let keep = prompt.keepTrailingTokens
            let cut = ids.count - keep - (textOnlyCount - prompt.maxTokens)
            guard cut > 0 else { throw .generation(.invalidParameters("the rendered Krea 2 reference prompt is invalid")) }
            ids = Array(ids.prefix(cut)) + ids.suffix(keep)
        }
        renderedTokenCount = rendered.count
        let pad = references.tokens.pad
        var out: [Int32] = [], encoder: [Int32] = []
        out.reserveCapacity(ids.count + total)
        encoder.reserveCapacity(ids.count + total)
        var spans: [Span] = []
        var slot = 0
        for id in ids {
            guard id == pad else {
                out.append(id)
                encoder.append(id)
                continue
            }
            guard spans.count < images.count else {
                throw .generation(.invalidParameters("the rendered prompt has more image pads than its \(images.count) image(s)"))
            }
            let image = images[spans.count]
            spans.append(Span(range: out.count..<(out.count + image.tokens), rows: image.tokenRows, columns: image.tokenColumns))
            for _ in 0..<image.tokens {
                out.append(pad)
                encoder.append(Int32(references.extensionBase + slot))
                slot += 1
            }
        }
        guard spans.count == images.count else {
            throw .generation(.invalidParameters("the rendered prompt has \(spans.count) image pad(s) for \(images.count) image(s)"))
        }
        guard let first = spans.first, first.range.lowerBound >= prompt.dropLeadingRows else {
            throw .generation(.invalidParameters("a reference image renders inside the dropped prefix rows"))
        }
        conditioningRows = out.count - prompt.dropLeadingRows
        guard let rows = references.rowRange, rows.contains(conditioningRows) else {
            throw .generation(.invalidParameters(
                "the reference prompt has \(conditioningRows) conditioning rows; they must be in \(references.rows)"))
        }
        self.ids = out
        encoderIDs = encoder
        self.spans = spans
        ropePositions = Self.positions(count: out.count, spans: spans)
    }

    /// Text tokens at the running position on all three axes. An image span starting at running position p with
    /// token grid (1, h, w) puts its tokens in raster order at (p, p + row, p + column); the running position then
    /// advances by max(1, h, w).
    static func positions(count: Int, spans: [Span]) -> [Int32] {
        var axes = [[Int32]](repeating: [], count: 3)
        for a in 0..<3 { axes[a].reserveCapacity(count) }
        var p: Int32 = 0, i = 0, k = 0
        while i < count {
            if k < spans.count, spans[k].range.lowerBound == i {
                let s = spans[k]
                for j in 0..<s.range.count {
                    axes[0].append(p)
                    axes[1].append(p + Int32(j / s.columns))
                    axes[2].append(p + Int32(j % s.columns))
                }
                p += Int32(max(1, s.rows, s.columns))
                i += s.range.count
                k += 1
                continue
            }
            for a in 0..<3 { axes[a].append(p) }
            p += 1
            i += 1
        }
        return axes.flatMap { $0 }
    }
}

extension Krea2Pipeline {
    /// Whether the export takes reference images (a `krea2.references` block, its `vision` graph and the text
    /// encoder's references function).
    public nonisolated var supportsReferences: Bool { krea2.references != nil }

    /// How the pipeline would render `prompt` with `references` (in prompt order), on the CPU: the images are
    /// preprocessed and each image pad expanded, but no graph loads and the content filter isn't consulted. `tokenIDs`
    /// has each image's rows as the pad id; `conditioningRows` counts them, so a host can order renders by it. Only
    /// the text is ever cut (``Krea2PreparedPrompt/truncated``). Throws `unsupportedOperation` for an export without
    /// references, and what ``encode(prompt:references:job:)`` would refuse.
    public nonisolated func prepare(prompt: String, references: [Krea2ReferenceImage]) throws -> Krea2PreparedPrompt {
        let built = try referencePrompt(prompt, references: references)
        return Krea2PreparedPrompt(
            tokenIDs: built.prompt.ids, renderedTokenCount: built.prompt.renderedTokenCount,
            conditioningRows: built.prompt.conditioningRows, truncated: built.prompt.truncated,
            maximumTokens: krea2.prompt.maxTokens)
    }

    nonisolated func referencePrompt(
        _ text: String, references images: [Krea2ReferenceImage]
    ) throws -> (prompt: Krea2ReferencePrompt, patches: [Krea2ReferencePatches], rules: Manifest.Krea2.References) {
        guard let rules = krea2.references else {
            throw AetherError.unsupportedOperation("this Krea 2 export takes no reference images")
        }
        var patches: [Krea2ReferencePatches] = []
        for image in images { patches.append(try Krea2ReferencePreprocessing.prepare(image, rule: rules.preprocess)) }
        let parts = (0..<images.count).map { rules.label($0 + 1) + "<|vision_start|><|image_pad|><|vision_end|>" }.joined()
        let rendered = try encodePrompt(parts + text)
        let prompt = try Krea2ReferencePrompt(
            rendered: rendered, textOnlyCount: try encodePrompt(text).count, images: patches, prompt: krea2.prompt,
            references: rules)
        try requireTextCapacity(rows: prompt.conditioningRows)
        return (prompt, patches, rules)
    }

    /// Encodes `prompt` with `references` (1…`maxImages`, in prompt order) into conditioning for
    /// ``generate(_:reusing:observing:)``: the vision graph embeds each image, then the text encoder's references
    /// function produces the rows. The filter reviews the prompt text first; reference images are the host's to vet.
    /// The conditioning carries the images' digests in order (``Krea2Conditioning/referenceDigests``), so a host can
    /// key its cache by them. GPU, turn, admission, retention, pressure and cancellation are ``encode(prompt:job:)``'s.
    public func encode(prompt text: String, references: [Krea2ReferenceImage], job: GPUJob? = nil) async throws -> Krea2Conditioning {
        try await encode(prompt: text, references: references, canvas: nil, job: job)
    }

    /// ``encode(prompt:references:job:)`` for the generation that follows at `width` × `height`, as
    /// ``encode(prompt:width:height:job:)`` is for a text-only prompt: the encode plans against that canvas, so the
    /// denoisers and decoder the generation keeps are never released by its change of canvas (the first encode on a new
    /// pipeline, or one for another size). The canvas is checked as a generation's would be (`invalidParameters`).
    public func encode(
        prompt text: String, references: [Krea2ReferenceImage], width: Int, height: Int, job: GPUJob? = nil
    ) async throws -> Krea2Conditioning {
        _ = try latent.tokenGrid(width: width, height: height)
        return try await encode(prompt: text, references: references, canvas: (width, height), job: job)
    }

    private func encode(
        prompt text: String, references: [Krea2ReferenceImage], canvas: (width: Int, height: Int)?, job: GPUJob?
    ) async throws -> Krea2Conditioning {
        pending += 1
        defer { pending -= 1 }
        try Task.checkCancellation()
        let (prompt, patches, rules) = try referencePrompt(text, references: references)
        try await filter.requireAllowed(prompt: text)
        try Task.checkCancellation()
        let grant = try await options.gpu.admit(job: job)
        defer { grant.release() }
        return try await gate.withTurn {
            let residency = try admission.begin(profile: encodeProfile(canvas: canvas), shape: denoiserShape(rows: prompt.conditioningRows))
            if let pressure = options.pressure,
               !Set([wiring.text.role, rules.vision]).isSubset(of: Set(admission.residentRoles)),
               await pressure.level == .critical {
                admission.release()
                throw AetherError.memoryPressure
            }
            do {
                // The reference conditioning graphs (the vision tower and the text encoder, whose references function
                // takes its own copy of the weights when it runs) are loaded for this encode alone: the text-only
                // encoder is released first and everything this phase loads after it, so the denoisers the last
                // generation kept never share memory with them.
                residency.releasePhase(.perPrompt)
                try await residency.beginPhase(.perPrompt)
                let arena = TensorArena()
                let embedded = try await embed(patches, rules: rules, residency: residency, arena: arena)
                let rows = try await encodeReferences(prompt, embedded: embedded, rules: rules, residency: residency, arena: arena)
                residency.sample()
                let conditioning = Krea2Conditioning(
                    prompt: text, rows: prompt.conditioningRows, columns: krea2.conditioning.columns,
                    truncated: prompt.truncated, referenceDigests: references.map(\.digest), export: export,
                    scalarType: rows.scalarType, bytes: try rows.readBytes())
                residency.endPhase(.perPrompt)
                residency.releasePhase(.perPrompt)
                try Task.checkCancellation()
                admission.finish()
                if retention == .releaseAfterEachGeneration { admission.release() }
                return conditioning
            } catch {
                settle(after: error)
                throw error
            }
        }
    }

    /// Each image through the vision graph; the image embeddings and each deepstack level, slot-indexed in prompt
    /// order (`[slot, hidden]`, `[level, slot, hidden]` for the text encoder's static capacity).
    private func embed(
        _ images: [Krea2ReferencePatches], rules: Manifest.Krea2.References, residency: GraphResidency, arena: TensorArena
    ) async throws -> (image: [Float], deepstack: [Float], hidden: Int, levels: Int) {
        guard let spec = manifest.graphs[rules.vision], let patchesName = spec.inputs["patches"]?.name,
              let gridName = spec.inputs["grid"]?.name, let embedsName = spec.outputs["imageEmbeds"]?.name,
              let deepName = spec.outputs["deepstackEmbeds"]?.name else {
            throw AetherError.manifestInvalid(field: "graphs.\(rules.vision)", reason: "needs patches, grid, imageEmbeds, deepstackEmbeds")
        }
        let vision = try await residency.graph(rules.vision)
        var image: [Float] = [], deepstack: [[Float]] = []
        var hidden = 0, levels = 0
        for patches in images {
            try Task.checkCancellation()
            let input = try vision.makeBuffer(for: patchesName, role: .input, dimensions: [], in: arena)
            let grid = try vision.makeBuffer(for: gridName, role: .input, dimensions: [], in: arena)
            let embeds = try vision.makeBuffer(for: embedsName, role: .output, dimensions: [], in: arena)
            let deep = try vision.makeBuffer(for: deepName, role: .output, dimensions: [], in: arena)
            let capacity = input.shape.first ?? 0
            guard input.shape.count == 2, input.shape[1] == rules.preprocess.rowLength, patches.patches <= capacity,
                  embeds.shape.count == 2, embeds.shape[0] >= patches.tokens, deep.shape.count == 3,
                  deep.shape[1] == embeds.shape[0], deep.shape[2] == embeds.shape[1], grid.count == 2 else {
                throw AetherError.manifestMismatch(
                    field: "graphs.\(rules.vision)",
                    expected: "patches [≥\(patches.patches), \(rules.preprocess.rowLength)], grid [2] → embeds [≥\(patches.tokens), H], deepstack [L, ≥\(patches.tokens), H]",
                    found: "\(input.shape), \(grid.shape) → \(embeds.shape), \(deep.shape)")
            }
            hidden = embeds.shape[1]
            levels = deep.shape[0]
            if deepstack.isEmpty { deepstack = Array(repeating: [], count: levels) }
            var padded = patches.values
            padded += [Float](repeating: 0, count: capacity * rules.preprocess.rowLength - padded.count)
            try write(padded, into: input)
            try grid.write([Int32(patches.rows), Int32(patches.columns)])
            var bindings = Bindings(arena: arena)
            try bindings.bindInput(patchesName, to: input)
            try bindings.bindInput(gridName, to: grid)
            try bindings.bindOutput(embedsName, to: embeds)
            try bindings.bindOutput(deepName, to: deep)
            try await vision.run(&bindings)
            residency.sample()
            let n = patches.tokens * hidden, rowsOut = embeds.shape[0]
            image += try embeds.readFloats(0..<n)
            for level in 0..<levels {
                deepstack[level] += try deep.readFloats((level * rowsOut * hidden)..<(level * rowsOut * hidden + n))
            }
        }
        return (image, deepstack.flatMap { $0 }, hidden, levels)
    }

    /// The text encoder's references function over `prompt` and the slot-indexed embeddings: the conditioning rows.
    private func encodeReferences(
        _ prompt: Krea2ReferencePrompt, embedded: (image: [Float], deepstack: [Float], hidden: Int, levels: Int),
        rules: Manifest.Krea2.References, residency: GraphResidency, arena: TensorArena
    ) async throws -> TensorBuffer {
        guard let spec = manifest.graphs[wiring.text.role] else {
            throw AetherError.manifestInvalid(field: "graphs.\(wiring.text.role)", reason: "required")
        }
        let inputs = spec.inputs(forPhase: rules.function)
        guard let tokensName = inputs["tokens"]?.name, let ropeName = inputs["ropePositions"]?.name,
              let embedsName = inputs["imageEmbeds"]?.name, let deepName = inputs["deepstackEmbeds"]?.name else {
            throw AetherError.manifestInvalid(
                field: "graphs.\(wiring.text.role).inputsByFunction.\(rules.function)",
                reason: "needs tokens, ropePositions, imageEmbeds, deepstackEmbeds")
        }
        guard let phased = try await residency.graph(wiring.text.role) as? any PhasedGraphBackend,
              let text = phased.function(forPhase: rules.function) else {
            throw AetherError.manifestInvalid(
                field: "graphs.\(wiring.text.role).functions.\(rules.function)", reason: "the loaded text encoder has no such function")
        }
        let n = prompt.encoderIDs.count, slots = prompt.spans.reduce(0) { $0 + $1.range.count }
        let tokens = try text.makeBuffer(for: tokensName, role: .input, dimensions: [n], in: arena)
        let rope = try text.makeBuffer(for: ropeName, role: .input, dimensions: [n], in: arena)
        let embeds = try text.makeBuffer(for: embedsName, role: .input, dimensions: [], in: arena)
        let deep = try text.makeBuffer(for: deepName, role: .input, dimensions: [], in: arena)
        let rows = try text.makeBuffer(for: wiring.text.conditioning, role: .output, dimensions: [prompt.conditioningRows], in: arena)
        let h = embedded.hidden, capacity = embeds.shape.first ?? 0
        guard tokens.count == n, rope.shape == [3, n], embeds.shape == [capacity, h], slots <= capacity,
              deep.shape == [embedded.levels, capacity, h],
              rows.shape == [prompt.conditioningRows, krea2.conditioning.columns] else {
            throw AetherError.manifestMismatch(
                field: "graphs.\(wiring.text.role).inputsByFunction.\(rules.function)",
                expected: "tokens [1, \(n)], rope [3, \(n)], embeds [≥\(slots), \(h)], deepstack [\(embedded.levels), ≥\(slots), \(h)] → [\(prompt.conditioningRows), \(krea2.conditioning.columns)]",
                found: "\(tokens.shape), \(rope.shape), \(embeds.shape), \(deep.shape) → \(rows.shape)")
        }
        try tokens.write(prompt.encoderIDs)
        try rope.write(prompt.ropePositions)
        try write(embedded.image + [Float](repeating: 0, count: (capacity - slots) * h), into: embeds)
        var levels: [Float] = []
        for level in 0..<embedded.levels {
            levels += embedded.deepstack[(level * slots * h)..<((level + 1) * slots * h)]
            levels += [Float](repeating: 0, count: (capacity - slots) * h)
        }
        try write(levels, into: deep)
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(tokensName, to: tokens)
        try bindings.bindInput(ropeName, to: rope)
        try bindings.bindInput(embedsName, to: embeds)
        try bindings.bindInput(deepName, to: deep)
        try bindings.bindOutput(wiring.text.conditioning, to: rows)
        try await text.run(&bindings)
        return rows
    }

    /// Writes floats into a float16 or float32 buffer.
    private nonisolated func write(_ values: [Float], into buffer: TensorBuffer) throws(AetherError) {
        switch buffer.scalarType {
        case .float16: try buffer.write(Float16.self, count: values.count) { Float16(values[$0]) }
        default: try buffer.write(values)
        }
    }
}
