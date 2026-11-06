public import AetherCore
public import CoreGraphics

/// Where the source image sits in a ``PreparedImage``, so a driver can map model-space geometry back to the image.
public struct ImageGeometry: Sendable, Hashable {
    /// The decoded, upright image.
    public let sourceWidth: Int
    public let sourceHeight: Int
    /// The prepared image (the tensor's H and W).
    public let width: Int
    public let height: Int
    /// The whole source image's rectangle in prepared pixels: the full canvas when stretched, inset when letterboxed,
    /// larger than the canvas when centre-cropped.
    public let content: CGRect

    public init(sourceWidth: Int, sourceHeight: Int, width: Int, height: Int, content: CGRect) {
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.width = width
        self.height = height
        self.content = content
    }

    /// `rect` in prepared pixels, as a fraction of the source image ([0, 1] when inside it; not clipped).
    public func normalizedSourceRect(fromPrepared rect: CGRect) -> CGRect {
        CGRect(x: (rect.minX - content.minX) / content.width, y: (rect.minY - content.minY) / content.height,
               width: rect.width / content.width, height: rect.height / content.height)
    }
}

/// An image as its graph input: the CPU half of a vision request, made before the GPU half so a real-time caller can
/// prepare the next frame while the last one runs (DESIGN.md §8).
public struct PreparedImage: Sendable {
    enum Values: Sendable {
        case float32([Float])
        case float16([Float16])
    }

    /// `[1, 3, H, W]`, `[1, V, 3, H, W]` or `[rows, dim]` (the manifest's layout).
    public let shape: [Int]
    let values: Values
    public let geometry: ImageGeometry

    /// `float32` or `float16`: the manifest's `hostDType`.
    public var scalarType: ScalarType {
        switch values {
        case .float32: .float32
        case .float16: .float16
        }
    }

    /// The values as Float, in layout order.
    public func floats() -> [Float] {
        switch values {
        case .float32(let v): v
        case .float16(let v): v.map(Float.init)
        }
    }
}

/// `preprocess.image` (MANIFEST.md §2.5) as a transform: decode, resize, normalize, lay out. One implementation on
/// every platform; the only platform library it leans on is Accelerate (the `vImage` filter, fp16 conversion).
public struct ImagePreprocessor: Sendable {
    public let configuration: Manifest.ImagePreprocess

    /// Checks `configuration` (errors name `preprocess.image.*`).
    public init(_ configuration: Manifest.ImagePreprocess) throws(AetherError) {
        try configuration.validate()
        self.configuration = configuration
    }

    /// The manifest's `preprocess.image`, which must be present.
    public init(manifest: Manifest) throws(AetherError) {
        guard let image = manifest.preprocess?.image else {
            throw .manifestInvalid(field: "preprocess.image", reason: "\(manifest.driver) requires it")
        }
        try self.init(image)
    }

    /// Decodes `input` and prepares it. Errors: `invalidInput` for an image that doesn't decode or that the layout
    /// can't take (a patch grid the size doesn't divide).
    public func prepare(_ input: ImageInput) throws(AetherError) -> PreparedImage {
        try prepare(input.decode(colorSpace: configuration.colorSpace ?? .deviceRGB))
    }

    func prepare(_ image: RGB8Image) throws(AetherError) -> PreparedImage {
        let (canvas, content) = try resized(image)
        let geometry = ImageGeometry(sourceWidth: image.width, sourceHeight: image.height, width: canvas.width,
                                     height: canvas.height, content: content)
        let (shape, values) = try laidOut(canvas)
        let prepared: PreparedImage.Values = switch configuration.hostDType ?? .float32 {
        case .float16: .float16(Self.halves(values))
        default: .float32(values)
        }
        return PreparedImage(shape: shape, values: prepared, geometry: geometry)
    }

    // MARK: - Resize

    /// The resized canvas and the source's rectangle on it.
    func resized(_ image: RGB8Image) throws(AetherError) -> (RGB8Image, CGRect) {
        let w = image.width, h = image.height
        guard let resize = configuration.resize else {
            return (image, CGRect(x: 0, y: 0, width: w, height: h))
        }
        switch resize.mode {
        case .stretch:
            let size = resize.size!
            return (try resample(image, size.width, size.height, resize.filter),
                    CGRect(x: 0, y: 0, width: size.width, height: size.height))
        case .letterbox:
            let size = resize.size!
            // Integer arithmetic, so the fitting side is exactly the canvas's.
            let (nw, nh) = size.width * h <= size.height * w
                ? (size.width, max(1, h * size.width / w)) : (max(1, w * size.height / h), size.height)
            let scaled = try resample(image, nw, nh, resize.filter)
            let (ox, oy) = resize.anchor == .topLeft ? (0, 0) : ((size.width - nw) / 2, (size.height - nh) / 2)
            let pad = resize.padValue ?? [0]
            let fill = (0..<3).map { UInt8(pad.count == 3 ? pad[$0] : pad[0]) }
            var canvas = [UInt8](repeating: 0, count: size.width * size.height * 3)
            for i in 0..<(size.width * size.height) {
                canvas[3 * i] = fill[0]
                canvas[3 * i + 1] = fill[1]
                canvas[3 * i + 2] = fill[2]
            }
            for y in 0..<nh {
                let d = ((y + oy) * size.width + ox) * 3, s = y * nw * 3
                canvas.replaceSubrange(d..<(d + nw * 3), with: scaled.rgb[s..<(s + nw * 3)])
            }
            return (try RGB8Image(rgb: canvas, width: size.width, height: size.height),
                    CGRect(x: ox, y: oy, width: nw, height: nh))
        case .centerCrop:
            let size = resize.size!
            let (rw, rh): (Int, Int)
            if let side = resize.shortestSide {
                // torchvision's `Resize(side)`: the shorter side to `side`, the longer one floored.
                (rw, rh) = w <= h ? (side, max(side, h * side / w)) : (max(side, w * side / h), side)
            } else {
                (rw, rh) = size.width * h >= size.height * w
                    ? (size.width, max(size.height, h * size.width / w)) : (max(size.width, w * size.height / h), size.height)
            }
            let scaled = try resample(image, rw, rh, resize.filter)
            // torchvision's `int(round((scaled - size) / 2.0))`: Python rounds ties to even.
            let cx = Int((Double(rw - size.width) / 2).rounded(.toNearestOrEven))
            let cy = Int((Double(rh - size.height) / 2).rounded(.toNearestOrEven))
            var canvas = [UInt8](repeating: 0, count: size.width * size.height * 3)
            for y in 0..<size.height {
                let s = ((y + cy) * rw + cx) * 3, d = y * size.width * 3
                canvas.replaceSubrange(d..<(d + size.width * 3), with: scaled.rgb[s..<(s + size.width * 3)])
            }
            return (try RGB8Image(rgb: canvas, width: size.width, height: size.height),
                    CGRect(x: -cx, y: -cy, width: rw, height: rh))
        case .fitLongSide:
            let limit = resize.maxSide!
            guard max(w, h) > limit else { return (image, CGRect(x: 0, y: 0, width: w, height: h)) }
            let (nw, nh) = w >= h ? (limit, max(1, h * limit / w)) : (max(1, w * limit / h), limit)
            return (try resample(image, nw, nh, resize.filter), CGRect(x: 0, y: 0, width: nw, height: nh))
        }
    }

    private func resample(
        _ image: RGB8Image, _ width: Int, _ height: Int, _ filter: Manifest.ImagePreprocess.Resize.Filter
    ) throws(AetherError) -> RGB8Image {
        if width == image.width && height == image.height { return image }
        switch filter {
        case .vImage: return try ImageResampler.vImage(image, width: width, height: height)
        case .pillowBicubic: return ImageResampler.bicubic(image, width: width, height: height, reference: .pillow)
        case .torchUint8Bicubic: return ImageResampler.bicubic(image, width: width, height: height, reference: .torchUint8)
        }
    }

    // MARK: - Normalize and lay out

    /// Per output channel, the normalized value of each 8-bit level: `(x · scale − mean) / std` in Float, the
    /// operation order of torchvision's rescale then normalize.
    func levels() -> [[Float]] {
        let scale = Float(configuration.scale ?? 1)
        let mean = (configuration.mean ?? [0, 0, 0]).map(Float.init), std = (configuration.std ?? [1, 1, 1]).map(Float.init)
        return (0..<3).map { c in
            let source = configuration.channelOrder == .BGR ? 2 - c : c
            return (0..<256).map { (Float($0) * scale - mean[source]) / std[source] }
        }
    }

    /// The source channel of output channel `c`.
    private func sourceChannel(_ c: Int) -> Int { configuration.channelOrder == .BGR ? 2 - c : c }

    func laidOut(_ image: RGB8Image) throws(AetherError) -> ([Int], [Float]) {
        let levels = levels()
        let w = image.width, h = image.height, plane = w * h
        let rgb = image.rgb
        switch configuration.layout ?? .NCHW {
        case .NCHW, .NVCHW:
            var chw = [Float](repeating: 0, count: 3 * plane)
            for c in 0..<3 {
                let s = sourceChannel(c), table = levels[c]
                for i in 0..<plane { chw[c * plane + i] = table[Int(rgb[3 * i + s])] }
            }
            if configuration.layout == .NVCHW {
                let views = configuration.views ?? 1
                return ([1, views, 3, h, w], Array([[Float]](repeating: chw, count: views).joined()))
            }
            return ([1, 3, h, w], chw)
        case .patches:
            return try patchify(image, levels: levels)
        }
    }

    private func patchify(_ image: RGB8Image, levels: [[Float]]) throws(AetherError) -> ([Int], [Float]) {
        let spec = configuration.patchify!
        let p = spec.patch, t = spec.temporal ?? 1, m = spec.merge ?? 1
        let w = image.width, h = image.height
        guard w % (p * m) == 0, h % (p * m) == 0 else {
            throw .invalidInput("a \(w)x\(h) image doesn't divide into \(p * m)-pixel patch blocks")
        }
        let gh = h / p, gw = w / p, dim = 3 * t * p * p
        // The patch grid positions in row order.
        var order: [(row: Int, column: Int)] = []
        order.reserveCapacity(gh * gw)
        switch spec.rowOrder {
        case .blockMajor:
            for br in 0..<(gh / m) { for bc in 0..<(gw / m) { for i in 0..<m { for j in 0..<m {
                order.append((br * m + i, bc * m + j))
            } } } }
        case .rowMajor:
            for r in 0..<gh { for c in 0..<gw { order.append((r, c)) } }
        }
        var out = [Float](repeating: 0, count: order.count * dim)
        var offset = 0
        for (row, column) in order {
            let top = row * p, left = column * p
            switch spec.withinPatch {
            case .channelFramePixel:
                for c in 0..<3 {
                    let s = sourceChannel(c), table = levels[c]
                    for _ in 0..<t {
                        for y in 0..<p {
                            for x in 0..<p {
                                out[offset] = table[Int(image.rgb[((top + y) * w + left + x) * 3 + s])]
                                offset += 1
                            }
                        }
                    }
                }
            case .pixelChannel:
                for y in 0..<p {
                    for x in 0..<p {
                        for c in 0..<3 {
                            out[offset] = levels[c][Int(image.rgb[((top + y) * w + left + x) * 3 + sourceChannel(c)])]
                            offset += 1
                        }
                    }
                }
            }
        }
        return ([order.count, dim], out)
    }

    /// Float to Float16, rounding to nearest even.
    static func halves(_ values: [Float]) -> [Float16] { values.map { Float16($0) } }
}

extension GraphBackend {
    /// A buffer for input `name` holding `image`, converted to the input's own floating-point type. The graph's shape
    /// must be the image's (dynamic dimensions take the image's); errors are `manifestMismatch` naming the input.
    public func makeBuffer(for name: String, holding image: PreparedImage, in arena: TensorArena) throws(AetherError) -> TensorBuffer {
        guard let spec = signature.input(name) else { throw .tensor(name: name, reason: .unknownName) }
        guard spec.rank == image.shape.count,
              zip(spec.shape, image.shape).allSatisfy({ $0 == TensorSpec.dynamic || $0 == $1 }) else {
            throw .manifestMismatch(field: "preprocess.image", expected: "the graph input '\(name)' shape \(spec.shape)",
                                    found: "a prepared image of \(image.shape)")
        }
        let dynamic = spec.dynamicAxes.map { image.shape[$0] }
        let buffer = try makeBuffer(for: name, role: .input, dimensions: dynamic, in: arena)
        switch (buffer.scalarType, image.values) {
        case (.float32, .float32(let v)): try buffer.write(v)
        case (.float16, .float16(let v)): try buffer.write(v)
        case (.float16, .float32(let v)): try buffer.write(Float16.self, count: v.count) { Float16(v[$0]) }
        case (.float32, .float16(let v)): try buffer.write(Float.self, count: v.count) { Float(v[$0]) }
        default:
            throw .manifestMismatch(field: "graphs.*.inputs", expected: "a float16 or float32 image input '\(name)'",
                                    found: buffer.scalarType.rawValue)
        }
        return buffer
    }
}
