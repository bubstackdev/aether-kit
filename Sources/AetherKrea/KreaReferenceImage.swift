// Derived from the Apache-2.0 Aether prototype (AetherVision, AetherGen/VisionReferences). See NOTICE.
import AetherKit
import CoreGraphics
import CryptoKit
import Foundation
import ImageIO

/// The `visionTokens` reference-image contract of the Krea 2 Turbo realism bundle (`referenceImages` and `vision`
/// manifest blocks). Values are fixed: a bundle declaring anything else is refused by `KreaAssets`.
public enum KreaReferenceContract {
    public static let maximumImages = 4
    public static let maximumSide = 384
    public static let maximumImageTokens = 576
    public static let label = "Picture {n}: "
    public static let patchSize = 16
    public static let mergeSize = 2
    public static let temporalPatchSize = 2
    public static let minimumPixels = 65_536
    public static let maximumPixels = 16_777_216
    public static let positionTableSide = 48
    public static let hiddenSize = 2560
    public static let deepstackLevels = 3
    public static let imageToken: Int32 = 151_655
    public static let visionStartToken: Int32 = 151_652
    public static let visionEndToken: Int32 = 151_653
    /// Values per patch row: channels × temporal patch × patch × patch.
    public static let rowLength = 3 * temporalPatchSize * patchSize * patchSize
    /// Encoded source bytes accepted for one reference (before decoding).
    public static let maximumEncodedBytes = 32 * 1024 * 1024
    /// Decoded source sides accepted before fitting (bounds decode memory).
    public static let maximumSourceSide = 8192

    public static func label(_ number: Int) -> String {
        label.replacingOccurrences(of: "{n}", with: String(number))
    }
}

/// One decoded reference: 8-bit RGB, rows top to bottom, EXIF orientation applied. Built only from in-memory bytes;
/// no file path is accepted and nothing is written to disk.
public struct KreaReferenceImage: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let rgb: [UInt8]
    /// SHA-256 over width, height (little-endian Int64) and the RGB bytes, as the prototype's text cache key.
    public let digest: String

    public init(rgb: [UInt8], width: Int, height: Int) throws {
        guard width > 0, height > 0, width <= KreaReferenceContract.maximumSourceSide,
              height <= KreaReferenceContract.maximumSourceSide, rgb.count == width * height * 3 else {
            throw AetherError.invalidRequest("reference RGB buffer does not match \(width)x\(height)")
        }
        self.width = width; self.height = height; self.rgb = rgb
        var hash = SHA256()
        withUnsafeBytes(of: Int64(width).littleEndian) { hash.update(bufferPointer: $0) }
        withUnsafeBytes(of: Int64(height).littleEndian) { hash.update(bufferPointer: $0) }
        rgb.withUnsafeBytes { hash.update(bufferPointer: $0) }
        digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Encoded bytes (PNG, JPEG, HEIC). Pixels are drawn into an 8-bit RGB context in the source's own RGB colour
    /// space when it has one (an untagged or sRGB PNG keeps its exact bytes), else sRGB; alpha is dropped onto black.
    public init(encoded data: Data) throws {
        guard !data.isEmpty, data.count <= KreaReferenceContract.maximumEncodedBytes,
              let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetCount(source) > 0 else {
            throw AetherError.invalidRequest("reference is not a supported image")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let declaredWidth = (properties?[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
        let declaredHeight = (properties?[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
        guard (1...KreaReferenceContract.maximumSourceSide).contains(declaredWidth),
              (1...KreaReferenceContract.maximumSourceSide).contains(declaredHeight) else {
            throw AetherError.invalidRequest("reference size is outside 1...\(KreaReferenceContract.maximumSourceSide)")
        }
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        let image: CGImage?
        if orientation == 1 {
            image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        } else {
            // The thumbnail path applies the EXIF transform; the longest side keeps full resolution.
            let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                            kCGImageSourceCreateThumbnailWithTransform: true,
                                            kCGImageSourceThumbnailMaxPixelSize: max(declaredWidth, declaredHeight)]
            image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        }
        guard let image else { throw AetherError.invalidRequest("reference could not be decoded") }
        try self.init(cgImage: image)
    }

    init(cgImage: CGImage) throws {
        let w = cgImage.width, h = cgImage.height
        guard w > 0, h > 0, w <= KreaReferenceContract.maximumSourceSide, h <= KreaReferenceContract.maximumSourceSide else {
            throw AetherError.invalidRequest("reference size is outside 1...\(KreaReferenceContract.maximumSourceSide)")
        }
        let space = cgImage.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
            ?? CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        var rgbx = [UInt8](repeating: 0, count: w * h * 4)
        let drawn: Bool = rgbx.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                          bytesPerRow: w * 4, space: space,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            context.interpolationQuality = .none
            context.setBlendMode(.copy)
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw AetherError.invalidRequest("reference could not be drawn as 8-bit RGB") }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[3 * i] = rgbx[4 * i]; rgb[3 * i + 1] = rgbx[4 * i + 1]; rgb[3 * i + 2] = rgbx[4 * i + 2]
        }
        try self.init(rgb: rgb, width: w, height: h)
    }
}

/// Normalized patch rows for the vision encoder, in merge-window order, with the unmerged and merged grids.
public struct KreaReferencePixels: Sendable, Equatable {
    /// Unmerged patch grid (time, height, width) in patches.
    public let grid: (t: Int, h: Int, w: Int)
    public let values: [Float]
    public let digest: String
    public var patches: Int { grid.t * grid.h * grid.w }
    /// Merged grid: the image's token raster in the prompt.
    public var mergedGrid: (t: Int, h: Int, w: Int) {
        (grid.t, grid.h / KreaReferenceContract.mergeSize, grid.w / KreaReferenceContract.mergeSize)
    }
    public var tokens: Int { mergedGrid.t * mergedGrid.h * mergedGrid.w }

    public static func == (a: Self, b: Self) -> Bool {
        a.grid == b.grid && a.values == b.values && a.digest == b.digest
    }

    /// Encoder input `pixels` as fp16 `[patches, rowLength]`.
    public var encoderPixels: Float16Tensor {
        get throws { try Float16Tensor(shape: [patches, KreaReferenceContract.rowLength], values: values.map(Float16.init)) }
    }
}

/// Bounded local preparation of `visionTokens` references: fit the longer side to 384 without upscaling, the Qwen-VL
/// smart resize, then normalize and patchify. Every size rule mirrors the prototype (measured against the reference
/// processor), so fixture tensors at multiple-of-32 sizes are bit-exact.
public enum KreaReferencePreprocessing {
    public static func fitted(width: Int, height: Int) -> (width: Int, height: Int) {
        let scale = min(1, Double(KreaReferenceContract.maximumSide) / Double(max(width, height)))
        guard scale < 1 else { return (width, height) }
        return (max(1, Int((Double(width) * scale).rounded())), max(1, Int((Double(height) * scale).rounded())))
    }

    /// The smart-resized size: sides round to multiples of patch·merge (ties to even), then scale into
    /// minimumPixels...maximumPixels. The aspect ratio may not exceed 200.
    public static func targetSize(width: Int, height: Int) throws -> (width: Int, height: Int) {
        guard width > 0, height > 0 else { throw AetherError.invalidRequest("empty reference image") }
        guard Double(max(width, height)) / Double(min(width, height)) <= 200 else {
            throw AetherError.invalidRequest("reference aspect ratio exceeds 200")
        }
        let f = Double(KreaReferenceContract.patchSize * KreaReferenceContract.mergeSize)
        let maxPixels = Double(KreaReferenceContract.maximumPixels)
        let minPixels = min(Double(KreaReferenceContract.minimumPixels), maxPixels)
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

    /// Fit, resize once from the source (antialiased bicubic, a = −0.5), normalize and patchify.
    public static func prepare(_ image: KreaReferenceImage) throws -> KreaReferencePixels {
        try Task.checkCancellation()
        let fit = fitted(width: image.width, height: image.height)
        let size = try targetSize(width: fit.width, height: fit.height)
        let rgb = (size.width, size.height) == (image.width, image.height) ? image.rgb
            : resizeBicubic(image.rgb, width: image.width, height: image.height, to: size.width, size.height)
        try Task.checkCancellation()
        return patchify(rgb, width: size.width, height: size.height, digest: image.digest)
    }

    /// Normalize ((v/255 − 0.5)/0.5, correctly rounded) and lay out rows over merge windows in raster order, each row `[C][T][py][px]`
    /// with the still image repeated over the temporal patch.
    static func patchify(_ rgb: [UInt8], width: Int, height: Int, digest: String) -> KreaReferencePixels {
        let p = KreaReferenceContract.patchSize, m = KreaReferenceContract.mergeSize, tp = KreaReferenceContract.temporalPatchSize
        let gh = height / p, gw = width / p
        let rowLength = KreaReferenceContract.rowLength
        var values = [Float](repeating: 0, count: gh * gw * rowLength)
        // (v/255 − 0.5)/0.5 rounded once: v − 127.5 is exact in Float, so one division matches the reference
        // processor's float32 tensor bit for bit (an affine v·a + b does not, e.g. at 140 and 180).
        let table = (0..<256).map { (Float($0) - 127.5) / 127.5 }
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
        return KreaReferencePixels(grid: (1, gh, gw), values: values, digest: digest)
    }

    struct Taps { var start: Int; var weights: [Float] }

    /// Separable antialiased Keys cubic (a = −0.5): kernels widen by the scale when downscaling, sample centres sit at
    /// pixel centres, and each output is rounded once after the vertical pass.
    static func resizeBicubic(_ rgb: [UInt8], width: Int, height: Int, to outW: Int, _ outH: Int) -> [UInt8] {
        let hTaps = taps(inSize: width, outSize: outW), vTaps = taps(inSize: height, outSize: outH)
        var mid = [Float](repeating: 0, count: outW * height * 3)
        for y in 0..<height {
            let row = y * width * 3
            for x in 0..<outW {
                let t = hTaps[x]
                var r: Float = 0, g: Float = 0, bl: Float = 0
                for (k, w) in t.weights.enumerated() {
                    let i = row + (t.start + k) * 3
                    r += w * Float(rgb[i]); g += w * Float(rgb[i + 1]); bl += w * Float(rgb[i + 2])
                }
                let o = (y * outW + x) * 3
                mid[o] = r; mid[o + 1] = g; mid[o + 2] = bl
            }
        }
        var out = [UInt8](repeating: 0, count: outW * outH * 3)
        for y in 0..<outH {
            let t = vTaps[y]
            for x in 0..<outW {
                var r: Float = 0, g: Float = 0, bl: Float = 0
                for (k, w) in t.weights.enumerated() {
                    let i = ((t.start + k) * outW + x) * 3
                    r += w * mid[i]; g += w * mid[i + 1]; bl += w * mid[i + 2]
                }
                let o = (y * outW + x) * 3
                out[o] = clamp8(r); out[o + 1] = clamp8(g); out[o + 2] = clamp8(bl)
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

/// Host-side per-patch tables for the Qwen3-VL vision encoder, in the pixel tensor's merge-window row order:
/// bilinear neighbours/weights into the learned 48×48 position table (corners aligned, as `linspace`), and each
/// patch's (row, col) for the encoder's 2-D rotary embedding.
public struct KreaPatchTables: Sendable, Equatable {
    public let patches: Int
    public let positionIndex: [Int32]
    public let positionWeight: [Float]
    public let coordinates: [Int32]

    public init(grid: (t: Int, h: Int, w: Int)) throws {
        let (t, h, w) = grid, merge = KreaReferenceContract.mergeSize, side = KreaReferenceContract.positionTableSide
        guard t >= 1, h > 0, w > 0, h % merge == 0, w % merge == 0 else {
            throw AetherError.invalidRequest("patch grid \(h)x\(w) is not a multiple of the merge size")
        }
        let ys = Self.axis(h, side: side), xs = Self.axis(w, side: side)
        var index: [Int32] = [], weight: [Float] = [], coord: [Int32] = []
        index.reserveCapacity(h * w * 4); weight.reserveCapacity(h * w * 4); coord.reserveCapacity(h * w * 2)
        for wr in 0..<(h / merge) {
            for wc in 0..<(w / merge) {
                for i in 0..<merge {
                    for j in 0..<merge {
                        let r = wr * merge + i, c = wc * merge + j
                        let (y0, y1, dy) = ys[r], (x0, x1, dx) = xs[c]
                        index += [Int32(y0 * side + x0), Int32(y0 * side + x1), Int32(y1 * side + x0), Int32(y1 * side + x1)]
                        weight += [Float((1 - dy) * (1 - dx)), Float((1 - dy) * dx), Float(dy * (1 - dx)), Float(dy * dx)]
                        coord += [Int32(r), Int32(c)]
                    }
                }
            }
        }
        patches = t * h * w
        positionIndex = Array([[Int32]](repeating: index, count: t).joined())
        positionWeight = Array([[Float]](repeating: weight, count: t).joined())
        coordinates = Array([[Int32]](repeating: coord, count: t).joined())
    }

    static func axis(_ n: Int, side: Int) -> [(Int, Int, Double)] {
        let step = n > 1 ? Double(side - 1) / Double(n - 1) : 0
        return (0..<n).map { i in
            let position = n > 1 && i == n - 1 ? Double(side - 1) : Double(i) * step
            let lo = Int(position.rounded(.down))
            return (lo, min(lo + 1, side - 1), position - Double(lo))
        }
    }

    /// Vision encoder inputs other than `pixels`.
    public var encoderInputs: [String: TensorValue] {
        get throws {
            [
                "pos_index": .int32(try Int32Tensor(shape: [patches, 4], values: positionIndex)),
                "pos_weight": .float32(try FloatTensor(shape: [patches, 4], values: positionWeight)),
                "patch_rc": .int32(try Int32Tensor(shape: [patches, 2], values: coordinates)),
            ]
        }
    }
}
