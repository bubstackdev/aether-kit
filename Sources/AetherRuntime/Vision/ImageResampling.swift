import Accelerate
import Foundation

/// The resampling filters of `preprocess.image.resize.filter` (MANIFEST.md §2.5).
///
/// The 8-bit parity filters are 0.0.x's antialiased bicubic (`Qwen3VLImagePreprocessor.resize`), which computed in
/// Double, moved to each reference's own fixed-point arithmetic so they reproduce it bit for bit: a kernel (Keys' cubic,
/// a = −0.5, or Pillow's triangle) widened by the scale when downscaling, weights normalized per output pixel and then
/// rounded to integers, a horizontal pass and then a vertical one, each rounded to 8 bits. Pillow and torch differ only
/// in the integer precision and in the order the cubic's second piece is evaluated. ``torchBilinear(_:channels:width:height:toWidth:height:)``
/// is the one float filter: torch's float32 bilinear, which never widens its kernel.
enum ImageResampler {
    /// Which reference's 8-bit arithmetic a resize reproduces.
    enum Fixed {
        /// Pillow's `Resample.c` bicubic: 22-bit weights.
        case pillowBicubic
        /// Pillow's `Resample.c` bilinear (its triangle filter, support 1): 22-bit weights.
        case pillowBilinear
        /// torch's `UpSampleKernel.cpp` uint8 bicubic (antialiased): int16 weights at the largest precision (≤ 22 bits)
        /// whose biggest weight still fits.
        case torchUint8Bicubic

        /// The kernel's half-width at scale 1.
        var support: Double { self == .pillowBilinear ? 1 : 2 }

        func kernel(_ x: Double) -> Double {
            switch self {
            case .pillowBicubic: ImageResampler.pillowCubic(x)
            case .pillowBilinear: ImageResampler.pillowTriangle(x)
            case .torchUint8Bicubic: ImageResampler.torchCubic(x)
            }
        }
    }

    /// `image` resampled to `width × height` with `filter`'s arithmetic.
    static func resample(_ image: RGB8Image, width: Int, height: Int, _ filter: Fixed) -> RGB8Image {
        precondition(width > 0 && height > 0)
        var pixels = image.rgb
        var current = (width: image.width, height: image.height)
        if width != current.width {
            pixels = horizontal(pixels, size: current, to: width, coefficients(input: current.width, output: width, filter))
            current.width = width
        }
        if height != current.height {
            pixels = vertical(pixels, size: current, to: height, coefficients(input: current.height, output: height, filter))
            current.height = height
        }
        return try! RGB8Image(rgb: pixels, width: width, height: height)
    }

    /// Per output index, the first input index and its integer weights (`count` of them), at `precision` bits.
    struct Coefficients {
        var first: [Int]
        var weights: [[Int]]
        var precision: Int
    }

    static func coefficients(input: Int, output: Int, _ filter: Fixed) -> Coefficients {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = filter.support * filterScale
        // Both references multiply by the reciprocal rather than divide.
        let inverse = 1 / filterScale
        let maxTaps = Int(support.rounded(.up)) * 2 + 1
        var first: [Int] = [], normalized: [[Double]] = []
        var largest = 0.0
        for i in 0..<output {
            let center = (Double(i) + 0.5) * scale
            // C's truncating casts, as both references write them.
            let lower = max(Int(center - support + 0.5), 0)
            var count = min(Int(center + support + 0.5), input) - lower
            if filter == .torchUint8Bicubic { count = min(max(count, 0), maxTaps) }
            var weights = (0..<count).map { j in filter.kernel((Double(j + lower) - center + 0.5) * inverse) }
            let total = weights.reduce(0, +)
            if total != 0 { weights = weights.map { $0 / total } }
            largest = weights.reduce(largest, max)
            first.append(lower)
            normalized.append(weights)
        }
        let precision: Int
        switch filter {
        case .pillowBicubic, .pillowBilinear:
            precision = 22
        case .torchUint8Bicubic:
            var bits = 0
            while bits < 22, Int(0.5 + largest * Double(1 << (bits + 1))) < (1 << 15) { bits += 1 }
            precision = bits
        }
        let one = Double(1 << precision)
        let weights = normalized.map { row in row.map { w in w < 0 ? Int(-0.5 + w * one) : Int(0.5 + w * one) } }
        return Coefficients(first: first, weights: weights, precision: precision)
    }

    /// Pillow's `bilinear_filter`: the triangle 1 − |x| on (−1, 1).
    static func pillowTriangle(_ value: Double) -> Double {
        let x = abs(value)
        return x < 1 ? 1 - x : 0
    }

    /// Keys' cubic, a = −0.5, in Pillow's operation order.
    static func pillowCubic(_ value: Double) -> Double {
        let a = -0.5, x = abs(value)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    /// Keys' cubic, a = −0.5, in torch's operation order (`cubic_convolution1/2`).
    static func torchCubic(_ value: Double) -> Double {
        let a = -0.5, x = abs(value)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        return 0
    }

    /// Rounds a fixed-point sum: half up through the bias added first, then an arithmetic shift, clamped to 8 bits.
    @inline(__always)
    private static func clamp8(_ sum: Int, _ precision: Int) -> UInt8 {
        UInt8(clamping: sum >> precision)
    }

    private static func horizontal(_ pixels: [UInt8], size: (width: Int, height: Int), to width: Int, _ c: Coefficients) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * size.height * 3)
        let bias = 1 << (c.precision - 1)
        pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<size.height {
                    let row = y * size.width * 3
                    for x in 0..<width {
                        let weights = c.weights[x], start = row + c.first[x] * 3
                        var r = bias, g = bias, b = bias
                        for (k, w) in weights.enumerated() {
                            let s = start + 3 * k
                            r += Int(src[s]) * w
                            g += Int(src[s + 1]) * w
                            b += Int(src[s + 2]) * w
                        }
                        let d = (y * width + x) * 3
                        dst[d] = clamp8(r, c.precision)
                        dst[d + 1] = clamp8(g, c.precision)
                        dst[d + 2] = clamp8(b, c.precision)
                    }
                }
            }
        }
        return out
    }

    private static func vertical(_ pixels: [UInt8], size: (width: Int, height: Int), to height: Int, _ c: Coefficients) -> [UInt8] {
        let rowBytes = size.width * 3
        var out = [UInt8](repeating: 0, count: rowBytes * height)
        let bias = 1 << (c.precision - 1)
        pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<height {
                    let weights = c.weights[y], start = c.first[y] * rowBytes
                    for i in 0..<rowBytes {
                        var sum = bias
                        for (k, w) in weights.enumerated() { sum += Int(src[start + k * rowBytes + i]) * w }
                        dst[y * rowBytes + i] = clamp8(sum, c.precision)
                    }
                }
            }
        }
        return out
    }

    // MARK: - torch's float32 bilinear

    /// Per output index, torch's two source indices and their weights (`compute_source_index_and_lambda`).
    struct LinearTaps {
        var lower: [Int], upper: [Int]
        var lowerWeight: [Float], upperWeight: [Float]
    }

    /// `area_pixel_compute_source_index` (align_corners false) in Float, as torch's CPU build computes it: the scale
    /// `Float(input) / Float(output)`, the source `scale · (i + 0.5) − 0.5` fused into one rounding (clang contracts it on
    /// arm64), clamped at 0; the index floored and capped at the last pixel, the upper weight `src − index` in [0, 1].
    static func linearTaps(input: Int, output: Int) -> LinearTaps {
        var taps = LinearTaps(lower: [], upper: [], lowerWeight: [], upperWeight: [])
        if input == output {
            taps.lower = Array(0..<output)
            taps.upper = taps.lower
            taps.lowerWeight = [Float](repeating: 1, count: output)
            taps.upperWeight = [Float](repeating: 0, count: output)
            return taps
        }
        let scale = Float(input) / Float(output)
        for i in 0..<output {
            let source = max(Float(-0.5).addingProduct(scale, Float(i) + 0.5), 0)
            let index = min(Int(source.rounded(.down)), input - 1)
            let upper = min(max(source - Float(index), 0), 1)
            taps.lower.append(index)
            taps.upper.append(index + (index < input - 1 ? 1 : 0))
            taps.upperWeight.append(upper)
            taps.lowerWeight.append(1 - upper)
        }
        return taps
    }

    /// Planar float32 `planes` (`channels` planes of `width × height`) resampled to `toWidth × toHeight` as torch's CPU
    /// `upsample_bilinear2d` (align_corners false, no antialias; torchvision's `resize(…, antialias=False)` of a float
    /// tensor) computes it, to the bit. torch picks its kernel by the output size: up to 128 for height + width, its
    /// channels-last kernel, which sums the four corners with combined weights; above, its separable generic kernel,
    /// a row, then the column. Either way the products are fused into the running sum as clang contracts them. (A
    /// single-threaded torch, `torch.set_num_threads(1)`, takes the channels-last kernel at every size for 3 channels;
    /// this is torch's default multi-threaded choice.)
    static func torchBilinear(_ planes: [Float], channels: Int, width: Int, height: Int, toWidth: Int, height toHeight: Int) -> [Float] {
        precondition(toWidth > 0 && toHeight > 0 && planes.count == channels * width * height)
        let rows = linearTaps(input: height, output: toHeight), columns = linearTaps(input: width, output: toWidth)
        let cornerKernel = toWidth + toHeight <= 128
        var out = [Float](repeating: 0, count: channels * toWidth * toHeight)
        planes.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for c in 0..<channels {
                    let plane = c * width * height, outPlane = c * toWidth * toHeight
                    for y in 0..<toHeight {
                        let top = plane + rows.lower[y] * width, bottom = plane + rows.upper[y] * width
                        let h0 = rows.lowerWeight[y], h1 = rows.upperWeight[y]
                        for x in 0..<toWidth {
                            let left = columns.lower[x], right = columns.upper[x]
                            let w0 = columns.lowerWeight[x], w1 = columns.upperWeight[x]
                            let value: Float
                            if cornerKernel {
                                // `i00 * w00 + i01 * w01 + i10 * w10 + i11 * w11`, the first product fused into the second.
                                var sum = (src[top + right] * (h0 * w1)).addingProduct(src[top + left], h0 * w0)
                                sum = sum.addingProduct(src[bottom + left], h1 * w0)
                                value = sum.addingProduct(src[bottom + right], h1 * w1)
                            } else {
                                // `t * wts + t2 * wts2` per axis, rows outermost.
                                let upper = (src[top + right] * w1).addingProduct(src[top + left], w0)
                                let lower = (src[bottom + right] * w1).addingProduct(src[bottom + left], w0)
                                value = (lower * h1).addingProduct(upper, h0)
                            }
                            dst[outPlane + y * toWidth + x] = value
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: - vImage

    /// `image` resampled to `width × height` with Accelerate's `vImageScale_ARGB8888` (its default Lanczos kernel, edges
    /// extended): the fast path, not a parity filter.
    static func vImage(_ image: RGB8Image, width: Int, height: Int) throws(AetherError) -> RGB8Image {
        if width == image.width && height == image.height { return image }
        var source = [UInt8](repeating: 255, count: image.width * image.height * 4)
        var scaled = [UInt8](repeating: 0, count: width * height * 4)
        var out = [UInt8](repeating: 0, count: width * height * 3)
        var rgb = image.rgb
        let error = rgb.withUnsafeMutableBytes { rgbBytes in
            source.withUnsafeMutableBytes { sourceBytes in
                scaled.withUnsafeMutableBytes { scaledBytes in
                    out.withUnsafeMutableBytes { outBytes -> vImage_Error in
                        var packed = vImage_Buffer(data: rgbBytes.baseAddress, height: vImagePixelCount(image.height),
                                                   width: vImagePixelCount(image.width), rowBytes: image.width * 3)
                        var four = vImage_Buffer(data: sourceBytes.baseAddress, height: vImagePixelCount(image.height),
                                                 width: vImagePixelCount(image.width), rowBytes: image.width * 4)
                        var resized = vImage_Buffer(data: scaledBytes.baseAddress, height: vImagePixelCount(height),
                                                    width: vImagePixelCount(width), rowBytes: width * 4)
                        var result = vImage_Buffer(data: outBytes.baseAddress, height: vImagePixelCount(height),
                                                   width: vImagePixelCount(width), rowBytes: width * 3)
                        var error = vImageConvert_RGB888toRGBA8888(&packed, nil, 255, &four, false, vImage_Flags(kvImageNoFlags))
                        guard error == kvImageNoError else { return error }
                        error = vImageScale_ARGB8888(&four, &resized, nil, vImage_Flags(kvImageEdgeExtend))
                        guard error == kvImageNoError else { return error }
                        return vImageConvert_RGBA8888toRGB888(&resized, &result, vImage_Flags(kvImageNoFlags))
                    }
                }
            }
        }
        guard error == kvImageNoError else {
            throw .runtime(underlying: NSErrorFromVImage(error))
        }
        return try RGB8Image(rgb: out, width: width, height: height)
    }

    private static func NSErrorFromVImage(_ code: vImage_Error) -> NSError {
        NSError(domain: "vImage", code: code, userInfo: [NSLocalizedDescriptionKey: "vImage failed with \(code)"])
    }
}
