import Accelerate
import Foundation

/// The resampling filters of `preprocess.image.resize.filter` on 8-bit RGB (MANIFEST.md §2.5).
///
/// The parity filters are 0.0.x's antialiased bicubic (`Qwen3VLImagePreprocessor.resize`), which computed in Double,
/// moved to each reference's own fixed-point arithmetic so they reproduce it bit for bit: Keys' cubic (a = −0.5) with
/// the kernel widened by the scale when downscaling, weights normalized per output pixel and then rounded to integers,
/// a horizontal pass and then a vertical one, each rounded to 8 bits. Pillow and torch differ only in the integer
/// precision and in the order the cubic's second piece is evaluated.
enum ImageResampler {
    /// Which reference's arithmetic a bicubic resize reproduces.
    enum Reference {
        /// Pillow's `Resample.c`: 22-bit weights.
        case pillow
        /// torch's `UpSampleKernel.cpp` uint8 path: int16 weights at the largest precision (≤ 22 bits) whose biggest
        /// weight still fits.
        case torchUint8
    }

    /// `image` resampled to `width × height` with the antialiased bicubic of `reference`.
    static func bicubic(_ image: RGB8Image, width: Int, height: Int, reference: Reference) -> RGB8Image {
        precondition(width > 0 && height > 0)
        var pixels = image.rgb
        var current = (width: image.width, height: image.height)
        if width != current.width {
            pixels = horizontal(pixels, size: current, to: width, coefficients(input: current.width, output: width, reference))
            current.width = width
        }
        if height != current.height {
            pixels = vertical(pixels, size: current, to: height, coefficients(input: current.height, output: height, reference))
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

    static func coefficients(input: Int, output: Int, _ reference: Reference) -> Coefficients {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
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
            if reference == .torchUint8 { count = min(max(count, 0), maxTaps) }
            var weights = (0..<count).map { j -> Double in
                let x = (Double(j + lower) - center + 0.5) * inverse
                return reference == .pillow ? pillowCubic(x) : torchCubic(x)
            }
            let total = weights.reduce(0, +)
            if total != 0 { weights = weights.map { $0 / total } }
            largest = weights.reduce(largest, max)
            first.append(lower)
            normalized.append(weights)
        }
        let precision: Int
        switch reference {
        case .pillow:
            precision = 22
        case .torchUint8:
            var bits = 0
            while bits < 22, Int(0.5 + largest * Double(1 << (bits + 1))) < (1 << 15) { bits += 1 }
            precision = bits
        }
        let one = Double(1 << precision)
        let weights = normalized.map { row in row.map { w in w < 0 ? Int(-0.5 + w * one) : Int(0.5 + w * one) } }
        return Coefficients(first: first, weights: weights, precision: precision)
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
