/// Qwen3-VL image preprocessing, reproducing the reference fast image processor
/// (`Qwen2VLImageProcessorFast`): smart resize to multiples of `patchSize · mergeSize` within the
/// pixel bounds, antialiased bicubic resampling of the 8-bit image, (x/255 − mean)/std, the frame
/// repeated `temporalPatchSize` times, and rows in merge-window order. Pure and Core-AI-independent.
public enum Qwen3VLImagePreprocessor {
    public struct Configuration: Sendable, Equatable, Codable {
        public var patchSize = 16
        public var mergeSize = 2
        public var temporalPatchSize = 2
        public var minimumPixels = 65_536
        public var maximumPixels = 16_777_216
        public var mean: [Float] = [0.5, 0.5, 0.5]
        public var standardDeviation: [Float] = [0.5, 0.5, 0.5]

        public init() {}
        public var factor: Int { patchSize * mergeSize }
        /// Values per pixel row of the encoder input: C · T · patch².
        public var rowLength: Int { 3 * temporalPatchSize * patchSize * patchSize }
    }

    public enum Failure: Error, Equatable {
        case invalidImage(width: Int, height: Int)
        case aspectRatio(Double)
    }

    /// The processor's `smart_resize`: each side rounds to the nearest multiple of the factor
    /// (ties to even); an area above the maximum scales both sides down and floors them, one below
    /// the minimum scales them up and ceils them. Aspect ratios above 200 are refused.
    public static func targetSize(width: Int, height: Int, configuration: Configuration = .init()) throws -> (width: Int, height: Int) {
        guard width > 0, height > 0 else { throw Failure.invalidImage(width: width, height: height) }
        let ratio = Double(max(width, height)) / Double(min(width, height))
        guard ratio <= 200 else { throw Failure.aspectRatio(ratio) }
        let factor = Double(configuration.factor)
        var h = (Double(height) / factor).rounded(.toNearestOrEven) * factor
        var w = (Double(width) / factor).rounded(.toNearestOrEven) * factor
        if h * w > Double(configuration.maximumPixels) {
            let beta = (Double(height) * Double(width) / Double(configuration.maximumPixels)).squareRoot()
            h = max(factor, (Double(height) / beta / factor).rounded(.down) * factor)
            w = max(factor, (Double(width) / beta / factor).rounded(.down) * factor)
        } else if h * w < Double(configuration.minimumPixels) {
            let beta = (Double(configuration.minimumPixels) / (Double(height) * Double(width))).squareRoot()
            h = (Double(height) * beta / factor).rounded(.up) * factor
            w = (Double(width) * beta / factor).rounded(.up) * factor
        }
        return (Int(w), Int(h))
    }

    /// Antialiased bicubic resampling (Keys a = −0.5, kernel widened by the scale when downscaling,
    /// PIL's coefficient layout) of interleaved RGB8, horizontal pass then vertical, each rounded to
    /// 8 bits.
    public static func resize(rgb: [UInt8], width: Int, height: Int, to target: (width: Int, height: Int)) -> [UInt8] {
        precondition(rgb.count == width * height * 3 && target.width > 0 && target.height > 0)
        var image = rgb
        var currentWidth = width
        if target.width != width {
            let (bounds, weights) = coefficients(input: width, output: target.width)
            var out = [UInt8](repeating: 0, count: target.width * height * 3)
            for y in 0..<height {
                for x in 0..<target.width {
                    let (start, count) = bounds[x]
                    for c in 0..<3 {
                        var sum = 0.0
                        for k in 0..<count { sum += weights[x][k] * Double(image[(y * width + start + k) * 3 + c]) }
                        out[(y * target.width + x) * 3 + c] = clamp8(sum)
                    }
                }
            }
            image = out
            currentWidth = target.width
        }
        if target.height != height {
            let (bounds, weights) = coefficients(input: height, output: target.height)
            var out = [UInt8](repeating: 0, count: currentWidth * target.height * 3)
            for y in 0..<target.height {
                let (start, count) = bounds[y]
                for x in 0..<currentWidth {
                    for c in 0..<3 {
                        var sum = 0.0
                        for k in 0..<count { sum += weights[y][k] * Double(image[((start + k) * currentWidth + x) * 3 + c]) }
                        out[(y * currentWidth + x) * 3 + c] = clamp8(sum)
                    }
                }
            }
            image = out
        }
        return image
    }

    /// Normalized patch rows [patches, rowLength] in merge-window order, and the patch grid
    /// (1, H/patch, W/patch). The image's sides must be multiples of `patchSize · mergeSize`.
    public static func patchify(rgb: [UInt8], width: Int, height: Int, configuration: Configuration = .init()) throws
        -> (pixels: [Float], grid: (t: Int, h: Int, w: Int)) {
        let p = configuration.patchSize, m = configuration.mergeSize, t = configuration.temporalPatchSize
        guard rgb.count == width * height * 3, width % (p * m) == 0, height % (p * m) == 0 else {
            throw Failure.invalidImage(width: width, height: height)
        }
        let gh = height / p, gw = width / p
        var pixels = [Float](repeating: 0, count: gh * gw * configuration.rowLength)
        let scale = (0..<3).map { 1 / (255 * configuration.standardDeviation[$0]) }
        let shift = (0..<3).map { configuration.mean[$0] / configuration.standardDeviation[$0] }
        var row = 0
        for windowRow in 0..<(gh / m) { for windowColumn in 0..<(gw / m) { for i in 0..<m { for j in 0..<m {
            let top = (windowRow * m + i) * p, left = (windowColumn * m + j) * p
            var offset = row * configuration.rowLength
            for c in 0..<3 {
                for _ in 0..<t {
                    for y in 0..<p {
                        for x in 0..<p {
                            pixels[offset] = Float(rgb[((top + y) * width + left + x) * 3 + c]) * scale[c] - shift[c]
                            offset += 1
                        }
                    }
                }
            }
            row += 1
        } } } }
        return (pixels, (1, gh, gw))
    }

    /// Resize (when needed) and patchify interleaved RGB8.
    public static func process(rgb: [UInt8], width: Int, height: Int, configuration: Configuration = .init()) throws
        -> (pixels: [Float], grid: (t: Int, h: Int, w: Int)) {
        let target = try targetSize(width: width, height: height, configuration: configuration)
        let resized = target == (width, height) ? rgb : resize(rgb: rgb, width: width, height: height, to: target)
        return try patchify(rgb: resized, width: target.width, height: target.height, configuration: configuration)
    }

    static func cubic(_ value: Double) -> Double {
        let a = -0.5, x = abs(value)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    /// Per output index: (first input index, tap count) and normalized weights, as PIL computes them.
    static func coefficients(input: Int, output: Int) -> ([(Int, Int)], [[Double]]) {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        var bounds: [(Int, Int)] = [], weights: [[Double]] = []
        for i in 0..<output {
            let center = (Double(i) + 0.5) * scale
            let lower = max(Int(center - support + 0.5), 0)
            let end = min(Int(center + support + 0.5), input)
            var w = (lower..<end).map { cubic((Double($0) - center + 0.5) / filterScale) }
            let total = w.reduce(0, +)
            if total != 0 { w = w.map { $0 / total } }
            bounds.append((lower, end - lower))
            weights.append(w)
        }
        return (bounds, weights)
    }

    private static func clamp8(_ value: Double) -> UInt8 { UInt8(max(0, min(255, value.rounded()))) }
}
