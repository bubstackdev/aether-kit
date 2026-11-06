import CoreGraphics
import CryptoKit
import Foundation
import Testing
import AetherHost
@testable import AetherRuntime
@testable import AetherVision

/// The Depth Anything 3 small gate (DESIGN.md §12): the exported `depth-estimator@1` bundle, run through the driver on
/// the GPU, against the unmodified upstream model on the CPU in float32 (`Forge/depth/reference_depth.py`) on the same
/// preprocessed tensor. Off by default; run with `AETHER_GATES=1 swift test --no-parallel --filter DepthEstimatorGateTests`.
///
/// - **Graph parity** (M4.md's bar): each case's `input.f32` straight into the graph, depth and confidence rel L2 ≤ 0.02,
///   and a repeat bit-identical.
/// - **The same pixels end to end:** each image case's input rebuilt as an 8-bit 504 x 504 image (exact: the reference's
///   input is Pillow's 8-bit resize / 255) and estimated from `ImageInput`, so decode, preprocess and postprocess are
///   in the path; the same bar.
/// - **The source frames end to end:** upstream's SOH frames (1208 x 680) estimated from their files, resized back to the
///   source: finite, the source's size, and the polarity (the sky, top, farther than the foreground, bottom). The rel L2
///   to the reference resized back the same way is reported, not gated: Kit stretches with vImage, the reference with
///   Pillow's bicubic. The frames come from `Forge/depth/fetch_upstream.py`'s directory
///   (`~/.cache/aether-kit/depth/upstream/depth-anything-3-<commit>`) or `AETHER_DEPTH_FRAMES`, checked by SHA-256; absent,
///   that part is skipped with a note.
///
/// The bundle and references default to the Forge outputs beside this repo (`aether/forge-outputs/depth-anything-3-small`),
/// or `AETHER_DEPTH_BUNDLE` and `AETHER_DEPTH_REFERENCE`.
@Suite("Depth estimator gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct DepthEstimatorGateTests {
    static let environment = ProcessInfo.processInfo.environment
    static let outputs = forgeOutputs().appending(path: "depth-anything-3-small")
    static let bundleURL = environment["AETHER_DEPTH_BUNDLE"].map { URL(filePath: $0) } ?? outputs.appending(path: "bundle")
    static let referenceURL = environment["AETHER_DEPTH_REFERENCE"].map { URL(filePath: $0) } ?? outputs.appending(path: "reference")
    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)
    static let bar = 0.02

    struct Index: Decodable {
        struct Source: Decodable { struct Code: Decodable { var commit: String }; var upstreamCode: Code }
        struct Entry: Decodable { var name: String }
        var source: Source
        var shapes: [String: [Int]]
        var cases: [Entry]
    }

    struct Case: Decodable {
        struct Source: Decodable { var kind: String; var upstreamFile: String?; var sha256: String? }
        var name: String
        var source: Source
    }

    /// A Forge export is laid out as an installed revision root; until it is catalogued its ref is local.
    static func localBundle(_ url: URL) throws -> InstalledBundle {
        let manifest = try Manifest.load(fromBundle: url)
        let ref = ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: manifest.variant.platform.rawValue,
            driver: manifest.driver, assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    static func floats(_ url: URL) throws -> [Float] {
        try Data(contentsOf: url).withUnsafeBytes { raw in
            (0..<(raw.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }

    /// ‖a − b‖ / ‖b‖ and the largest absolute difference.
    static func compare(_ a: [Float], _ b: [Float]) -> (relL2: Double, maxAbs: Double) {
        var diff = 0.0, norm = 0.0, worst = 0.0
        for (x, y) in zip(a, b) {
            let d = Double(x) - Double(y)
            diff += d * d
            norm += Double(y) * Double(y)
            worst = max(worst, abs(d))
        }
        return ((diff / max(norm, .leastNonzeroMagnitude)).squareRoot(), worst)
    }

    /// An NCHW [1, 3, H, W] tensor on [0, 1] as the 8-bit sRGB image it was made from.
    static func image(fromInput input: [Float], width: Int, height: Int) throws -> ImageInput {
        let plane = width * height
        var bytes = [UInt8](repeating: 255, count: 4 * plane)
        for i in 0..<plane {
            for c in 0..<3 { bytes[4 * i + c] = UInt8((input[c * plane + i] * 255).rounded()) }
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = try #require(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4 * width,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        return ImageInput(image)
    }

    /// The mean of the top and bottom tenth of a map's rows.
    static func bands(_ map: DepthMap) -> (top: Double, bottom: Double) {
        let rows = max(1, map.height / 10)
        func mean(_ range: Range<Int>) -> Double {
            Double(range.reduce(Float(0)) { $0 + map.values[($1 * map.width)..<(($1 + 1) * map.width)].reduce(0, +) })
                / Double(range.count * map.width)
        }
        return (mean(0..<rows), mean((map.height - rows)..<map.height))
    }

    @Test func depthAnything3SmallMatchesUpstream() async throws {
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: Self.referenceURL.appending(path: "references.json")))
        let model = try await Registry.vision.load(
            try Self.localBundle(Self.bundleURL), as: DepthEstimatorDriver.self,
            options: LoadOptions(device: Self.mac, cachePolicy: .default, cache: .default))
        let estimator = try model.estimator()
        #expect(estimator.postprocess == .init(polarity: .largerIsFarther, resizeBack: true))
        let shape = try #require(index.shapes["input"])
        let (height, width) = (shape[2], shape[3])
        let frames = Self.environment["AETHER_DEPTH_FRAMES"].map { URL(filePath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(
                path: ".cache/aether-kit/depth/upstream/depth-anything-3-\(index.source.upstreamCode.commit.prefix(12))")

        for entry in index.cases {
            let directory = Self.referenceURL.appending(path: entry.name)
            let c = try JSONDecoder().decode(Case.self, from: Data(contentsOf: directory.appending(path: "case.json")))
            let input = try Self.floats(directory.appending(path: "input.f32"))
            let depth = try Self.floats(directory.appending(path: "depth.f32"))
            let confidence = try Self.floats(directory.appending(path: "confidence.f32"))
            #expect(input.count == 3 * width * height && depth.count == width * height && confidence.count == depth.count)

            // (a) Graph parity: the reference tensor straight in.
            let geometry = ImageGeometry(sourceWidth: width, sourceHeight: height, width: width, height: height,
                                         content: CGRect(x: 0, y: 0, width: width, height: height))
            let prepared = PreparedImage(shape: shape, values: .float32(input), geometry: geometry)
            let started = ContinuousClock.now
            let map = try await estimator.estimate(prepared, DepthRequest(resizeBack: false))
            let elapsed = ContinuousClock.now - started
            #expect(map.width == width && map.height == height && map.polarity == .largerIsFarther)
            let d = Self.compare(map.values, depth)
            let k = Self.compare(try #require(map.confidence), confidence)
            print("gate: depth \(entry.name) graph: depth rel L2 \(d.relL2) (max |Δ| \(d.maxAbs)), confidence rel L2 "
                  + "\(k.relL2) (max |Δ| \(k.maxAbs)), \(elapsed)")
            #expect(d.relL2 <= Self.bar, "\(entry.name): depth rel L2")
            #expect(k.relL2 <= Self.bar, "\(entry.name): confidence rel L2")
            let repeated = try await estimator.estimate(prepared, DepthRequest(resizeBack: false))
            #expect(repeated == map, "\(entry.name): repeat bit-identical")

            // The image cases' inputs are 8-bit pixels / 255; the synthetic case's are not.
            guard c.source.kind == "image", let file = c.source.upstreamFile, let sha = c.source.sha256 else { continue }

            // (b) The same pixels from an ImageInput: decode, preprocess (an identity stretch) and postprocess.
            let pixels = try await estimator.estimate(Self.image(fromInput: input, width: width, height: height))
            #expect(pixels.isSourceSized && pixels.width == width)
            let p = Self.compare(pixels.values, depth)
            print("gate: depth \(entry.name) pixels end to end: depth rel L2 \(p.relL2) (max |Δ| \(p.maxAbs))")
            #expect(p.relL2 <= Self.bar, "\(entry.name): end-to-end rel L2 from the same pixels")

            // (c) The source frame, resized back.
            let url = frames.appending(path: file)
            guard let data = try? Data(contentsOf: url) else {
                print("gate: depth \(entry.name): no source frame at \(url.path) (Forge/depth/fetch_upstream.py); skipped")
                continue
            }
            #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == sha, "\(file): SHA-256")
            let frameStarted = ContinuousClock.now
            let frame = try await estimator.estimate(ImageInput(contentsOf: url))
            let frameElapsed = ContinuousClock.now - frameStarted
            let g = frame.geometry
            #expect(frame.isSourceSized && frame.values.count == g.sourceWidth * g.sourceHeight)
            #expect(frame.values.allSatisfy(\.isFinite) && frame.confidence?.allSatisfy(\.isFinite) == true)
            let (top, bottom) = Self.bands(frame)
            #expect(top > 1.5 * bottom, "\(entry.name): the sky is farther than the foreground (largerIsFarther)")
            let resizedReference = DepthResampler(g).resample(depth)
            let f = Self.compare(frame.values, resizedReference)
            print("gate: depth \(entry.name) frame \(g.sourceWidth)x\(g.sourceHeight) end to end: rel L2 to the resized "
                  + "reference \(f.relL2) (not gated: vImage vs Pillow bicubic), top/bottom band means \(top) / \(bottom), "
                  + "range \(frame.range.map { "\($0)" } ?? "none"), \(frameElapsed)")
        }
        for (key, footprint) in estimator.options.ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: depth \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
        }
        await estimator.unload()
        #expect(await estimator.residentRoles.isEmpty)
    }
}

/// The Forge outputs directory beside the repo: the first `forge-outputs` found walking up from this file, so the gates
/// find it from a worktree under `.worktrees/` as from the main checkout (`AETHER_FORGE_OUTPUTS` overrides).
func forgeOutputs(from file: String = #filePath) -> URL {
    if let path = ProcessInfo.processInfo.environment["AETHER_FORGE_OUTPUTS"] { return URL(filePath: path) }
    var directory = URL(filePath: file).deletingLastPathComponent()
    while directory.path != "/" {
        let candidate = directory.appending(path: "forge-outputs")
        if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        directory = directory.deletingLastPathComponent()
    }
    return URL(filePath: file).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appending(path: "forge-outputs")
}
