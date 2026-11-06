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
/// - **The host's preprocessing, apart from the decode:** the RGB Pillow decoded for the reference (`decoded.png`,
///   lossless and untagged) through the bundle's `preprocess.image` (`pillowBicubic`, the 504 x 504 stretch) is the
///   reference's `input.f32` bit for bit at the graph's float16 input, so its depth is the graph path's exactly.
/// - **End to end from Pillow's pixels:** `decoded.png` estimated from ``ImageInput`` and resized back to the source
///   (1208 x 680), against the reference resized back the same way: the same bar.
/// - **The source frames end to end:** upstream's SOH frames estimated from their files (ImageIO's decode), resized
///   back: finite, the source's size, and the polarity (the sky, top, farther than the foreground, bottom). Where ImageIO
///   decodes the very bytes Pillow does (the frames are PNGs), the tensor must be the reference's and the rel L2 to the
///   resized reference is gated at the same bar; where it does not (a JPEG source), it is reported, the resize being
///   gated by the step above. The frames come from `Forge/depth/fetch_upstream.py`'s directory
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

    /// How many values differ, and the largest difference.
    static func differences<T: BinaryFloatingPoint>(_ a: [T], _ b: [T]) -> (count: Int, max: Double) {
        guard a.count == b.count else { return (max(a.count, b.count), .infinity) }
        var count = 0, largest = 0.0
        for (x, y) in zip(a, b) where x != y {
            count += 1
            largest = max(largest, abs(Double(x) - Double(y)))
        }
        return (count, largest)
    }

    static func differences(_ a: [UInt8], _ b: [UInt8]) -> (count: Int, max: Double) {
        guard a.count == b.count else { return (max(a.count, b.count), .infinity) }
        var count = 0, largest = 0
        for i in a.indices where a[i] != b[i] {
            count += 1
            largest = max(largest, abs(Int(a[i]) - Int(b[i])))
        }
        return (count, Double(largest))
    }

    /// Host float32 values as the graph's float16 input sees them (the reference's `input.f32` is fp16-representable).
    static func float16(_ values: [Float]) -> [Float] { values.map { Float(Float16($0)) } }

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

            guard c.source.kind == "image", let file = c.source.upstreamFile, let sha = c.source.sha256 else { continue }

            // (b) Pillow's decode through the host's preprocessing (`pillowBicubic`'s 504 x 504 stretch): the reference
            // tensor bit for bit at the graph's float16 input, so the graph sees what upstream saw.
            let preprocessor = try ImagePreprocessor(manifest: model.manifest)
            let pillowRGB = try ImageInput(contentsOf: directory.appending(path: "decoded.png")).decode(colorSpace: .deviceRGB)
            let fromPillow = try preprocessor.prepare(pillowRGB)
            let resizeDiff = Self.differences(Self.float16(fromPillow.floats()), input)
            print("gate: depth \(entry.name) Pillow's decode (\(pillowRGB.width)x\(pillowRGB.height)) through preprocess.image: "
                  + "\(resizeDiff.count) values differ from input.f32 at float16 (max |Δ| \(resizeDiff.max))")
            #expect(fromPillow.shape == shape, "\(entry.name): prepared shape")
            #expect(resizeDiff.count == 0, "\(entry.name): \(resizeDiff.count) values differ from the reference tensor")
            let pillowMap = try await estimator.estimate(fromPillow, DepthRequest(resizeBack: false))
            #expect(pillowMap.values == map.values && pillowMap.confidence == map.confidence,
                    "\(entry.name): depth from Pillow's pixels is the graph path's, bit for bit")

            // (c) End to end from Pillow's pixels: decode, preprocess, graph and the resize back to the source, against
            // the reference resized back the same way.
            let pixels = try await estimator.estimate(ImageInput(contentsOf: directory.appending(path: "decoded.png")))
            let g = pixels.geometry
            #expect(pixels.isSourceSized && g.sourceWidth == pillowRGB.width && g.sourceHeight == pillowRGB.height)
            let resizedReference = DepthResampler(g).resample(depth)
            let p = Self.compare(pixels.values, resizedReference)
            print("gate: depth \(entry.name) Pillow's pixels end to end \(g.sourceWidth)x\(g.sourceHeight): rel L2 to the "
                  + "resized reference \(p.relL2) (max |Δ| \(p.maxAbs))")
            #expect(p.relL2 <= Self.bar, "\(entry.name): end-to-end rel L2 from Pillow's pixels")

            // (d) The source frame, ImageIO's decode, resized back: gated at the same bar where ImageIO decodes the very
            // pixels Pillow does (the SOH frames are PNGs); otherwise (a JPEG) reported only, the resize gated by (b).
            let url = frames.appending(path: file)
            guard let data = try? Data(contentsOf: url) else {
                print("gate: depth \(entry.name): no source frame at \(url.path) (Forge/depth/fetch_upstream.py); skipped")
                continue
            }
            #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == sha, "\(file): SHA-256")
            let source = ImageInput(contentsOf: url)
            let decodeDiff = Self.differences(try source.decode(colorSpace: .deviceRGB).rgb, pillowRGB.rgb)
            let framePrepared = try estimator.prepare(source)
            let tensorDiff = Self.differences(Self.float16(framePrepared.floats()), input)
            let frameStarted = ContinuousClock.now
            let frame = try await estimator.estimate(source)
            let frameElapsed = ContinuousClock.now - frameStarted
            #expect(frame.geometry == g && frame.isSourceSized && frame.values.count == g.sourceWidth * g.sourceHeight)
            #expect(frame.values.allSatisfy(\.isFinite) && frame.confidence?.allSatisfy(\.isFinite) == true)
            let (top, bottom) = Self.bands(frame)
            #expect(top > 1.5 * bottom, "\(entry.name): the sky is farther than the foreground (largerIsFarther)")
            let f = Self.compare(frame.values, resizedReference)
            print("gate: depth \(entry.name) frame \(g.sourceWidth)x\(g.sourceHeight) end to end: rel L2 to the resized "
                  + "reference \(f.relL2) (max |Δ| \(f.maxAbs)); ImageIO vs Pillow decode \(decodeDiff.count) bytes differ "
                  + "(max \(Int(decodeDiff.max))), tensor \(tensorDiff.count) values differ (max \(tensorDiff.max)); "
                  + "top/bottom band means \(top) / \(bottom), range \(frame.range.map { "\($0)" } ?? "none"), \(frameElapsed)")
            if decodeDiff.count == 0 {
                #expect(tensorDiff.count == 0, "\(entry.name) end to end: the same decode must give the reference tensor")
                #expect(f.relL2 <= Self.bar, "\(entry.name) end to end: rel L2 to the resized reference")
            } else {
                print("gate: depth \(entry.name) end to end not gated: ImageIO's decode is not Pillow's (the resize is gated above)")
            }
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
