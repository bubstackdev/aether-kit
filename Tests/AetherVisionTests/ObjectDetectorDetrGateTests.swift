import CoreGraphics
import CryptoKit
import Foundation
import Testing
import AetherHost
@testable import AetherRuntime
@testable import AetherVision

/// The RF-DETR gates (DESIGN.md §12), nano and seg-nano: the exported `detector-detr@1` bundles, run through the driver
/// on the GPU, against the unmodified upstream model on the CPU in float32 (`Forge/rfdetr/reference_rfdetr.py`) on the
/// same input tensor. Off by default; run with
/// `AETHER_GATES=1 swift test --no-parallel --filter ObjectDetectorDetrGateTests`.
///
/// - **The postprocess against upstream's:** the reference's raw outputs through the driver's `detrSigmoid` head give
///   exactly upstream `PostProcess`'s detections (`case.json`): the same classes in the same order, scores within 1e-6,
///   boxes within 1e-5.
/// - **Detection parity** (M4.md's bar): each case's `input.f32` straight into the graph, against the reference's raw
///   outputs through the same head: the same class set; detections matched class by class (greedy on box IoU > 0.5),
///   every pair box IoU ≥ 0.95, |score difference| ≤ 0.01 and (seg-nano) mask IoU ≥ 0.95; an unmatched detection only
///   within 0.01 of the threshold; a repeat bit-identical.
/// - **The host's preprocessing, apart from the decode:** the RGB Pillow decoded for the reference (`decoded.png`,
///   lossless and untagged) through the bundle's `preprocess.image` (`torchBilinear`) is the reference's `input.f32` bit
///   for bit, so its detections are the graph path's exactly.
/// - **End to end, gated at the same bars:** the source photos (`Forge/rfdetr/fetch_upstream.py`'s
///   `~/.cache/aether-kit/rfdetr/images`, or `AETHER_RFDETR_IMAGES`, checked by SHA-256) detected from ``ImageInput``
///   (ImageIO's decode, converted to sRGB) at their own size, against the reference outputs mapped to the same source.
///   How far ImageIO's decode is from Pillow's (bytes) and the stretched tensor from the reference's is printed, so a
///   decode difference is told apart from a resize one.
/// - **Reported:** each image case's input rebuilt as an 8-bit image (the reference's input is a float resize, so this
///   rounds it) and detected from `ImageInput`.
///
/// The bundles and references default to the Forge outputs beside this repo (`aether/forge-outputs/rf-detr-nano`,
/// `rf-detr-seg-nano`), or `AETHER_RFDETR_OUTPUTS` (the directory holding both).
@Suite("RF-DETR gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct ObjectDetectorDetrGateTests {
    static let environment = ProcessInfo.processInfo.environment
    static let outputs = environment["AETHER_RFDETR_OUTPUTS"].map { URL(filePath: $0) } ?? forgeOutputs()
    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)
    static let iouBar = 0.95, scoreBar: Float = 0.01, maskBar = 0.95

    struct Index: Decodable {
        struct Entry: Decodable { var name: String }
        var shapes: [String: [Int]]
        var cases: [Entry]
    }

    struct Case: Decodable {
        struct Source: Decodable { var kind: String; var file: String?; var sha256: String? }
        struct Reference: Decodable { var score: Float; var classId: Int; var label: String; var boxXYXY: [Double] }
        var name: String
        var source: Source
        var detections: [Reference]
    }

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

    /// An NCHW [1, 3, H, W] tensor on [0, 1] as an 8-bit sRGB image, rounded.
    static func image(fromInput input: [Float], width: Int, height: Int) throws -> ImageInput {
        let plane = width * height
        var bytes = [UInt8](repeating: 255, count: 4 * plane)
        for i in 0..<plane {
            for c in 0..<3 { bytes[4 * i + c] = UInt8((min(max(input[c * plane + i], 0), 1) * 255).rounded()) }
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = try #require(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4 * width,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        return ImageInput(image)
    }

    /// How many values differ, and by how much at most.
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

    static func iou(_ a: CGRect, _ b: CGRect) -> Double {
        let i = a.intersection(b)
        guard !i.isNull else { return 0 }
        let inter = Double(i.width * i.height)
        let union = Double(a.width * a.height + b.width * b.height) - inter
        return union > 0 ? inter / union : 1
    }

    /// Detections matched class by class, greedily by box IoU (> 0.5).
    struct Match {
        var classSetEqual = true
        var pairs = 0
        var minIoU = 1.0, maxScoreDiff: Float = 0, minMaskIoU: Double?
        /// Unmatched detections not within the score bar of the threshold.
        var hardUnmatched: [String] = []
        var borderline = 0

        var passes: Bool {
            classSetEqual && minIoU >= ObjectDetectorDetrGateTests.iouBar && maxScoreDiff <= ObjectDetectorDetrGateTests.scoreBar
                && (minMaskIoU ?? 1) >= ObjectDetectorDetrGateTests.maskBar && hardUnmatched.isEmpty
        }

        var summary: String {
            "classes \(classSetEqual ? "same" : "DIFFER"), \(pairs) pairs, min IoU \(String(format: "%.6f", minIoU)), max |Δscore| "
                + "\(String(format: "%.2g", maxScoreDiff))" + (minMaskIoU.map { ", min mask IoU \(String(format: "%.6f", $0))" } ?? "")
                + ", unmatched \(hardUnmatched.count) (+\(borderline) borderline)"
        }
    }

    static func match(_ got: [Detection], _ want: [Detection], threshold: Float) -> Match {
        var m = Match()
        m.classSetEqual = Set(got.map(\.label)) == Set(want.map(\.label))
        func unmatched(_ d: Detection, _ side: String) {
            if abs(d.score - threshold) <= scoreBar { m.borderline += 1 } else { m.hardUnmatched.append("\(side) \(d.label.name) \(d.score)") }
        }
        for label in Set(got.map(\.label)).union(want.map(\.label)) {
            let g = got.filter { $0.label == label }, w = want.filter { $0.label == label }
            var candidates: [(Double, Int, Int)] = []
            for (i, a) in g.enumerated() { for (j, b) in w.enumerated() { candidates.append((iou(a.box, b.box), i, j)) } }
            var usedG = Set<Int>(), usedW = Set<Int>()
            for (score, i, j) in candidates.sorted(by: { $0.0 > $1.0 }) where score > 0.5 && !usedG.contains(i) && !usedW.contains(j) {
                usedG.insert(i)
                usedW.insert(j)
                m.pairs += 1
                m.minIoU = min(m.minIoU, score)
                m.maxScoreDiff = max(m.maxScoreDiff, abs(g[i].score - w[j].score))
                if let a = g[i].mask, let b = w[j].mask { m.minMaskIoU = min(m.minMaskIoU ?? 1, a.intersectionOverUnion(b)) }
            }
            for (k, d) in g.enumerated() where !usedG.contains(k) { unmatched(d, "got") }
            for (k, d) in w.enumerated() where !usedW.contains(k) { unmatched(d, "want") }
        }
        return m
    }

    @Test(arguments: ["rf-detr-nano", "rf-detr-seg-nano"])
    func rfDetrMatchesUpstream(_ id: String) async throws {
        let directory = Self.outputs.appending(path: id)
        let referenceURL = directory.appending(path: "reference")
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: referenceURL.appending(path: "references.json")))
        let model = try await Registry.vision.load(
            try Self.localBundle(directory.appending(path: "bundle")), as: DetrDetectorDriver.self,
            options: LoadOptions(device: Self.mac, cachePolicy: .default, cache: .default))
        let detector = try model.detector()
        guard case .detrSigmoid(let post)? = detector.manifest.postprocess else { Issue.record("not detrSigmoid"); return }
        let segmented = index.shapes["masks"] != nil
        #expect(post.classIndex == .cocoId && post.selection == .topKPairs && (post.masks != nil) == segmented)
        guard case .detr(let head) = try ObjectDetector.Configuration(model.manifest).head else { Issue.record("not detr"); return }
        func decode(_ raw: [String: DetectorOutput], _ geometry: ImageGeometry) -> [Detection] {
            head.detections(raw, geometry: geometry, scoreThreshold: post.defaults.scoreThreshold, maxDetections: post.defaults.maxDetections)
        }
        let threshold = Float(post.defaults.scoreThreshold)
        let shape = try #require(index.shapes["input"])
        let size = shape[3]
        let images = Self.environment["AETHER_RFDETR_IMAGES"].map { URL(filePath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/aether-kit/rfdetr/images")

        for entry in index.cases {
            let caseURL = referenceURL.appending(path: entry.name)
            let c = try JSONDecoder().decode(Case.self, from: Data(contentsOf: caseURL.appending(path: "case.json")))
            let input = try Self.floats(caseURL.appending(path: "input.f32"))
            var raw: [String: DetectorOutput] = [:]
            for (role, file) in [("dets", "dets"), ("labels", "labels"), ("masks", "masks")] where index.shapes[file] != nil {
                raw[role] = DetectorOutput(shape: index.shapes[file]!, values: try Self.floats(caseURL.appending(path: "\(file).f32")))
            }
            let geometry = ImageGeometry(sourceWidth: size, sourceHeight: size, width: size, height: size,
                                         content: CGRect(x: 0, y: 0, width: size, height: size))

            // The head on the reference's raw outputs is upstream's PostProcess.
            let reference = decode(raw, geometry)
            #expect(reference.map(\.label.id) == c.detections.map(\.classId), "\(entry.name): upstream's classes, in order")
            var worstScore: Float = 0, worstBox = 0.0
            for (d, u) in zip(reference, c.detections) {
                worstScore = max(worstScore, abs(d.score - u.score))
                let b = [d.box.minX, d.box.minY, d.box.maxX, d.box.maxY]
                worstBox = max(worstBox, zip(b, u.boxXYXY).map { abs(Double($0) - $1) }.max()!)
                #expect(d.label.name == u.label)
            }
            #expect(worstScore <= 1e-6 && worstBox <= 1e-5, "\(entry.name): postprocess vs upstream score \(worstScore), box \(worstBox)")
            #expect(reference.allSatisfy { ($0.mask != nil) == segmented })

            // (a) Detection parity: the reference tensor straight in.
            let prepared = PreparedImage(shape: shape, values: .float32(input), geometry: geometry)
            let started = ContinuousClock.now
            let got = try await detector.detect(prepared)
            let elapsed = ContinuousClock.now - started
            let m = Self.match(got, reference, threshold: threshold)
            print("gate: \(id) \(entry.name) graph: \(got.count) vs \(reference.count) detections, \(m.summary); postprocess vs "
                  + "upstream max |Δscore| \(worstScore), max |Δbox| \(worstBox); \(elapsed)")
            #expect(m.passes, "\(entry.name): \(m.summary) \(m.hardUnmatched)")
            #expect(try await detector.detect(prepared) == got, "\(entry.name): repeat bit-identical")

            guard c.source.kind == "image", let file = c.source.file, let sha = c.source.sha256 else { continue }

            // (b) The same pixels, rounded to 8 bits, from an ImageInput (an identity stretch): reported.
            let pixels = try await detector.detect(Self.image(fromInput: input, width: size, height: size))
            print("gate: \(id) \(entry.name) 8-bit pixels end to end: \(pixels.count) detections, "
                  + Self.match(pixels, reference, threshold: threshold).summary)

            // (c) Pillow's decode through the host's preprocessing: the reference tensor, bit for bit (so the detections
            // are (a)'s, the graph being deterministic).
            let pillowRGB = try ImageInput(contentsOf: caseURL.appending(path: "decoded.png")).decode(colorSpace: .deviceRGB)
            let fromPillow = try ImagePreprocessor(manifest: model.manifest).prepare(pillowRGB)
            let resizeDiff = Self.differences(fromPillow.floats(), input)
            print("gate: \(id) \(entry.name) Pillow's decode \(pillowRGB.width)x\(pillowRGB.height) through preprocess.image: "
                  + "\(resizeDiff.count) values differ from input.f32 (max |Δ| \(resizeDiff.max))")
            #expect(fromPillow.shape == shape && resizeDiff.count == 0, "\(entry.name): \(resizeDiff.count) values differ from the reference tensor")

            // (d) The source photo at its own size, ImageIO's decode: gated at the same bars.
            let url = images.appending(path: file)
            guard let data = try? Data(contentsOf: url) else {
                print("gate: \(id) \(entry.name): no source photo at \(url.path) (Forge/rfdetr/fetch_upstream.py); skipped")
                continue
            }
            #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == sha, "\(file): SHA-256")
            let photoInput = ImageInput(contentsOf: url)
            let colorSpace = model.manifest.preprocess?.image?.colorSpace ?? .deviceRGB
            let decodeDiff = Self.differences(try photoInput.decode(colorSpace: colorSpace).rgb, pillowRGB.rgb)
            let photoPrepared = try detector.prepare(photoInput)
            let tensorDiff = Self.differences(photoPrepared.floats(), input)
            let photoStarted = ContinuousClock.now
            let photo = try await detector.detect(photoInput)
            let photoElapsed = ContinuousClock.now - photoStarted
            let g = photoPrepared.geometry
            let mapped = decode(raw, g)
            let end = Self.match(photo, mapped, threshold: threshold)
            #expect(photo.allSatisfy { d in d.mask.map { $0.width == g.sourceWidth && $0.height == g.sourceHeight } ?? true })
            print("gate: \(id) \(entry.name) photo \(g.sourceWidth)x\(g.sourceHeight) end to end: \(photo.count) detections "
                  + "\(Set(photo.map(\.label.name)).sorted()), vs the reference mapped to the photo: \(end.summary); ImageIO "
                  + "(\(colorSpace.rawValue)) vs Pillow decode \(decodeDiff.count) bytes differ (max \(Int(decodeDiff.max))), tensor "
                  + "\(tensorDiff.count) values differ (max \(String(format: "%.3g", tensorDiff.max))); \(photoElapsed)")
            #expect(end.passes, "\(entry.name) end to end: \(end.summary) \(end.hardUnmatched)")
        }
        for (key, footprint) in detector.options.ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: \(id) \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
        }
        await detector.unload()
        #expect(await detector.residentRoles.isEmpty)
    }
}
