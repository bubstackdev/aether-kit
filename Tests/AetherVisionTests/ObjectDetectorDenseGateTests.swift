import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import Testing
import AetherHost
@testable import AetherRuntime
@testable import AetherVision

/// The YOLOX-S gate (DESIGN.md §12): the exported `detector-dense@1` bundle, run through the driver on the GPU, against
/// the unmodified upstream model on the CPU in float32 with the reference NMS (`Forge/yolox/reference_yolox.py`) on the
/// same letterboxed tensor. Off by default; run with
/// `AETHER_GATES=1 swift test --no-parallel --filter ObjectDetectorDenseGateTests`.
///
/// - **Host NMS:** the driver's ``DenseNMS`` on each case's reference `preds.f32` keeps exactly the reference's
///   detections (anchors, classes, scores and boxes), so the postprocess is gated apart from the graph.
/// - **Detection parity** (M4.md's bar): each case's `input.f32` straight into the detector, with the reference's
///   letterbox geometry: the same class set; each reference detection paired with the unpaired same-class detection of
///   highest IoU, every pair IoU ≥ 0.95 and scores within 0.01; a repeat bit-identical. The boxes are compared in the
///   normalized source image, the reference's mapped through the same geometry; how far that is from the reference's
///   own `boxSourceNormalized` (upstream divides by the ratio r, the kit by the content rectangle) is reported.
/// - **The host's preprocessing, apart from the decode:** the RGB Pillow decoded for the reference (`decoded.png`,
///   lossless and untagged) through the bundle's `preprocess.image` (`pillowBilinear`, YOLOX's letterbox sizes) is the
///   reference's `input.f32` bit for bit, with its geometry, so its detections are the graph path's exactly.
/// - **End to end:** the upstream images (`assets/dog.jpg`, `assets/demo.png` and its crops) detected from
///   ``ImageInput`` (ImageIO's decode), against the reference detections through the same geometry. Where ImageIO
///   decodes the very bytes Pillow does (the PNG cases), gated at the same bars as the tensor path; where it does not
///   (`dog.jpg`: the two JPEG decoders differ by up to 24 levels, enough to move scores by about 0.02), printed with
///   the decode difference, the resize being gated by the step above. Always: upstream's README sanity (the dog and
///   the bicycle in `dog.jpg`). The images come from `Forge/yolox/fetch_upstream.py`'s directory
///   (`~/.cache/aether-kit/yolox/upstream/yolox-<commit>`) or `AETHER_YOLOX_UPSTREAM`, checked by SHA-256; absent, that
///   part is skipped with a note.
///
/// The bundle and references default to the Forge outputs beside this repo (`aether/forge-outputs/yolox-s`), or
/// `AETHER_YOLOX_BUNDLE` and `AETHER_YOLOX_REFERENCE`.
@Suite("Object detector (dense) gates", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct ObjectDetectorDenseGateTests {
    static let environment = ProcessInfo.processInfo.environment
    static let outputs = forgeOutputs().appending(path: "yolox-s")
    static let bundleURL = environment["AETHER_YOLOX_BUNDLE"].map { URL(filePath: $0) } ?? outputs.appending(path: "bundle")
    static let referenceURL = environment["AETHER_YOLOX_REFERENCE"].map { URL(filePath: $0) } ?? outputs.appending(path: "reference")
    static let mac = Device(platform: .macOS, architecture: Device.current.architecture)
    static let iouBar = 0.95, scoreBar = 0.01

    struct Index: Decodable {
        struct Source: Decodable { struct Code: Decodable { var commit: String }; var upstreamCode: Code }
        struct Entry: Decodable { var name: String }
        var source: Source
        var shapes: [String: [Int]]
        var cases: [Entry]
    }

    struct Case: Decodable {
        struct Letterbox: Decodable { var source: [Int]; var ratio: Double; var resized: [Int] }
        struct Source: Decodable {
            var kind: String
            var upstreamFile: String?
            var crop: [Int]?
            var sha256: String?
            var letterbox: Letterbox?
        }
        var name: String
        var source: Source
    }

    struct Reference: Decodable {
        var anchor: Int
        var classIndex: Int
        var score: Double
        var box: [Double]
        var cocoId: Int
        var label: String
        var boxSourceNormalized: [Double]?
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

    static func iou(_ a: CGRect, _ b: CGRect) -> Double {
        let inter = a.intersection(b)
        let i = inter.isNull ? 0 : Double(inter.width * inter.height)
        let union = Double(a.width * a.height + b.width * b.height) - i
        return union > 0 ? i / union : 0
    }

    struct Parity: CustomStringConvertible {
        var reference = 0, detected = 0, matched = 0
        var sameClassSet = true
        var minIoU = 1.0, maxScoreDiff = 0.0
        var description: String {
            "\(detected) detected / \(reference) reference, \(matched) matched, same class set \(sameClassSet), "
                + "min IoU \(String(format: "%.5f", minIoU)), max |Δscore| \(String(format: "%.6f", maxScoreDiff))"
        }
    }

    /// `yolox_graphs.match_detections`: each reference detection, in descending score, paired with the unpaired
    /// same-class detection of highest IoU.
    static func match(_ got: [Detection], _ want: [(label: Int, score: Double, box: CGRect)]) -> Parity {
        var parity = Parity(reference: want.count, detected: got.count)
        parity.sameClassSet = got.map(\.label.id).sorted() == want.map(\.label).sorted()
        var used = Set<Int>()
        for w in want.sorted(by: { $0.score > $1.score }) {
            var best: (Int, Double)?
            for (j, g) in got.enumerated() where !used.contains(j) && g.label.id == w.label {
                let v = iou(g.box, w.box)
                if v > (best?.1 ?? -1) { best = (j, v) }
            }
            guard let (j, v) = best else { continue }
            used.insert(j)
            parity.matched += 1
            parity.minIoU = min(parity.minIoU, v)
            parity.maxScoreDiff = max(parity.maxScoreDiff, abs(Double(got[j].score) - w.score))
        }
        return parity
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

    static func image(_ url: URL, crop: [Int]?) throws -> ImageInput {
        guard let crop else { return ImageInput(contentsOf: url) }
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let full = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        return ImageInput(try #require(full.cropping(to: CGRect(x: crop[0], y: crop[1], width: crop[2] - crop[0], height: crop[3] - crop[1]))))
    }

    @Test func yoloxSMatchesUpstream() async throws {
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: Self.referenceURL.appending(path: "references.json")))
        let model = try await Registry.vision.load(
            try Self.localBundle(Self.bundleURL), as: DenseDetectorDriver.self,
            options: LoadOptions(device: Self.mac, cachePolicy: .default, cache: .default))
        let detector = try model.detector()
        #expect(detector.defaults == .init(scoreThreshold: 0.3, maxDetections: 50) && detector.labels == .coco91)
        guard case .denseNms(let postprocess)? = model.manifest.postprocess else { Issue.record("not denseNms"); return }
        let shape = try #require(index.shapes["input"]), preds = try #require(index.shapes["preds"])
        let (height, width) = (shape[2], shape[3])
        let upstream = Self.environment["AETHER_YOLOX_UPSTREAM"].map { URL(filePath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(
                path: ".cache/aether-kit/yolox/upstream/yolox-\(index.source.upstreamCode.commit.prefix(12))")

        for entry in index.cases {
            let directory = Self.referenceURL.appending(path: entry.name)
            let c = try JSONDecoder().decode(Case.self, from: Data(contentsOf: directory.appending(path: "case.json")))
            let reference = try JSONDecoder().decode([Reference].self, from: Data(contentsOf: directory.appending(path: "detections.json")))
            let input = try Self.floats(directory.appending(path: "input.f32"))
            let referencePreds = try Self.floats(directory.appending(path: "preds.f32"))
            #expect(input.count == shape.reduce(1, *) && referencePreds.count == preds.reduce(1, *))

            // (a) The host NMS on the reference's own preds: exactly its detections.
            let kept = DenseNMS(postprocess, columns: preds[2]).select(referencePreds, scoreThreshold: 0.3, maxDetections: 50)
            #expect(kept.map(\.anchor) == reference.map(\.anchor), "\(entry.name): NMS anchors")
            #expect(kept.map(\.column) == reference.map(\.classIndex), "\(entry.name): NMS classes")
            let nmsScore = zip(kept, reference).map { abs($0.score - $1.score) }.max() ?? 0
            let nmsBox = zip(kept, reference).flatMap { k, r in (0..<4).map { abs(k.box[$0] - r.box[$0]) } }.max() ?? 0
            #expect(nmsScore <= 1e-12 && nmsBox == 0, "\(entry.name): NMS scores and boxes")
            #expect(reference.allSatisfy { detector.labels.label(column: $0.classIndex, classIndex: .contiguous)
                == .init(id: $0.cocoId, name: $0.label) }, "\(entry.name): coco-91 labels")

            // (b) Detection parity: the reference tensor straight in, with its letterbox geometry.
            let source = c.source.letterbox?.source ?? [width, height], resized = c.source.letterbox?.resized ?? [width, height]
            let geometry = ImageGeometry(sourceWidth: source[0], sourceHeight: source[1], width: width, height: height,
                                         content: CGRect(x: 0, y: 0, width: resized[0], height: resized[1]))
            let boxes = DetectionBoxes(format: .cxcywh, space: .letterboxPixels, geometry: geometry)
            let want = reference.map { (label: $0.cocoId, score: $0.score, box: boxes.source($0.box[0], $0.box[1], $0.box[2], $0.box[3])) }
            let prepared = PreparedImage(shape: shape, values: .float32(input), geometry: geometry)
            let started = ContinuousClock.now
            let detections = try await detector.detect(prepared)
            let elapsed = ContinuousClock.now - started
            let parity = Self.match(detections, want)
            var mapping = 0.0
            for r in reference {
                guard let n = r.boxSourceNormalized else { continue }
                let b = boxes.source(r.box[0], r.box[1], r.box[2], r.box[3])
                let clipped = n.map { min(max($0, 0), 1) }
                mapping = max(mapping, abs(b.minX - clipped[0]), abs(b.minY - clipped[1]), abs(b.maxX - clipped[2]), abs(b.maxY - clipped[3]))
            }
            print("gate: yolox \(entry.name) graph: \(parity), \(elapsed); box mapping vs upstream's (÷ r) max |Δ| \(mapping)")
            #expect(parity.sameClassSet, "\(entry.name): class set")
            #expect(parity.matched == reference.count && detections.count == reference.count, "\(entry.name): counts")
            #expect(parity.minIoU >= Self.iouBar, "\(entry.name): IoU")
            #expect(parity.maxScoreDiff <= Self.scoreBar, "\(entry.name): scores")
            #expect(mapping < 2e-3, "\(entry.name): box mapping")
            #expect(try await detector.detect(prepared) == detections, "\(entry.name): repeat bit-identical")

            // (c) Pillow's decode through the host's preprocessing: the reference tensor, bit for bit.
            guard c.source.kind == "image", let file = c.source.upstreamFile, let sha = c.source.sha256 else { continue }
            let preprocessor = try ImagePreprocessor(manifest: model.manifest)
            let pillowRGB = try ImageInput(contentsOf: directory.appending(path: "decoded.png")).decode(colorSpace: .deviceRGB)
            let fromPillow = try preprocessor.prepare(pillowRGB)
            let resizeDiff = Self.differences(fromPillow.floats(), input)
            print("gate: yolox \(entry.name) Pillow's decode through preprocess.image: geometry \(fromPillow.geometry.content), "
                  + "\(resizeDiff.count) values differ from input.f32 (max |Δ| \(resizeDiff.max))")
            #expect(fromPillow.geometry == geometry, "\(entry.name): letterbox geometry")
            #expect(resizeDiff.count == 0, "\(entry.name): \(resizeDiff.count) values differ from the reference tensor")
            #expect(try await detector.detect(fromPillow) == detections, "\(entry.name): detections from Pillow's pixels")

            // (d) End to end from the upstream image, ImageIO's decode: gated at the same bars.
            let url = upstream.appending(path: file)
            guard let data = try? Data(contentsOf: url) else {
                print("gate: yolox \(entry.name): no image at \(url.path) (Forge/yolox/fetch_upstream.py); skipped")
                continue
            }
            #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == sha, "\(file): SHA-256")
            let image = try Self.image(url, crop: c.source.crop)
            let decodeDiff = Self.differences(try image.decode(colorSpace: .deviceRGB).rgb, pillowRGB.rgb)
            let photoPrepared = try detector.prepare(image)
            let tensorDiff = Self.differences(photoPrepared.floats(), input)
            let endStarted = ContinuousClock.now
            let found = try await detector.detect(image)
            let endElapsed = ContinuousClock.now - endStarted
            let end = Self.match(found, want)
            print("gate: yolox \(entry.name) end to end \(source[0])x\(source[1]): \(end), \(endElapsed); ImageIO vs Pillow "
                  + "decode \(decodeDiff.count) bytes differ (max \(Int(decodeDiff.max))), tensor \(tensorDiff.count) values "
                  + "differ (max \(tensorDiff.max)); \(found.map { "\($0.label.name) \(String(format: "%.3f", $0.score))" })")
            #expect(photoPrepared.geometry == geometry, "\(entry.name): end-to-end letterbox geometry")
            if decodeDiff.count == 0 {
                #expect(tensorDiff.count == 0, "\(entry.name) end to end: the same decode must give the reference tensor")
                #expect(end.sameClassSet, "\(entry.name) end to end: class set")
                #expect(end.matched == reference.count && found.count == reference.count, "\(entry.name) end to end: counts")
                #expect(end.minIoU >= Self.iouBar, "\(entry.name) end to end: IoU")
                #expect(end.maxScoreDiff <= Self.scoreBar, "\(entry.name) end to end: scores")
            } else {
                print("gate: yolox \(entry.name) end to end not gated: ImageIO's decode is not Pillow's (the resize is gated above)")
            }
            #expect(found.allSatisfy { $0.box.minX >= 0 && $0.box.maxX <= 1 && $0.box.minY >= 0 && $0.box.maxY <= 1 })
            if entry.name == "dog" {
                #expect(Set(found.map(\.label.name)).isSuperset(of: ["dog", "bicycle"]), "dog.jpg: upstream's README sanity")
            }
        }
        for (key, footprint) in detector.options.ledger.snapshot.sorted(by: { $0.key.description < $1.key.description }) {
            print("gate: yolox \(key): peak \(footprint.peakBytes >> 20) MiB, held \(footprint.heldBytes >> 20) MiB")
        }
        await detector.unload()
        #expect(await detector.residentRoles.isEmpty)
    }
}
