import CoreGraphics
import Foundation
import Synchronization
import Testing
@_spi(Testing) import AetherHost
@testable import AetherVision

/// `detector-dense@1` over fake graphs (DESIGN.md §12): the contract, the `denseNms` postprocess (scoring, NMS, the
/// request's thresholds), the box mapping through the letterbox, the COCO-91 table and cancellation. A constant probe
/// and no GPU admission, so nothing depends on the machine.
@Suite("Object detector (dense)")
struct ObjectDetectorTests {
    typealias Postprocess = Manifest.Postprocess

    /// A YOLOX-shaped manifest with an 8x8 letterbox (top-left, pad 114); `edit` changes the JSON object before decoding.
    static func manifest(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> Manifest {
        var json: [String: Any] = [
            "schema": 1, "id": "detector-test", "task": "object-detection", "driver": "detector-dense@1",
            "variant": ["platform": "macos"], "assets": ["main": ["subtree": ""]],
            "graphs": ["detector": [
                "file": "main:detector.aimodel", "function": "main", "inputs": ["image": "image"],
                "outputs": ["preds": "preds"], "compute": "gpu",
            ] as [String: Any]],
            "preprocess": ["image": [
                "resize": ["mode": "letterbox", "size": [8, 8], "anchor": "topLeft", "padValue": 114, "filter": "vImage"]
                    as [String: Any],
                "channelOrder": "BGR", "layout": "NCHW", "hostDType": "float32",
            ] as [String: Any]],
            "postprocess": [
                "kind": "denseNms", "boxFormat": "cxcywh", "boxSpace": "letterboxPixels", "activation": "applied",
                "nms": ["type": "perClass", "iou": 0.45] as [String: Any],
                "defaults": ["scoreThreshold": 0.3, "maxDetections": 50] as [String: Any], "classIndex": "contiguous",
            ] as [String: Any],
            "labels": ["set": "coco-91"],
        ]
        edit(&json)
        return try Manifest.decode(JSONSerialization.data(withJSONObject: json))
    }

    /// One anchor's row: box, objectness, and `classes` class scores with `scores` at their columns.
    static func row(_ box: [Float], obj: Float, _ scores: [Int: Float], classes: Int = 80) -> [Float] {
        var row = box + [obj] + [Float](repeating: 0, count: classes)
        for (k, v) in scores { row[5 + k] = v }
        return row
    }

    static let postprocess = Postprocess.DenseNMS(
        boxFormat: .cxcywh, boxSpace: .letterboxPixels, activation: .applied, nms: .init(type: .perClass, iou: 0.45),
        defaults: .init(scoreThreshold: 0.3, maxDetections: 50), classIndex: .contiguous)

    static func select(
        _ rows: [[Float]], _ postprocess: Postprocess.DenseNMS = postprocess, threshold: Double = 0.3, max: Int = 50
    ) -> [DenseNMS.Kept] {
        DenseNMS(postprocess, columns: rows[0].count).select(Array(rows.joined()), scoreThreshold: threshold, maxDetections: max)
    }

    final class Log: Sendable {
        private let state = Mutex<[String]>([])
        var entries: [String] { state.withLock { $0 } }
        func append(_ entry: String) { state.withLock { $0.append(entry) } }
    }

    /// Holds a run until released.
    actor Gate {
        private var entered = false, released = false
        private var waiting: CheckedContinuation<Void, Never>?
        private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
        func enter() async {
            entered = true
            enteredWaiters.forEach { $0.resume() }
            enteredWaiters = []
            if !released { await withCheckedContinuation { waiting = $0 } }
        }
        func waitUntilEntered() async { if !entered { await withCheckedContinuation { enteredWaiters.append($0) } } }
        func release() {
            released = true
            waiting?.resume()
            waiting = nil
        }
    }

    /// YOLOX's I/O on an 8x8 input: writes `rows`, and checks it was fed BGR on 0...255 with the letterbox's padding.
    final class FakeDetector: GraphBackend {
        let signature: GraphSignature
        let rows: [[Float]]
        let log: Log
        let gate: Gate?
        init(rows: [[Float]], log: Log, gate: Gate?, columns: Int? = nil) {
            signature = GraphSignature(
                function: "main", inputs: [TensorSpec(name: "image", scalarType: .float32, shape: [1, 3, 8, 8])], states: [],
                outputs: [TensorSpec(name: "preds", scalarType: .float32, shape: [1, rows.count, columns ?? rows[0].count])])
            self.rows = rows
            self.log = log
            self.gate = gate
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            let image = try bindings.input("image")!.read(Float.self)
            log.append("run pad \(image[63]) b \(image[0]) r \(image[128])")
            await gate?.enter()
            try bindings.output("preds")!.write(Array(rows.joined()))
        }
    }

    static let device = Device(platform: .macOS, architecture: "test")

    static func detector(
        _ manifest: Manifest, rows: [[Float]], log: Log = Log(), gate: Gate? = nil, columns: Int? = nil
    ) throws -> ObjectDetector {
        try ObjectDetector(
            manifest: manifest, device: device,
            options: PipelineOptions(memory: .bytes(1 << 40), gpu: .unadmitted, probe: GPUMemoryProbe { 0 }),
            load: { role throws(AetherError) in
                log.append("load \(role)")
                return FakeDetector(rows: rows, log: log, gate: gate, columns: columns)
            },
            fileBytes: { _ in 100 })
    }

    /// A `width x height` opaque sRGB image of one colour.
    static func image(width: Int, height: Int, rgb: (UInt8, UInt8, UInt8)) throws -> ImageInput {
        let bytes = [UInt8]((0..<(width * height)).flatMap { _ in [rgb.0, rgb.1, rgb.2, 255] })
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = try #require(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4 * width,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        return ImageInput(image)
    }

    static func isInvalid(_ field: String) -> (any Error) -> Bool {
        { error in
            switch error as? AetherError {
            case .manifestInvalid(let f, _)?: f == field
            default: false
            }
        }
    }

    static func close(_ a: CGRect, _ b: CGRect, _ tolerance: Double = 1e-9) -> Bool {
        abs(a.minX - b.minX) < tolerance && abs(a.minY - b.minY) < tolerance && abs(a.width - b.width) < tolerance
            && abs(a.height - b.height) < tolerance
    }

    // MARK: - Contract

    @Test func theDriverIsRegisteredWithItsContract() throws {
        #expect(Registry.vision.supports(.detectorDense))
        #expect(try Registry.vision.driver(for: .detectorDense) == DenseDetectorDriver.self)
        #expect(DriverCatalog.contracts[.detectorDense] == .detectorDense)
        try DriverContract.detectorDense.validate(Self.manifest())
        let detector = try Self.detector(Self.manifest(), rows: [Self.row([0, 0, 0, 0], obj: 0, [:])])
        #expect(detector.defaults == .init(scoreThreshold: 0.3, maxDetections: 50))
        #expect(detector.labels == .coco91)
    }

    @Test func theManifestIsCheckedBeforeAnythingLoads() throws {
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("graphs.detector.outputs.preds", { json in
                json["graphs"] = ["detector": ["file": "main:d.aimodel", "function": "main", "inputs": ["image": "image"],
                                               "outputs": [String: String](), "compute": "gpu"] as [String: Any]]
            }),
            ("postprocess", { json in json["postprocess"] = ["kind": "depth", "depth": ["polarity": "largerIsFarther"]] }),
            ("postprocess", { json in json["postprocess"] = nil }),
            ("labels", { json in json["labels"] = nil }),
            ("labels.file", { json in json["labels"] = ["file": "main:labels.json"] }),
            ("preprocess.image", { json in json["preprocess"] = nil }),
            ("preprocess.image.layout", { json in
                var image = (json["preprocess"] as! [String: Any])["image"] as! [String: Any]
                image["layout"] = "NVCHW"
                image["views"] = 2
                json["preprocess"] = ["image": image]
            }),
        ]
        for (field, edit) in cases {
            let manifest = try Self.manifest(edit)
            #expect("\(field)", performing: { _ = try Self.detector(manifest, rows: [Self.row([0, 0, 0, 0], obj: 0, [:])]) },
                    throws: Self.isInvalid(field))
        }
    }

    @Test func aGraphWhoseRowsDoNotMatchTheLabelSetIsRefused() async throws {
        let log = Log()
        let detector = try Self.detector(Self.manifest(), rows: [Self.row([4, 4, 2, 2], obj: 1, [0: 1], classes: 79)], log: log)
        await #expect {
            _ = try await detector.detect(Self.image(width: 8, height: 8, rgb: (0, 0, 0)))
        } throws: { error in
            if case .manifestMismatch(let field, _, _)? = error as? AetherError { field == "graphs.detector.outputs" } else { false }
        }
        #expect(!log.entries.contains { $0.hasPrefix("run") })
    }

    // MARK: - denseNms

    @Test func scoresAreObjectnessTimesTheBestClassFirstOnTies() {
        let kept = Self.select([
            Self.row([10, 10, 4, 4], obj: 0.8, [3: 0.5, 7: 0.9]),       // 0.72, class 7
            Self.row([50, 50, 4, 4], obj: 0.9, [2: 0.6, 5: 0.6]),       // 0.54, class 2 (first of the tie)
            Self.row([90, 90, 4, 4], obj: 0.5, [1: 0.59]),              // 0.295: under 0.3
            Self.row([130, 130, 4, 4], obj: 0.5, [1: 0.6]),             // exactly 0.3: kept (≥)
        ])
        #expect(kept.map(\.anchor) == [0, 1, 3])
        #expect(kept.map(\.column) == [7, 2, 1])
        #expect(abs(kept[0].score - Double(Float(0.8)) * Double(Float(0.9))) < 1e-15)
        #expect(kept[0].box == SIMD4(10, 10, 4, 4))
    }

    @Test func nmsSuppressesOnlyAboveTheIoUAndWithinAClass() {
        // Two 10x10 boxes offset by d along x: IoU = (10 - d) / (10 + d). d = 3.793103... gives 0.45 exactly;
        // 3.7 (0.4599) suppresses, 3.9 (0.4388) does not.
        let a = Self.row([20, 20, 10, 10], obj: 1, [0: 0.9])
        #expect(Self.select([a, Self.row([23.7, 20, 10, 10], obj: 1, [0: 0.8])]).map(\.anchor) == [0])
        #expect(Self.select([a, Self.row([23.9, 20, 10, 10], obj: 1, [0: 0.8])]).map(\.anchor) == [0, 1])
        // Another class is never suppressed per class, and is across classes.
        let other = Self.row([20, 20, 10, 10], obj: 1, [4: 0.8])
        #expect(Self.select([a, other]).map(\.anchor) == [0, 1])
        var agnostic = Self.postprocess
        agnostic.nms = .init(type: .classAgnostic, iou: 0.45)
        #expect(Self.select([a, other], agnostic).map(\.anchor) == [0])
        // IoU exactly at the bound keeps the box (upstream suppresses IoU > iou).
        #expect(DenseNMS.iou(SIMD4(0, 0, 2, 1), SIMD4(1, 0, 3, 1)) == 1.0 / 3)
        var third = Self.postprocess
        third.nms = .init(type: .perClass, iou: 1.0 / 3)
        let left = Self.row([1, 0.5, 2, 1], obj: 1, [0: 0.9]), right = Self.row([2, 0.5, 2, 1], obj: 1, [0: 0.8])
        #expect(Self.select([left, right], third).map(\.anchor) == [0, 1])
    }

    @Test func nmsIsGreedyInScoreOrderWithTiesInAnchorOrder() {
        // B overlaps A and C; C does not overlap A. A (best) suppresses B, so C survives.
        let kept = Self.select([
            Self.row([18, 0, 10, 10], obj: 1, [0: 0.7]),    // C: IoU 0.25 with A
            Self.row([15, 0, 10, 10], obj: 1, [0: 0.8]),    // B: IoU 0.54 with A and with C
            Self.row([12, 0, 10, 10], obj: 1, [0: 0.9]),    // A
        ])
        #expect(kept.map(\.anchor) == [2, 0])
        // Equal scores: the earlier anchor is kept and listed first.
        let tie = Self.select([
            Self.row([100, 0, 10, 10], obj: 1, [3: 0.5]),
            Self.row([0, 0, 10, 10], obj: 1, [1: 0.5]),
            Self.row([1, 0, 10, 10], obj: 1, [1: 0.5]),
        ])
        #expect(tie.map(\.anchor) == [0, 1])
    }

    @Test func maxDetectionsKeepsTheHighestAfterNMS() {
        let rows = (0..<10).map { i in Self.row([Float(20 * i), 0, 10, 10], obj: 1, [i % 3: 0.35 + Float(i) / 20]) }
        let kept = Self.select(rows, max: 4)
        #expect(kept.map(\.anchor) == [9, 8, 7, 6])
        #expect(Self.select(rows, threshold: 0.575).map(\.anchor) == [9, 8, 7, 6, 5])
        #expect(Self.select(rows, threshold: 1).isEmpty)
    }

    @Test func sigmoidActivationAndXYXYBoxes() {
        var logits = Self.postprocess
        logits.activation = .sigmoid
        logits.boxFormat = .xyxy
        // sigmoid(2) · sigmoid(1) = 0.6439; sigmoid(0) · sigmoid(0) = 0.25 is dropped.
        let kept = Self.select([Self.row([0, 0, 10, 10], obj: 2, [5: 1]), Self.row([50, 50, 60, 60], obj: 0, [:])], logits)
        #expect(kept.count == 1 && kept[0].column == 5)
        #expect(abs(kept[0].score - 1 / (1 + exp(-2.0)) / (1 + exp(-1.0))) < 1e-12)
        // xyxy overlap: [0,0,10,10] and [2,0,12,10] have IoU 8/12.
        let pair = Self.select([Self.row([0, 0, 10, 10], obj: 5, [0: 5]), Self.row([2, 0, 12, 10], obj: 4, [0: 4])], logits)
        #expect(pair.map(\.anchor) == [0])
    }

    @Test func nonFiniteScoresAreNeverKept() {
        let kept = Self.select([
            Self.row([0, 0, 10, 10], obj: .nan, [0: 1]),
            Self.row([0, 0, 10, 10], obj: .infinity, [0: 1]),
            Self.row([50, 0, 10, 10], obj: 1, [0: 0.9]),
        ])
        #expect(kept.map(\.anchor) == [2])
    }

    // MARK: - Boxes and labels

    @Test func boxesMapThroughTheLetterboxToTheNormalizedSource() {
        // A 1000x500 source letterboxed top-left into 640x640: content (0, 0, 640, 320).
        let geometry = ImageGeometry(sourceWidth: 1000, sourceHeight: 500, width: 640, height: 640,
                                     content: CGRect(x: 0, y: 0, width: 640, height: 320))
        let boxes = DetectionBoxes(format: .cxcywh, space: .letterboxPixels, geometry: geometry)
        #expect(Self.close(boxes.source(320, 160, 640, 320), CGRect(x: 0, y: 0, width: 1, height: 1)))
        #expect(Self.close(boxes.source(160, 80, 64, 32), CGRect(x: 0.2, y: 0.2, width: 0.1, height: 0.1)))
        // Into the padding: clipped to the image; wholly in it, an empty box on the edge.
        #expect(Self.close(boxes.source(600, 300, 100, 100), CGRect(x: 550.0 / 640, y: 250.0 / 320, width: 90.0 / 640, height: 70.0 / 320)))
        let outside = boxes.source(320, 500, 40, 40)
        #expect(outside.height == 0 && outside.width == 0 && outside.minY == 1)
        // xyxy, and normalized to the graph's input.
        let xyxy = DetectionBoxes(format: .xyxy, space: .letterboxPixels, geometry: geometry)
        #expect(Self.close(xyxy.source(64, 32, 128, 64), CGRect(x: 0.1, y: 0.1, width: 0.1, height: 0.1)))
        let normalized = DetectionBoxes(format: .xyxy, space: .normalized, geometry: geometry)
        #expect(Self.close(normalized.source(0.1, 0.05, 0.2, 0.1), CGRect(x: 0.1, y: 0.1, width: 0.1, height: 0.1)))
        // A centred letterbox: the content's origin is subtracted.
        let centred = DetectionBoxes(format: .cxcywh, space: .letterboxPixels, geometry: ImageGeometry(
            sourceWidth: 1000, sourceHeight: 500, width: 640, height: 640, content: CGRect(x: 0, y: 160, width: 640, height: 320)))
        #expect(Self.close(centred.source(320, 320, 640, 320), CGRect(x: 0, y: 0, width: 1, height: 1)))
    }

    @Test func coco91HasTheEightyUsedIdsInOrder() {
        let labels = DetectionLabels.coco91.labels
        let unused: Set = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
        #expect(labels.map(\.id) == (1...90).filter { !unused.contains($0) })
        #expect(Set(labels.map(\.name)).count == 80)
        let coco = DetectionLabels.coco91
        #expect(coco.label(column: 0, classIndex: .contiguous) == .init(id: 1, name: "person"))
        #expect(coco.label(column: 16, classIndex: .contiguous) == .init(id: 18, name: "dog"))
        #expect(coco.label(column: 79, classIndex: .contiguous) == .init(id: 90, name: "toothbrush"))
        #expect(coco.label(column: 80, classIndex: .contiguous) == nil)
        #expect(coco.label(column: 18, classIndex: .cocoId) == .init(id: 18, name: "dog"))
        #expect(coco.label(column: 12, classIndex: .cocoId) == nil && coco.label(column: 0, classIndex: .cocoId) == nil)
    }

    // MARK: - End to end

    @Test func aDetectionComesBackLabelledInTheSourceImage() async throws {
        let log = Log()
        // An 8x4 source: the letterbox's content is the top 8x4 of the 8x8 input.
        let rows = [
            Self.row([4, 2, 8, 4], obj: 0.9, [16: 0.95]),           // the whole image: dog, 0.855
            Self.row([2, 1, 2, 2], obj: 0.8, [1: 0.5]),             // bicycle 0.4, top-left quarter
            Self.row([4, 2, 8, 4], obj: 0.5, [16: 0.7]),            // a duplicate dog: suppressed
        ]
        let detector = try Self.detector(Self.manifest(), rows: rows, log: log)
        let input = try Self.image(width: 8, height: 4, rgb: (200, 100, 10))
        let detections = try await detector.detect(input)
        #expect(detections.count == 2)
        #expect(detections[0].label == .init(id: 18, name: "dog") && abs(detections[0].score - 0.855) < 1e-6)
        #expect(Self.close(detections[0].box, CGRect(x: 0, y: 0, width: 1, height: 1)))
        #expect(detections[1].label.name == "bicycle" && Self.close(detections[1].box, CGRect(x: 0.125, y: 0, width: 0.25, height: 0.5)))
        #expect(detections.allSatisfy { $0.mask == nil })
        // BGR on 0...255, padded with 114 below the content.
        #expect(log.entries == ["load detector", "run pad 114.0 b 10.0 r 200.0"])

        // The request's thresholds override the manifest's.
        #expect(try await detector.detect(input, DetectionRequest(scoreThreshold: 0.5)).map(\.label.name) == ["dog"])
        #expect(try await detector.detect(input, DetectionRequest(maxDetections: 1)).count == 1)
        #expect(try await detector.detect(input, DetectionRequest(scoreThreshold: 0.9)).isEmpty)
        // The prepared (CPU) and graph (GPU) halves separately give the same detections.
        #expect(try await detector.detect(detector.prepare(input)) == detections)
    }

    @Test func anInvalidRequestFailsBeforeTheGraph() async throws {
        let log = Log()
        let detector = try Self.detector(Self.manifest(), rows: [Self.row([4, 2, 8, 4], obj: 1, [0: 1])], log: log)
        let input = try Self.image(width: 8, height: 4, rgb: (0, 0, 0))
        for request in [DetectionRequest(scoreThreshold: 1.5), DetectionRequest(scoreThreshold: .nan), DetectionRequest(maxDetections: 0)] {
            await #expect(throws: AetherError.self) { _ = try await detector.detect(input, request) }
        }
        #expect(log.entries.isEmpty)
    }

    @Test func cancellationStopsAfterTheGraphCallAndTheDetectorRecovers() async throws {
        let log = Log(), gate = Gate()
        let detector = try Self.detector(Self.manifest(), rows: [Self.row([4, 2, 8, 4], obj: 1, [0: 1])], log: log, gate: gate)
        let input = try Self.image(width: 8, height: 4, rgb: (0, 0, 0))
        let task = Task { try await detector.detect(input) }
        await gate.waitUntilEntered()
        let queued = Task { try await detector.detect(input) }
        task.cancel()
        await gate.release()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(try await queued.value.map(\.label.name) == ["person"])
        #expect(log.entries.filter { $0.hasPrefix("load") }.count == 2)
        await detector.unload()
        #expect(await detector.residentRoles.isEmpty)
    }
}
