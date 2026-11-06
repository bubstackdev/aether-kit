import CoreGraphics
import Foundation
import Synchronization
import Testing
@_spi(Testing) import AetherHost
@testable import AetherVision

/// `detector-detr@1` over fake graphs (DESIGN.md §12): the contract, the `detrSigmoid` postprocess (selection, labels,
/// box mapping, masks), the request overrides and cancellation. A constant probe and no GPU admission, so nothing
/// depends on the machine.
@Suite("Object detector, DETR")
struct ObjectDetectorDetrTests {
    static let columns = 91

    /// An RF-DETR-shaped manifest with a 4x4 graph input; `masks` adds the seg variant's output and block; `edit` changes
    /// the JSON object before decoding.
    static func manifest(masks: Bool = false, _ edit: (inout [String: Any]) -> Void = { _ in }) throws -> Manifest {
        var outputs: [String: Any] = ["boxes": "dets", "classLogits": "labels"]
        var postprocess: [String: Any] = [
            "kind": "detrSigmoid", "boxFormat": "cxcywh", "boxSpace": "normalized", "classIndex": "cocoId",
            "selection": "topKPairs", "defaults": ["scoreThreshold": 0.5, "maxDetections": 300] as [String: Any],
        ]
        if masks {
            outputs["masks"] = "masks"
            postprocess["masks"] = ["logitThreshold": 0.0]
        }
        var json: [String: Any] = [
            "schema": 1, "id": "detr-test", "task": "object-detection", "driver": "detector-detr@1",
            "variant": ["platform": "macos"], "assets": ["main": ["subtree": ""]],
            "graphs": ["detector": [
                "file": "main:detector.aimodel", "function": "main", "inputs": ["image": "image"], "outputs": outputs,
                "compute": "gpu",
            ] as [String: Any]],
            "preprocess": ["image": [
                "resize": ["mode": "stretch", "size": [4, 4], "filter": "vImage"] as [String: Any], "colorSpace": "sRGB",
                "scale": 1.0 / 255, "layout": "NCHW", "hostDType": "float32", "orientation": "exif",
            ] as [String: Any]],
            "postprocess": postprocess,
            "labels": ["set": "coco-91"],
        ]
        edit(&json)
        return try Manifest.decode(JSONSerialization.data(withJSONObject: json))
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

    /// Raw outputs a fake graph returns: `queries` boxes (cxcywh, normalized), logits (`queries · columns`, default
    /// -10) and optional 2x2 mask planes.
    struct Outputs: Sendable {
        var queries = 3
        var columns = ObjectDetectorDetrTests.columns
        var boxes: [Float]
        var logits: [Float]
        var masks: [Float]?

        init(queries: Int = 3, columns: Int = ObjectDetectorDetrTests.columns, boxes: [Float]? = nil, masks: [Float]? = nil,
             logits: [(query: Int, column: Int, logit: Float)] = []) {
            self.queries = queries
            self.columns = columns
            self.boxes = boxes ?? (0..<queries).flatMap { _ in [Float(0.5), 0.5, 0.5, 0.5] }
            var l = [Float](repeating: -10, count: queries * columns)
            for (q, c, v) in logits { l[q * columns + c] = v }
            self.logits = l
            self.masks = masks
        }
    }

    final class FakeDetector: GraphBackend {
        let signature: GraphSignature
        let log: Log, gate: Gate?, outputs: Outputs
        init(log: Log, gate: Gate?, outputs: Outputs) {
            var specs = [TensorSpec(name: "dets", scalarType: .float32, shape: [1, outputs.queries, 4]),
                         TensorSpec(name: "labels", scalarType: .float32, shape: [1, outputs.queries, outputs.columns])]
            if outputs.masks != nil { specs.append(TensorSpec(name: "masks", scalarType: .float32, shape: [1, outputs.queries, 2, 2])) }
            signature = GraphSignature(
                function: "main", inputs: [TensorSpec(name: "image", scalarType: .float32, shape: [1, 3, 4, 4])], states: [],
                outputs: specs)
            self.log = log
            self.gate = gate
            self.outputs = outputs
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            log.append("run")
            await gate?.enter()
            try bindings.output("dets")!.write(outputs.boxes)
            try bindings.output("labels")!.write(outputs.logits)
            if let masks = outputs.masks { try bindings.output("masks")!.write(masks) }
        }
    }

    static let device = Device(platform: .macOS, architecture: "test")

    static func detector(_ manifest: Manifest, _ outputs: Outputs = Outputs(), log: Log = Log(), gate: Gate? = nil) throws -> ObjectDetector {
        try ObjectDetector(
            manifest: manifest, device: device,
            options: PipelineOptions(memory: .bytes(1 << 40), gpu: .unadmitted, probe: GPUMemoryProbe { 0 }),
            load: { role throws(AetherError) in
                log.append("load \(role)")
                return FakeDetector(log: log, gate: gate, outputs: outputs)
            },
            fileBytes: { _ in 100 })
    }

    /// A `width x height` opaque sRGB image of one colour.
    static func image(width: Int, height: Int) throws -> ImageInput {
        let bytes = [UInt8](repeating: 128, count: 4 * width * height)
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

    static func close(_ a: CGRect, _ b: CGRect) -> Bool {
        abs(a.minX - b.minX) < 1e-6 && abs(a.minY - b.minY) < 1e-6 && abs(a.width - b.width) < 1e-6 && abs(a.height - b.height) < 1e-6
    }

    // MARK: - Contract

    @Test func theDriverIsRegisteredWithItsContract() throws {
        #expect(Registry.vision.supports(.detectorDetr))
        #expect(try Registry.vision.driver(for: .detectorDetr) == DetrDetectorDriver.self)
        #expect(DriverCatalog.contracts[.detectorDetr] == .detectorDetr)
        try DriverContract.detectorDetr.validate(Self.manifest())
        try DriverContract.detectorDetr.validate(Self.manifest(masks: true))
    }

    @Test func theManifestIsCheckedBeforeAnythingLoads() throws {
        func graph(_ outputs: [String: Any]) -> [String: Any] {
            ["detector": ["file": "main:d.aimodel", "function": "main", "inputs": ["image": "image"], "outputs": outputs,
                          "compute": "gpu"] as [String: Any]]
        }
        let cases: [(String, Bool, (inout [String: Any]) -> Void)] = [
            ("graphs.detector.outputs.classLogits", false, { $0["graphs"] = graph(["boxes": "dets"]) }),
            ("postprocess", false, { $0["postprocess"] = ["kind": "depth", "depth": ["polarity": "largerIsFarther"]] }),
            ("postprocess", false, { $0["postprocess"] = nil }),
            // Masks bound without the block, and the block without masks.
            ("postprocess.masks", false, { $0["graphs"] = graph(["boxes": "dets", "classLogits": "labels", "masks": "masks"]) }),
            ("postprocess.masks", true, { $0["graphs"] = graph(["boxes": "dets", "classLogits": "labels"]) }),
            ("labels", false, { $0["labels"] = nil }),
            ("labels.file", false, { $0["labels"] = ["file": "main:labels.json"] }),
            ("preprocess.image", false, { $0["preprocess"] = nil }),
            ("preprocess.image.layout", false, { json in
                var image = (json["preprocess"] as! [String: Any])["image"] as! [String: Any]
                image["layout"] = "NVCHW"
                image["views"] = 2
                json["preprocess"] = ["image": image]
            }),
        ]
        for (field, masks, edit) in cases {
            let manifest = try Self.manifest(masks: masks, edit)
            #expect("\(field)", performing: { _ = try Self.detector(manifest) }, throws: Self.isInvalid(field))
        }
        // Another driver's manifest with a detrSigmoid block is refused by the contract.
        let other = try Self.manifest { $0["driver"] = "detector-dense@1" }
        #expect(throws: AetherError.self) { _ = try Self.detector(other) }
    }

    @Test func aGraphWhoseOutputsAreNotTheHeadsShapesIsRefusedBeforeItRuns() async throws {
        // 92 class columns: more than coco-91 labels.
        let log = Log()
        let detector = try Self.detector(Self.manifest(), Outputs(columns: 92), log: log)
        await #expect {
            _ = try await detector.detect(Self.image(width: 4, height: 4))
        } throws: { error in
            if case .manifestMismatch(let field, _, _)? = error as? AetherError { field == "graphs.detector.outputs" } else { false }
        }
        let resident = await detector.residentRoles
        #expect(!log.entries.contains("run") && resident.isEmpty)
    }

    // MARK: - Selection

    @Test func selectionRanksEveryQueryClassPairAndKeepsThoseStrictlyAboveTheThreshold() {
        // Flat indices 0...5 over 2 queries x 3 classes.
        let logits: [Float] = [2, 0, 1, 1, 3, -1]
        func select(_ threshold: Float, _ limit: Int, labelled: [Bool] = [true, true, true]) -> [Int] {
            DetrHead.select(logits, columns: 3, labelled: labelled, threshold: threshold, limit: limit).map(\.index)
        }
        // Threshold 0.5: logit 0 scores exactly 0.5, not above it.
        #expect(select(0.5, 300) == [4, 0, 2, 3])    // 3 > 2 > (1 at 2, 1 at 3: ties by index)
        #expect(DetrHead.select(logits, columns: 3, labelled: [true, true, true], threshold: 0.5, limit: 1)[0].score == 1 / (1 + exp(-3)))
        // The best `limit`, ties broken by index.
        #expect(select(0.5, 3) == [4, 0, 2])
        #expect(select(0.95, 3) == [4])
        #expect(select(0, 0).isEmpty)
        // Unlabelled columns are left out before the cut.
        #expect(select(0.5, 2, labelled: [false, true, true]) == [4, 2])
    }

    @Test func detectionsCarryCOCOLabelsBoxesInSourceSpaceAndOneQueryMayYieldSeveralClasses() async throws {
        // Query 0: a cat (17) and a dog (18); query 1: a person (1), box beyond the image (clamped); query 2: below threshold.
        let outputs = Outputs(
            boxes: [0.5, 0.5, 0.5, 0.25, 0.9, 0.1, 0.4, 0.4, 0.5, 0.5, 1, 1],
            logits: [(0, 17, 4), (0, 18, 1), (1, 1, 2), (2, 3, -0.1)])
        let log = Log()
        let detector = try Self.detector(Self.manifest(), outputs, log: log)
        #expect(detector.manifest.postprocess?.kind == .detrSigmoid && detector.defaults.maxDetections == 300)
        // A 8x6 source stretched to 4x4: normalized boxes are already the source's.
        let detections = try await detector.detect(Self.image(width: 8, height: 6))
        #expect(detections.map(\.label) == [.init(id: 17, name: "cat"), .init(id: 1, name: "person"), .init(id: 18, name: "dog")])
        #expect(detections.map(\.score) == [4, 2, 1].map { 1 / (1 + exp(-$0)) })
        #expect(Self.close(detections[0].box, CGRect(x: 0.25, y: 0.375, width: 0.5, height: 0.25)))
        #expect(detections[2].box == detections[0].box && detections.allSatisfy { $0.mask == nil })
        #expect(Self.close(detections[1].box, CGRect(x: 0.7, y: 0, width: 0.3, height: 0.3)))
        #expect(log.entries == ["load detector", "run"])

        // The request overrides the manifest's defaults.
        let one = try await detector.detect(Self.image(width: 8, height: 6), DetectionRequest(maxDetections: 1))
        #expect(one.map(\.label.id) == [17])
        let strict = try await detector.detect(Self.image(width: 8, height: 6), DetectionRequest(scoreThreshold: 0.9))
        #expect(strict.map(\.label.id) == [17])
        let loose = try await detector.detect(Self.image(width: 8, height: 6), DetectionRequest(scoreThreshold: 0.4))
        #expect(loose.map(\.label.id) == [17, 1, 18, 3])
        // The prepared (CPU) and the graph (GPU) halves separately give the same detections.
        let prepared = try detector.prepare(Self.image(width: 8, height: 6))
        #expect(try await detector.detect(prepared) == detections)
    }

    @Test func anUnlabelledColumnIsNeverReported() async throws {
        // Column 0 (no COCO category) outscores the cat; it is left out before the cut, so the cat still returns.
        let outputs = Outputs(logits: [(0, 0, 5), (1, 17, 3)])
        let detector = try Self.detector(Self.manifest(), outputs)
        #expect(try await detector.detect(Self.image(width: 4, height: 4)).map(\.label.id) == [17])
        #expect(try await detector.detect(Self.image(width: 4, height: 4), DetectionRequest(maxDetections: 1)).map(\.label.id) == [17])
    }

    @Test func letterboxedBoxesMapToTheSource() async throws {
        // A 4x2 source letterboxed into rows 1...2 of the 4x4 input: the input's middle half is the whole source.
        let manifest = try Self.manifest { json in
            var image = (json["preprocess"] as! [String: Any])["image"] as! [String: Any]
            image["resize"] = ["mode": "letterbox", "size": [4, 4], "filter": "vImage"] as [String: Any]
            json["preprocess"] = ["image": image]
        }
        let outputs = Outputs(queries: 2, boxes: [0.5, 0.5, 0.5, 0.25, 0.5, 0.5, 1, 1], logits: [(0, 17, 4), (1, 18, 3)])
        let detections = try await Self.detector(manifest, outputs).detect(Self.image(width: 4, height: 2))
        #expect(Self.close(detections[0].box, CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)))
        // The padding is outside the source: clipped away.
        #expect(Self.close(detections[1].box, CGRect(x: 0, y: 0, width: 1, height: 1)))
    }

    // MARK: - Masks

    @Test func masksAreBilinearAtPixelCentresAndStrictlyAboveTheThreshold() {
        // A 2x1 plane onto 4x1 source pixels (stretched): centres at -0.25, 0.25, 0.75, 1.25, clamped: -1, 0, 2, 3.
        let geometry = ImageGeometry(sourceWidth: 4, sourceHeight: 1, width: 8, height: 2, content: CGRect(x: 0, y: 0, width: 8, height: 2))
        let resampler = MaskResampler(geometry, planeShape: [1, 1, 1, 2])
        let mask = resampler.mask([-1, 3], plane: 0, threshold: 0)
        #expect((0..<4).map { mask[x: $0, y: 0] } == [false, false, true, true])
        #expect(mask.bounds == .init(x: 2, y: 0, width: 2, height: 1) && mask.width == 4 && mask.area == 2)
        #expect(DetectionMask(width: 4, height: 1, foreground: [false, false, true, true]) == mask)
        #expect(mask.foreground == [false, false, true, true])
        #expect(mask.alpha() == [0, 0, 255, 255])
        // Lower the threshold: the interpolated 0 is now above it.
        #expect(resampler.mask([-1, 3], plane: 0, threshold: -0.5).area == 3)
        // Nothing above the threshold: empty.
        let empty = resampler.mask([-1, -2], plane: 0, threshold: 0)
        #expect(empty.isEmpty && empty.area == 0 && !empty[x: 0, y: 0] && empty.alpha() == [0, 0, 0, 0])
    }

    @Test func theMaskRegionShortcutMatchesResamplingEveryPixel() {
        // Random planes, a stretched and a letterboxed geometry: the mask equals thresholding the full resample.
        var generator = SystemRandomNumberGenerator()
        let geometries = [
            ImageGeometry(sourceWidth: 37, sourceHeight: 23, width: 16, height: 16, content: CGRect(x: 0, y: 0, width: 16, height: 16)),
            ImageGeometry(sourceWidth: 30, sourceHeight: 15, width: 16, height: 16, content: CGRect(x: 0, y: 4, width: 16, height: 8)),
        ]
        for geometry in geometries {
            let resampler = MaskResampler(geometry, planeShape: [1, 2, 4, 4])
            let plane = ImageGeometry(sourceWidth: geometry.sourceWidth, sourceHeight: geometry.sourceHeight, width: 4, height: 4,
                                      content: CGRect(x: geometry.content.minX / 4, y: geometry.content.minY / 4,
                                                      width: geometry.content.width / 4, height: geometry.content.height / 4))
            for _ in 0..<20 {
                let planes = (0..<32).map { _ in Float.random(in: -3...1, using: &generator) }
                for index in 0..<2 {
                    let full = DepthResampler(plane).resample(Array(planes[(16 * index)..<(16 * index + 16)])).map { $0 > 0 }
                    let mask = resampler.mask(planes, plane: index, threshold: 0)
                    #expect(mask.alpha().map { $0 == 255 } == full)
                    #expect(mask.area == full.filter { $0 }.count)
                }
            }
        }
    }

    @Test func segmentationDetectionsCarryTheirQuerysMask() async throws {
        // Query 0 (a cat and a dog) has a mask on the left column, query 1 (a person) on the bottom row.
        let outputs = Outputs(queries: 2, masks: [5, -5, 5, -5, -5, -5, 5, 5], logits: [(0, 17, 4), (0, 18, 2), (1, 1, 3)])
        let detector = try Self.detector(Self.manifest(masks: true), outputs)
        // A 4x4 source onto the 4x4 input, the 2x2 planes at stride 2.
        let detections = try await detector.detect(Self.image(width: 4, height: 4))
        #expect(detections.map(\.label.id) == [17, 1, 18])
        let cat = try #require(detections[0].mask), person = try #require(detections[1].mask)
        #expect(detections[2].mask == cat)
        #expect(cat.width == 4 && cat.height == 4)
        // Logits 5 and -5 cross zero halfway between the cells' centres (source x = 2).
        #expect(cat.bounds == .init(x: 0, y: 0, width: 2, height: 4) && cat.area == 8)
        #expect(person.bounds == .init(x: 0, y: 2, width: 4, height: 2) && person.area == 8)
        #expect(abs(cat.intersectionOverUnion(person) - 4.0 / 12) < 1e-12 && cat.intersectionOverUnion(cat) == 1)
    }

    // MARK: - Requests

    @Test func cancellationStopsAfterTheGraphCallAndTheDetectorRecovers() async throws {
        let log = Log(), gate = Gate()
        let detector = try Self.detector(Self.manifest(), Outputs(logits: [(0, 17, 4)]), log: log, gate: gate)
        let input = try Self.image(width: 4, height: 4)
        let task = Task { try await detector.detect(input) }
        await gate.waitUntilEntered()
        let queued = Task { try await detector.detect(input) }
        task.cancel()
        await gate.release()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(try await queued.value.map(\.label.id) == [17])
        #expect(log.entries == ["load detector", "run", "load detector", "run"])
        await detector.unload()
        #expect(await detector.residentRoles.isEmpty)
    }
}
