import CoreGraphics
import Foundation
import Synchronization
import Testing
@_spi(Testing) import AetherHost
@testable import AetherVision

/// `depth-estimator@1` over fake graphs (DESIGN.md §12): the contract, the `depth` postprocess (resize-back geometry
/// and polarity), the request options and cancellation. A constant probe and no GPU admission, so nothing depends on
/// the machine.
@Suite("Depth estimator")
struct DepthEstimatorTests {
    /// A DA3-shaped manifest with a 4x2 graph input; `edit` changes the JSON object before decoding.
    static func manifest(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> Manifest {
        var json: [String: Any] = [
            "schema": 1, "id": "depth-test", "task": "depth-estimation", "driver": "depth-estimator@1",
            "variant": ["platform": "macos"], "assets": ["main": ["subtree": ""]],
            "graphs": ["estimator": [
                "file": "main:estimator.aimodel", "function": "main", "inputs": ["image": "image"],
                "outputs": ["depth": "depth", "confidence": "depth_conf"], "compute": "gpu",
            ] as [String: Any]],
            "preprocess": ["image": [
                "resize": ["mode": "stretch", "size": [4, 2], "filter": "vImage"] as [String: Any], "colorSpace": "sRGB",
                "scale": 1.0 / 255, "layout": "NCHW", "hostDType": "float32", "orientation": "exif",
            ] as [String: Any]],
            "postprocess": ["kind": "depth", "depth": ["polarity": "largerIsFarther", "resizeBack": true] as [String: Any]],
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

    /// DA3's I/O at `width x height`: depth = 10 · the image's red channel, confidence = 1 + the green channel.
    final class FakeEstimator: GraphBackend {
        let signature: GraphSignature
        let log: Log
        let gate: Gate?
        let plane: Int
        init(log: Log, gate: Gate?, width: Int = 4, height: Int = 2, outputHeight: Int? = nil) {
            plane = width * height
            let out = [1, outputHeight ?? height, width]
            signature = GraphSignature(
                function: "main", inputs: [TensorSpec(name: "image", scalarType: .float16, shape: [1, 3, height, width])],
                states: [],
                outputs: [TensorSpec(name: "depth", scalarType: .float32, shape: out),
                          TensorSpec(name: "depth_conf", scalarType: .float32, shape: out)])
            self.log = log
            self.gate = gate
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            log.append(bindings.output("depth_conf") == nil ? "run (no confidence)" : "run")
            await gate?.enter()
            let image = try bindings.input("image")!.read(Float16.self)
            try bindings.output("depth")!.write((0..<plane).map { 10 * Float(image[$0]) })
            try bindings.output("depth_conf")?.write((0..<plane).map { 1 + Float(image[plane + $0]) })
        }
    }

    static let device = Device(platform: .macOS, architecture: "test")

    static func estimator(
        _ manifest: Manifest, log: Log = Log(), gate: Gate? = nil, outputHeight: Int? = nil
    ) throws -> DepthEstimator {
        try DepthEstimator(
            manifest: manifest, device: device,
            options: PipelineOptions(memory: .bytes(1 << 40), gpu: .unadmitted, probe: GPUMemoryProbe { 0 }),
            load: { role throws(AetherError) in
                log.append("load \(role)")
                return FakeEstimator(log: log, gate: gate, outputHeight: outputHeight)
            },
            fileBytes: { _ in 100 })
    }

    /// A `width x height` opaque sRGB image, `pixel(x, y)` its RGB.
    static func image(width: Int, height: Int, _ pixel: (Int, Int) -> (UInt8, UInt8, UInt8)) throws -> ImageInput {
        var bytes: [UInt8] = []
        for y in 0..<height {
            for x in 0..<width {
                let (r, g, b) = pixel(x, y)
                bytes += [r, g, b, 255]
            }
        }
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

    // MARK: - Contract

    @Test func theDriverIsRegisteredWithItsContract() throws {
        #expect(Registry.vision.supports(.depthEstimator))
        #expect(try Registry.vision.driver(for: .depthEstimator) == DepthEstimatorDriver.self)
        #expect(DriverCatalog.contracts[.depthEstimator] == .depthEstimator)
        try DriverContract.depthEstimator.validate(Self.manifest())
    }

    @Test func theManifestIsCheckedBeforeAnythingLoads() throws {
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("graphs.estimator.outputs.depth", { json in
                json["graphs"] = ["estimator": ["file": "main:e.aimodel", "function": "main", "inputs": ["image": "image"],
                                                "outputs": ["confidence": "depth_conf"], "compute": "gpu"] as [String: Any]]
            }),
            ("graphs.extra", { json in
                var graphs = json["graphs"] as! [String: Any]
                graphs["extra"] = graphs["estimator"]
                json["graphs"] = graphs
            }),
            ("postprocess", { json in json["postprocess"] = ["kind": "gliner"] }),
            ("postprocess", { json in json["postprocess"] = nil }),
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
            #expect("\(field)", performing: { _ = try Self.estimator(manifest) }, throws: Self.isInvalid(field))
        }
    }

    @Test func aGraphWhoseMapIsNotThePreparedImagesSizeIsRefused() async throws {
        let log = Log()
        let estimator = try Self.estimator(Self.manifest(), log: log, outputHeight: 3)
        let input = try Self.image(width: 4, height: 2) { _, _ in (0, 0, 0) }
        await #expect {
            _ = try await estimator.estimate(input)
        } throws: { error in
            if case .manifestMismatch(let field, _, _)? = error as? AetherError { field == "graphs.estimator.outputs" } else { false }
        }
        #expect(await estimator.residentRoles.isEmpty && !log.entries.contains("run"))
    }

    // MARK: - Postprocess

    @Test func resizeBackIsBilinearAtPixelCentres() {
        // A stretch from 4x1 source pixels onto a 2x1 map: centres at -0.25, 0.25, 0.75, 1.25, clamped.
        let stretch = DepthResampler(ImageGeometry(sourceWidth: 4, sourceHeight: 1, width: 2, height: 1,
                                                   content: CGRect(x: 0, y: 0, width: 2, height: 1)))
        #expect(stretch.resample([0, 10]) == [0, 2.5, 7.5, 10])
        // Same size: the identity.
        let same = DepthResampler(ImageGeometry(sourceWidth: 3, sourceHeight: 2, width: 3, height: 2,
                                                content: CGRect(x: 0, y: 0, width: 3, height: 2)))
        #expect(same.resample([1, 2, 3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6])
        // Two dimensions: a 2x2 map onto 4x4, separable.
        let square = DepthResampler(ImageGeometry(sourceWidth: 4, sourceHeight: 4, width: 2, height: 2,
                                                  content: CGRect(x: 0, y: 0, width: 2, height: 2)))
        let out = square.resample([0, 4, 8, 12])
        #expect(out[0..<4] == [0, 1, 3, 4] && out[4..<8] == [2, 3, 5, 6] && out[12..<16] == [8, 9, 11, 12])
    }

    @Test func resizeBackNeverSamplesLetterboxPadding() {
        // A 2x2 source letterboxed into the middle columns of a 4x2 canvas; the padding columns hold a sentinel.
        let geometry = ImageGeometry(sourceWidth: 2, sourceHeight: 2, width: 4, height: 2,
                                     content: CGRect(x: 1, y: 0, width: 2, height: 2))
        let map: [Float] = [999, 1, 2, 999,
                            999, 3, 4, 999]
        #expect(DepthResampler(geometry).resample(map) == [1, 2, 3, 4])
        // A wider source on the same content: interpolated inside it, clamped at its edges, still never the padding.
        let wide = DepthResampler(ImageGeometry(sourceWidth: 4, sourceHeight: 2, width: 4, height: 2,
                                                content: CGRect(x: 1, y: 0, width: 2, height: 2)))
        let out = wide.resample(map)
        #expect(out.allSatisfy { $0 < 999 })
        #expect(out[0..<4] == [1, 1.25, 1.75, 2])
        // Centre-cropped: the source extends past the canvas, which clamps to its edge values.
        let crop = DepthResampler(ImageGeometry(sourceWidth: 4, sourceHeight: 1, width: 2, height: 1,
                                                content: CGRect(x: -1, y: 0, width: 4, height: 1)))
        #expect(crop.resample([5, 7]) == [5, 5, 7, 7])
    }

    @Test func anEstimateIsResizedBackToTheSourceWithItsPolarityAndConfidence() async throws {
        let log = Log()
        let estimator = try Self.estimator(Self.manifest(), log: log)
        // 8x4, red rising left to right, so the map must rise left to right at the source's size.
        let input = try Self.image(width: 8, height: 4) { x, _ in (UInt8(x * 32), 51, 0) }
        let map = try await estimator.estimate(input)
        #expect(map.width == 8 && map.height == 4 && map.isSourceSized)
        #expect(map.polarity == .largerIsFarther)
        #expect(map.geometry == ImageGeometry(sourceWidth: 8, sourceHeight: 4, width: 4, height: 2,
                                              content: CGRect(x: 0, y: 0, width: 4, height: 2)))
        #expect(map.values.count == 32 && map.values.allSatisfy(\.isFinite))
        for y in 0..<4 {
            let row = (0..<8).map { map[x: $0, y: y] }
            #expect(zip(row, row.dropFirst()).allSatisfy { $0 <= $1 }, "row \(y): \(row)")
        }
        let confidence = try #require(map.confidence)
        #expect(confidence.count == 32 && confidence.allSatisfy { $0 == 1 + Float(Float16(Float(51) / 255)) })
        #expect(map.range.map { $0.lowerBound < $0.upperBound } == true)
        #expect(log.entries == ["load estimator", "run"])

        // The request keeps the graph's resolution; the geometry places it.
        let raw = try await estimator.estimate(input, DepthRequest(resizeBack: false))
        #expect(raw.width == 4 && raw.height == 2 && !raw.isSourceSized && raw.geometry == map.geometry)
        let prepared = try estimator.prepare(input)
        #expect(raw.values == prepared.floats()[0..<8].map { 10 * Float(Float16($0)) })
        // The prepared (CPU) and the graph (GPU) halves separately give the same map.
        #expect(try await estimator.estimate(prepared) == map)
    }

    @Test func polarityAndResizeBackComeFromTheManifest() async throws {
        let manifest = try Self.manifest { json in json["postprocess"] = ["kind": "depth", "depth": ["polarity": "largerIsNearer"]] }
        let estimator = try Self.estimator(manifest)
        #expect(estimator.postprocess == .init(polarity: .largerIsNearer))
        let input = try Self.image(width: 8, height: 4) { _, _ in (0, 0, 0) }
        let map = try await estimator.estimate(input)
        // resizeBack defaults to false.
        #expect(map.polarity == .largerIsNearer && map.width == 4 && map.height == 2)
        let back = try await estimator.estimate(input, DepthRequest(resizeBack: true))
        #expect(back.width == 8 && back.height == 4)
    }

    @Test func confidenceIsOptional() async throws {
        let manifest = try Self.manifest { json in
            json["graphs"] = ["estimator": ["file": "main:estimator.aimodel", "function": "main", "inputs": ["image": "image"],
                                            "outputs": ["depth": "depth"], "unusedOutputs": ["depth_conf"], "compute": "gpu"] as [String: Any]]
        }
        let log = Log()
        let estimator = try Self.estimator(manifest, log: log)
        let map = try await estimator.estimate(try Self.image(width: 4, height: 2) { _, _ in (10, 0, 0) })
        #expect(map.confidence == nil && map.values.count == 8)
        #expect(log.entries == ["load estimator", "run (no confidence)"])
    }

    // MARK: - Requests

    @Test func cancellationStopsAfterTheGraphCallAndTheEstimatorRecovers() async throws {
        let log = Log(), gate = Gate()
        let estimator = try Self.estimator(Self.manifest(), log: log, gate: gate)
        let input = try Self.image(width: 4, height: 2) { _, _ in (0, 0, 0) }
        let task = Task { try await estimator.estimate(input) }
        await gate.waitUntilEntered()
        // A second request waits its turn and runs once the first ends.
        let queued = Task { try await estimator.estimate(input) }
        task.cancel()
        await gate.release()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(try await queued.value.values.count == 8)
        #expect(log.entries == ["load estimator", "run", "load estimator", "run"])
        await estimator.unload()
        #expect(await estimator.residentRoles.isEmpty)
    }

    @Test func anUndecodableImageFailsBeforeTheGraphLoads() async throws {
        let log = Log()
        let estimator = try Self.estimator(Self.manifest(), log: log)
        await #expect(throws: AetherError.self) {
            _ = try await estimator.estimate(ImageInput(contentsOf: URL(filePath: "/nonexistent/aether-depth.png")))
        }
        #expect(log.entries.isEmpty)
    }
}
