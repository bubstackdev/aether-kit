import CoreGraphics
import Foundation
import Synchronization
import Testing
@_spi(Testing) import AetherHost

/// `StatelessPipeline` over fake graphs (DESIGN.md §12), as M4's depth driver will use it: a DA3-shaped manifest, a
/// prepared image bound into an fp16 input, residency and admission, GPU priority, cancellation and recovery.
@Suite("Stateless pipeline")
struct StatelessPipelineTests {
    static let manifestJSON = """
    { "schema": 1, "id": "depth-test", "task": "depth-estimation", "driver": "depth-estimator@1",
      "variant": { "platform": "macos" }, "assets": { "main": { "subtree": "" } },
      "graphs": { "estimator": { "file": "main:estimator.aimodel", "function": "main",
                                 "inputs": { "image": "image" }, "outputs": { "depth": "depth", "confidence": "depth_conf" },
                                 "compute": "gpu" } },
      "preprocess": { "image": { "resize": { "mode": "stretch", "size": [4, 2], "filter": "vImage" }, "colorSpace": "sRGB",
                                 "scale": 0.00392156862745098, "layout": "NCHW", "hostDType": "float32", "orientation": "exif",
                                 "x-notes": "ignored" } },
      "postprocess": { "kind": "depth", "depth": { "polarity": "largerIsFarther", "resizeBack": true } } }
    """

    static func manifest() throws -> Manifest { try Manifest.decode(Data(manifestJSON.utf8)) }

    static let contract = DriverContract(
        driver: "depth-estimator@1",
        graphs: ["estimator": .init(
            inputs: ["image": .init(required: true, rank: 4, scalar: .floatingPoint)], allowsStaticSourcedInputs: false,
            outputs: ["depth": .init(required: true, rank: 3, scalar: .floatingPoint),
                      "confidence": .init(required: false, rank: 3, scalar: .floatingPoint)],
            minimumSequenceStates: 0, allowsRecurrent: false, phases: ["step"], allowsFunctionFamily: false,
            requiresMaxContext: false)],
        requiresTokenizer: false)

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

    /// DA3's I/O at 2x4: depth = 10 · the image's first channel, confidence = 1.
    final class FakeEstimator: GraphBackend {
        let signature: GraphSignature
        let log: Log
        let gate: Gate?
        init(log: Log, gate: Gate?, withDepth: Bool = true) {
            signature = GraphSignature(
                function: "main", inputs: [TensorSpec(name: "image", scalarType: .float16, shape: [1, 3, 2, 4])], states: [],
                outputs: (withDepth ? [TensorSpec(name: "depth", scalarType: .float32, shape: [1, 2, 4])] : [])
                    + [TensorSpec(name: "depth_conf", scalarType: .float32, shape: [1, 2, 4])])
            self.log = log
            self.gate = gate
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            log.append("run")
            await gate?.enter()
            let image = try bindings.input("image")!.read(Float16.self)
            try bindings.output("depth")!.write((0..<8).map { 10 * Float(image[$0]) })
            try bindings.output("depth_conf")!.fill(Float(1))
        }
    }

    /// What a depth driver built on the pipeline looks like.
    actor Depth {
        let pipeline: StatelessPipeline
        let preprocessor: ImagePreprocessor

        init(log: Log, gate: Gate? = nil, gpu: GPUAdmissionOwner = .unadmitted, budget: Int64 = 1 << 40,
             withDepth: Bool = true) throws {
            let manifest = try StatelessPipelineTests.manifest()
            preprocessor = try ImagePreprocessor(manifest: manifest)
            pipeline = try StatelessPipeline(
                manifest: manifest, device: Device(platform: .macOS, architecture: "test"), contract: StatelessPipelineTests.contract,
                options: PipelineOptions(memory: .bytes(budget), gpu: gpu, probe: GPUMemoryProbe { 0 }),
                load: { role throws(AetherError) in
                    log.append("load \(role)")
                    return FakeEstimator(log: log, gate: gate, withDepth: withDepth)
                },
                fileBytes: { _ in 100 })
        }

        func estimate(_ image: PreparedImage, job: GPUJob? = nil) async throws -> [Float] {
            try await pipeline.perform(job: job) { residency in
                let graph = try await residency.graph("estimator")
                let arena = TensorArena()
                var bindings = Bindings(arena: arena)
                try bindings.bindInput("image", to: graph.makeBuffer(for: "image", holding: image, in: arena))
                let depth = try graph.makeBuffer(for: "depth", role: .output, dimensions: [], in: arena)
                try bindings.bindOutput("depth", to: depth)
                try bindings.bindOutput("depth_conf", to: graph.makeBuffer(for: "depth_conf", role: .output, dimensions: [], in: arena))
                try await residency.run(graph, &bindings)
                return try depth.readFloats()
            }
        }

        func unload() async { await pipeline.unload() }
        var residentRoles: [String] { pipeline.residentRoles }
    }

    /// A 4x2 image whose red channel is 0, 51, …, 255 · k/7.
    static func image(_ preprocessor: ImagePreprocessor) throws -> PreparedImage {
        let pixels: [UInt8] = (0..<8).flatMap { (i: Int) -> [UInt8] in [UInt8(i * 255 / 7), 0, 0, 255] }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        let image = try #require(CGImage(
            width: 4, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 16, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent))
        return try preprocessor.prepare(ImageInput(image))
    }

    @Test func aRequestBindsThePreparedImageAndTheGraphStaysResident() async throws {
        let log = Log()
        let depth = try Depth(log: log)
        let prepared = try Self.image(depth.preprocessor)
        #expect(prepared.shape == [1, 3, 2, 4] && prepared.scalarType == .float32)
        let first = try await depth.estimate(prepared)
        // The fp32 host values went into the fp16 input, rounded to nearest even.
        #expect(first == prepared.floats()[0..<8].map { 10 * Float(Float16($0)) })
        _ = try await depth.estimate(prepared)
        #expect(log.entries == ["load estimator", "run", "run"])
        #expect(await depth.residentRoles == ["estimator"])
        await depth.unload()
        #expect(await depth.residentRoles.isEmpty)
        _ = try await depth.estimate(prepared)
        #expect(log.entries.filter { $0.hasPrefix("load") }.count == 2)
    }

    @Test func theContractIsCheckedAtInitAndEachGraphAtLoad() async throws {
        // The manifest against the driver's roles.
        var wrong = try Self.manifest()
        wrong.graphs["extra"] = wrong.graphs["estimator"]
        #expect(throws: AetherError.self) {
            try StatelessPipeline(manifest: wrong, device: Device(platform: .macOS, architecture: "test"), contract: Self.contract,
                                  options: PipelineOptions(gpu: .unadmitted), load: { _ throws(AetherError) in fatalError() },
                                  fileBytes: { _ in 0 })
        }
        // A graph without the manifest's `depth` output is refused when it loads, and nothing stays resident.
        let log = Log()
        let depth = try Depth(log: log, withDepth: false)
        let prepared = try Self.image(depth.preprocessor)
        await #expect(throws: AetherError.self) { try await depth.estimate(prepared) }
        #expect(await depth.residentRoles.isEmpty && !log.entries.contains("run"))
    }

    @Test func aRequestThatDoesNotFitIsRefusedBeforeAnythingLoads() async throws {
        let log = Log()
        let depth = try Depth(log: log, budget: 50)
        let prepared = try Self.image(depth.preprocessor)
        await #expect {
            try await depth.estimate(prepared)
        } throws: { error in
            if case .capacity = error as? AetherError { true } else { false }
        }
        #expect(log.entries.isEmpty)
    }

    @Test func cancellationStopsAfterTheGraphCallAndThePipelineRecovers() async throws {
        let log = Log(), gate = Gate()
        let depth = try Depth(log: log, gate: gate)
        let prepared = try Self.image(depth.preprocessor)
        let task = Task { try await depth.estimate(prepared) }
        await gate.waitUntilEntered()
        // A second request waits its turn (no busy error) and runs once the first ends.
        let queued = Task { try await depth.estimate(prepared) }
        task.cancel()
        await gate.release()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(try await queued.value.count == 8)
        // The cancelled request released the graphs; the queued one loaded them again.
        #expect(log.entries == ["load estimator", "run", "load estimator", "run"])
    }

    @Test func requestsTakeTheGPUInPriorityOrder() async throws {
        let admission = try GPUAdmission()
        let log = Log()
        let depth = try Depth(log: log, gpu: admission.owner("depth"))
        let prepared = try Self.image(depth.preprocessor)
        let held = try await admission.admit(owner: "image")
        let order = Log()
        func request(_ label: String, _ job: GPUJob?) -> Task<Void, any Error> {
            Task {
                _ = try await depth.estimate(prepared, job: job)
                order.append(label)
            }
        }
        let low = GPUJob(priority: .low)
        let a = request("low", low)
        while admission.snapshot.queued < 1 { await Task.yield() }
        let b = request("medium", nil)
        while admission.snapshot.queued < 2 { await Task.yield() }
        let c = request("high", GPUJob(priority: .high))
        while admission.snapshot.queued < 3 { await Task.yield() }
        held.release()
        for task in [a, b, c] { try await task.value }
        #expect(order.entries == ["high", "medium", "low"])
        #expect(admission.snapshot.running == 0 && admission.snapshot.queued == 0)
    }

    @Test func theManifestsVisionBlocksAreTyped() throws {
        let manifest = try Self.manifest()
        #expect(manifest.postprocess == .depth(.init(polarity: .largerIsFarther, resizeBack: true)))
        #expect(manifest.preprocess?.image?.resize?.size == .init(width: 4, height: 2))
        #expect(manifest.preprocess?.image?.colorSpace == .sRGB)
        #expect(try Manifest.decode(manifest.encoded()) == manifest)
    }
}
