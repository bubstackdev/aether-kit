import AetherCoreAI
import AetherKit
import AetherSpeech
import Darwin
import Foundation
import Metal
import Testing

/// Process footprint (includes GPU allocations on unified memory) and the default Metal device's
/// allocation for this process, in bytes.
func kokoroMemorySnapshot() -> (footprint: UInt64, metal: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    let footprint = result == KERN_SUCCESS ? info.phys_footprint : 0
    return (footprint, UInt64(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0))
}

private struct SweepPoint: Encodable {
    let frames: Int
    let shape: Int
    let firstSeconds: Double
    let repeatSeconds: Double
    let footprintMiB: Double
    let metalMiB: Double
}

/// Opt-in decoder shape sweep: runs the bundle's decoder directly at the listed frame counts
/// (synthetic inputs, real weights), recording first-call (specialization) and repeat latency and
/// memory after each call. With `AETHER_NATIVE_KOKORO_SWEEP_BUCKETS` set, a bucketed decoder runs
/// each count padded to its bucket, so the shape set is bounded.
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_KOKORO_SWEEP"] == "1"), .timeLimit(.minutes(60)))
func kokoroNativeDecoderShapeSweep() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = try nativePath("AETHER_NATIVE_KOKORO_OUTPUT")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    let assets = try KokoroAssets.open(at: nativePath("AETHER_KOKORO_BUNDLE"))
    let frames = try #require(nativeEnvironment["AETHER_NATIVE_KOKORO_SWEEP_FRAMES"]).split(separator: ",").compactMap { Int($0) }
    let buckets = (nativeEnvironment["AETHER_NATIVE_KOKORO_SWEEP_BUCKETS"] ?? "").split(separator: ",").compactMap { Int($0) }
    let baseline = kokoroMemorySnapshot()
    let session = try await ModelSession.open(at: assets.decoder.url, compute: .preferGPU)
    let bucketed = session.functions.first { $0.name == "decode" }?.inputs["frames"] != nil
    #expect(bucketed == !buckets.isEmpty)
    let opened = kokoroMemorySnapshot()
    var points: [SweepPoint] = []
    let style = try FloatTensor(shape: [1, 256], values: (0..<256).map { Float(($0 % 17) - 8) / 80 })
    func run(_ count: Int) async throws -> (Double, Int) {
        let shape = bucketed ? try #require(buckets.first { $0 >= count }) : count
        var inputs: [String: TensorValue] = [
            "prosody": .float32(try FloatTensor(shape: [1, 640, shape], values: (0..<(640 * shape)).map { i in i % shape < count ? Float(i % 23) / 23 - 0.5 : 0 })),
            "text": .float32(try FloatTensor(shape: [1, 512, shape], values: (0..<(512 * shape)).map { i in i % shape < count ? Float(i % 19) / 19 - 0.5 : 0 })),
            "style": .float32(style),
            "noise": .float32(try FloatTensor(shape: [1, shape, 600, 9], values: KokoroNoise.generate(seed: UInt64(count), count: shape * 5400))),
        ]
        if bucketed { inputs["frames"] = .int32(try Int32Tensor(shape: [1], values: [Int32(count)])) }
        let started = ContinuousClock.now
        _ = try await session.runTensors(function: "decode", inputs: inputs)
        let elapsed = ContinuousClock.now - started
        return (Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18, shape)
    }
    for count in frames {
        let (first, shape) = try await run(count)
        let (again, _) = try await run(count)
        let memory = kokoroMemorySnapshot()
        points.append(SweepPoint(frames: count, shape: shape, firstSeconds: first, repeatSeconds: again,
                                 footprintMiB: Double(memory.footprint) / 1_048_576, metalMiB: Double(memory.metal) / 1_048_576))
    }
    await session.unload()
    let closed = kokoroMemorySnapshot()
    struct Report: Encodable {
        let bundleFingerprint: String
        let bucketed: Bool
        let buckets: [Int]
        let baselineFootprintMiB: Double
        let openedFootprintMiB: Double
        let closedFootprintMiB: Double
        let closedMetalMiB: Double
        let points: [SweepPoint]
    }
    let report = Report(bundleFingerprint: assets.fingerprint, bucketed: bucketed, buckets: buckets,
                        baselineFootprintMiB: Double(baseline.footprint) / 1_048_576, openedFootprintMiB: Double(opened.footprint) / 1_048_576,
                        closedFootprintMiB: Double(closed.footprint) / 1_048_576, closedMetalMiB: Double(closed.metal) / 1_048_576, points: points)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(report).write(to: output.appendingPathComponent("sweep.json"))
}
