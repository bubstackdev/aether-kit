import AetherCoreAI
import AetherFlux2
import AetherFlux2CoreAI
import AetherFlux2Tokenizers
import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import AetherSpeech
import AetherSpeechCoreAI
import Darwin
import Foundation
import Metal
import Testing

/// Opt-in native load/unload cycling: in one process, N cycles of open session → one small request → `unload()` →
/// drop the session, recording memory after each unload. It tells a one-time residue (cycle 1 above the baseline,
/// later cycles flat) from per-cycle growth (each cycle above the last). Each test runs one model; run them one at a
/// time under the shared GPU lock, release build:
///
/// AETHER_NATIVE_UNLOAD_CYCLING=flux2|krea|kokoro
/// AETHER_NATIVE_UNLOAD_OUTPUT=<new directory>
/// AETHER_NATIVE_UNLOAD_CYCLES=<cycles, default 5>
/// AETHER_FLUX2_BUNDLE / AETHER_KREA_BUNDLE / AETHER_KOKORO_BUNDLE
/// AETHER_NATIVE_UNLOAD_KOKORO_CAP=extended (optional: prewarm through bucket 2048 each cycle)
/// AETHER_NATIVE_UNLOAD_MEMGRAPH=1 (optional: memory graphs of the baseline and the last cycle; `full` adds the whole
/// allocation history, for a run with MallocStackLogging=1)
///
/// The tokenizer (FLUX.2, Krea) is opened once before the baseline, so it is not part of the cycle. After each unload
/// the test records memory at once, after a 2 s settle, and after `malloc_zone_pressure_relief`, which returns free
/// malloc pages: what it returns was allocator caching, not live memory. `footprint` snapshots (per-category dirty and
/// wired memory) are saved for the baseline, after cycle 1 and after the last cycle.
private func unloadCyclingEnabled(_ model: String) -> Bool {
    nativeEnvironment["AETHER_NATIVE_UNLOAD_CYCLING"] == model && nativeEnvironment["AETHER_NATIVE_UNLOAD_OUTPUT"] != nil
}

@Test(.enabled(if: unloadCyclingEnabled("flux2")), .timeLimit(.minutes(20)))
func flux2UnloadCycling() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let assets = try Flux2Assets.open(at: nativePath("AETHER_FLUX2_BUNDLE"))
    let tokenizer = try await Flux2Tokenizer.open(assets: assets)
    let prompt = try tokenizer.prepare("a red fox in fresh snow")
    let configuration = try Flux2Configuration(width: 256, height: 256, steps: 4, seed: 7)
    try await runUnloadCycles(model: "flux2", fingerprint: assets.fingerprint,
                              request: "256x256, 4 steps, prompt through the text encoder, balanced residency") {
        let session = Flux2GenerationSession(assets: assets)
        let image = try await session.generate(configuration: configuration, prompt: prompt)
        let loaded = UnloadMemory.read()
        await session.unload()
        return (loaded, nativeHash(image.pixelsCHW.values))
    }
}

@Test(.enabled(if: unloadCyclingEnabled("krea")), .timeLimit(.minutes(20)))
func kreaUnloadCycling() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let assets = try KreaAssets.open(at: nativePath("AETHER_KREA_BUNDLE"))
    let tokenizer = try await KreaTokenizer.open(assets: assets)
    let prompt = try tokenizer.prepare("a red fox in fresh snow")
    let configuration = try KreaConfiguration(width: 256, height: 256, steps: 2, seed: 7)
    try await runUnloadCycles(model: "krea", fingerprint: assets.fingerprint,
                              request: "256x256, 2 steps, prompt through the text encoder, balanced residency, dynamic denoisers") {
        let session = KreaGenerationSession(assets: assets)
        let image = try await session.generate(configuration: configuration, prompt: prompt)
        let loaded = UnloadMemory.read()
        await session.unload()
        return (loaded, nativeHash(image.pixelsCHW.values))
    }
}

@Test(.enabled(if: unloadCyclingEnabled("kokoro")), .timeLimit(.minutes(20)))
func kokoroUnloadCycling() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let assets = try KokoroAssets.open(at: nativePath("AETHER_KOKORO_BUNDLE"))
    let extended = nativeEnvironment["AETHER_NATIVE_UNLOAD_KOKORO_CAP"] == "extended"
    let cap: KokoroFrameCap = extended ? .extended : .standard
    let configuration = try KokoroSpeechConfiguration()
    try await runUnloadCycles(model: extended ? "kokoro-extended" : "kokoro", fingerprint: assets.fingerprint,
                              request: extended ? "short utterance, then prewarm through bucket 2048 (frame cap .extended)"
                                                : "short utterance (frame cap .standard)") {
        let session = KokoroSpeechSession(assets: assets, frameCap: cap)
        let speech = try await session.synthesize(try session.phonemes("həlˈO"), configuration: configuration)
        if extended { try await session.prewarm(maximumFrames: KokoroFrameCap.extended.frames) }
        let loaded = UnloadMemory.read()
        await session.unload()
        return (loaded, nativeHash(speech.audio.channels.flatMap { $0 }))
    }
}

/// One sample of this process's memory counters. Footprint counters overlap; do not sum them.
struct UnloadMemory: Codable, Sendable {
    var footprintBytes: UInt64
    var footprintPeakBytes: Int64
    var residentBytes: UInt64
    var internalBytes: UInt64
    var compressedBytes: UInt64
    /// Kernel ledgers for IOKit memory tagged graphics (Metal) and neural (ANE), counted in the footprint.
    var graphicsFootprintBytes: Int64
    var neuralFootprintBytes: Int64
    /// All malloc zones: bytes in live allocations, and bytes the zones hold from the system.
    var mallocInUseBytes: UInt64
    var mallocAllocatedBytes: UInt64
    var metalAllocatedBytes: UInt64
    /// System-wide wired memory: other processes move it too.
    var systemWiredBytes: UInt64

    private static let device = MTLCreateSystemDefaultDevice()

    static func read() -> Self {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        _ = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        var inUse: UInt64 = 0, allocated: UInt64 = 0
        var zones: UnsafeMutablePointer<vm_address_t>?
        var zoneCount: UInt32 = 0
        if malloc_get_all_zones(mach_task_self_, nil, &zones, &zoneCount) == KERN_SUCCESS, let zones {
            for index in 0..<Int(zoneCount) {
                guard let zone = UnsafeMutablePointer<malloc_zone_t>(bitPattern: UInt(zones[index])) else { continue }
                var statistics = malloc_statistics_t()
                malloc_zone_statistics(zone, &statistics)
                inUse += UInt64(statistics.size_in_use); allocated += UInt64(statistics.size_allocated)
            }
        }
        var vm = vm_statistics64_data_t()
        var vmCount = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        _ = withUnsafeMutablePointer(to: &vm) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &vmCount) }
        }
        return Self(footprintBytes: info.phys_footprint, footprintPeakBytes: info.ledger_phys_footprint_peak,
                    residentBytes: info.resident_size, internalBytes: info.internal, compressedBytes: info.compressed,
                    graphicsFootprintBytes: info.ledger_tag_graphics_footprint, neuralFootprintBytes: info.ledger_tag_neural_footprint,
                    mallocInUseBytes: inUse, mallocAllocatedBytes: allocated,
                    metalAllocatedBytes: UInt64(device?.currentAllocatedSize ?? 0),
                    systemWiredBytes: UInt64(vm.wire_count) * UInt64(getpagesize()))
    }
}

private struct UnloadCycle: Codable {
    let cycle: Int
    let seconds: Double
    let outputSHA256: String
    let loaded: UnloadMemory
    let unloaded: UnloadMemory
    let settled: UnloadMemory
    let relieved: UnloadMemory
}

private struct UnloadCyclingReport: Codable {
    let model: String
    let bundleFingerprint: String
    let request: String
    let cycles: Int
    var baseline: UnloadMemory
    var runs: [UnloadCycle] = []
    var outputsIdentical = true
    var footprintSnapshots: [String] = []
    var scope = "One process; counters sampled at the listed points, not continuously. Footprint, resident, graphics and malloc "
        + "counters overlap and must not be summed. System wired memory includes every process."
}

private func runUnloadCycles(model: String, fingerprint: String, request: String,
                             cycle: () async throws -> (loaded: UnloadMemory, output: String)) async throws {
    let output = try nativePath("AETHER_NATIVE_UNLOAD_OUTPUT")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    let cycles = Int(nativeEnvironment["AETHER_NATIVE_UNLOAD_CYCLES"] ?? "5") ?? 5
    func snapshot(_ name: String) {
        let file = output.appendingPathComponent("footprint-\(name).json")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/footprint")
        process.arguments = ["--pid", String(getpid()), "--wired", "--format", "bytes", "-j", file.path]
        process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        if (try? process.run()) != nil { process.waitUntilExit() }
    }
    // With AETHER_NATIVE_UNLOAD_MEMGRAPH=1, a memory graph of the baseline and of the last cycle (`leaks --outputGraph`),
    // for `heap` and, when the run sets MallocStackLogging, `malloc_history`.
    func memgraph(_ name: String) {
        guard let mode = nativeEnvironment["AETHER_NATIVE_UNLOAD_MEMGRAPH"], ["1", "full"].contains(mode) else { return }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/leaks")
        process.arguments = ["--outputGraph=" + output.appendingPathComponent("\(name).memgraph").path, String(getpid())]
            + (mode == "full" ? ["--fullStackHistory"] : [])
        process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        if (try? process.run()) != nil { process.waitUntilExit() }
    }
    func relieve() -> UnloadMemory { malloc_zone_pressure_relief(nil, 0); return UnloadMemory.read() }
    _ = relieve()
    var report = UnloadCyclingReport(model: model, bundleFingerprint: fingerprint, request: request, cycles: cycles,
                                     baseline: UnloadMemory.read())
    snapshot("baseline"); report.footprintSnapshots.append("footprint-baseline.json")
    memgraph("baseline")
    func write() throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: output.appendingPathComponent("report.json"))
    }
    try write()
    let gib = { (bytes: UInt64) in String(format: "%.3f", Double(bytes) / 1_073_741_824) }
    for index in 1...cycles {
        let started = ContinuousClock.now
        let (loaded, digest) = try await cycle()
        let unloaded = UnloadMemory.read()
        let elapsed = ContinuousClock.now - started
        try await Task.sleep(for: .seconds(2))
        let settled = UnloadMemory.read()
        let relieved = relieve()
        if let first = report.runs.first, first.outputSHA256 != digest { report.outputsIdentical = false }
        report.runs.append(UnloadCycle(cycle: index, seconds: Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18,
                                       outputSHA256: digest, loaded: loaded, unloaded: unloaded, settled: settled, relieved: relieved))
        if index == 1 || index == cycles { snapshot("cycle\(index)"); report.footprintSnapshots.append("footprint-cycle\(index).json") }
        if index == cycles { memgraph("cycle\(index)") }
        try write()
        print("unload-cycling \(model) cycle \(index): loaded footprint \(gib(loaded.footprintBytes)) GiB, metal \(gib(loaded.metalAllocatedBytes)) GiB;"
              + " relieved footprint \(gib(relieved.footprintBytes)) GiB (baseline \(gib(report.baseline.footprintBytes))),"
              + " metal \(gib(relieved.metalAllocatedBytes)) GiB, malloc in use \(gib(relieved.mallocInUseBytes)) GiB,"
              + " graphics \(gib(UInt64(max(0, relieved.graphicsFootprintBytes)))) GiB")
    }
    #expect(report.outputsIdentical)
}
