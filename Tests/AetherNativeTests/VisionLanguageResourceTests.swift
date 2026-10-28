import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision
import Darwin
import Foundation
import Metal
import Testing

/// Samples Metal's allocated size and the process footprint every 50 ms, and the host's memorystatus level and free
/// memory, keeping each phase's peaks (and the host's minima) between `mark` calls. With `abortBelowLevel` or
/// `abortBelowFreeBytes`, a sample under either sets `tripped` (the coexistence guard).
private final class ResourceSampler: @unchecked Sendable {
    struct Window: Codable {
        let phase: String
        let metalBeforeMiB: Int, metalPeakMiB: Int, metalAfterMiB: Int
        let footprintBeforeMiB: Int, footprintPeakMiB: Int, footprintAfterMiB: Int
        let minimumMemorystatusLevel: Int, minimumFreeMiB: Int
        let samples: Int
        var reservedMiB: Int?
    }
    private let lock = NSLock()
    private let device = MTLCreateSystemDefaultDevice()
    private var thread: Thread?
    private var running = true
    private var phaseStart: (metal: Int, footprint: Int) = (0, 0)
    private var peak: (metal: Int, footprint: Int) = (0, 0)
    private var low: (level: Int, free: Int) = (.max, .max)
    private var count = 0
    private(set) var windows: [Window] = []
    private let abortBelowLevel: Int?, abortBelowFreeBytes: Int?
    private var trippedReason: String?
    var tripped: String? { lock.withLock { trippedReason } }

    init(abortBelowLevel: Int? = nil, abortBelowFreeBytes: Int? = nil) {
        self.abortBelowLevel = abortBelowLevel
        self.abortBelowFreeBytes = abortBelowFreeBytes
        let now = Self.read(device)
        phaseStart = (now.metal, now.footprint); peak = phaseStart
        let thread = Thread { [weak self] in
            while let self, self.lock.withLock({ self.running }) {
                self.sample()
                Thread.sleep(forTimeInterval: 0.05)
            }
        }
        self.thread = thread
        thread.start()
    }

    func stop() { lock.withLock { running = false } }

    private static func read(_ device: (any MTLDevice)?) -> (metal: Int, footprint: Int, level: Int, free: Int) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let ok = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        } == KERN_SUCCESS
        var level: Int32 = 0
        var size = MemoryLayout<Int32>.size
        sysctlbyname("kern.memorystatus_level", &level, &size, nil, 0)
        var vm = vm_statistics64_data_t()
        var vmCount = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let host = withUnsafeMutablePointer(to: &vm) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &vmCount) }
        } == KERN_SUCCESS
        let free = host ? Int(vm.free_count) * Int(getpagesize()) : Int.max
        return ((device?.currentAllocatedSize ?? 0) >> 20, ok ? Int(info.phys_footprint >> 20) : 0, Int(level), free)
    }

    private func sample() {
        let now = Self.read(device)
        lock.withLock {
            peak = (max(peak.metal, now.metal), max(peak.footprint, now.footprint))
            low = (min(low.level, now.level), min(low.free, now.free))
            count += 1
            if trippedReason == nil {
                if let floor = abortBelowLevel, now.level < floor { trippedReason = "memorystatus level \(now.level) below \(floor)" }
                if let floor = abortBelowFreeBytes, now.free < floor { trippedReason = "free memory \(now.free >> 20) MiB below \(floor >> 20) MiB" }
            }
        }
    }

    /// Ends the current window as `phase` (a final sample included) and starts the next.
    @discardableResult
    func mark(_ phase: String, reserved: UInt64? = nil) -> Window {
        sample()
        let now = Self.read(device)
        return lock.withLock {
            let window = Window(phase: phase, metalBeforeMiB: phaseStart.metal, metalPeakMiB: peak.metal, metalAfterMiB: now.metal,
                                footprintBeforeMiB: phaseStart.footprint, footprintPeakMiB: peak.footprint, footprintAfterMiB: now.footprint,
                                minimumMemorystatusLevel: low.level, minimumFreeMiB: low.free == .max ? -1 : low.free >> 20,
                                samples: count, reservedMiB: reserved.map { Int($0 >> 20) })
            windows.append(window)
            phaseStart = (now.metal, now.footprint); peak = phaseStart; low = (.max, .max); count = 0
            return window
        }
    }
}

private func gradient(width: Int, height: Int, seed: Int) throws -> VisionRGBImage {
    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let i = 3 * (y * width + x)
            rgb[i] = UInt8((x * 255 / max(1, width - 1) + seed * 40) % 256)
            rgb[i + 1] = UInt8((y * 255 / max(1, height - 1) + seed * 70) % 256)
            rgb[i + 2] = UInt8(((x + y) * 127 / max(1, width + height - 2) + seed * 90) % 256)
        }
    }
    return try VisionRGBImage(rgb: rgb, width: width, height: height)
}

/// The resource profile under test: the design's resident estimates (docs/qwen3-vl-resource-admission.md) with a
/// generous capacity, so the measurement observes, rather than trips, the ledger.
private func profile(for manifest: LanguageModelManifest, coordinator: ResourceCoordinator) throws -> LanguageResourceAdmission {
    guard let vision = manifest.vision else { throw LanguageGenerationError.invalid("not a Qwen3-VL bundle") }
    let fourB = manifest.sourceRepository.contains("4B")
    let mib: UInt64 = 1 << 20
    return try LanguageResourceAdmission(coordinator: coordinator, models: [
        try LanguageModelResourceEstimate(modelSHA256: manifest.modelSHA256, residentBytes: (fourB ? 11_600 : 6_300) * mib,
                                          runScratchBytes: (fourB ? 512 : 256) * mib),
        try LanguageModelResourceEstimate(modelSHA256: vision.modelSHA256,
                                          residentBytes: (fourB ? 2_200 : 2_000) * mib, runScratchBytes: 256 * mib),
    ], maximumContextLength: manifest.maximumContextLength, maximumImageTokens: vision.maximumImageTokens)
}

private struct PeakReport: Codable {
    var device = LanguageGenerationSession.deviceArchitectureName
    var bundle = ""
    var hostLoad = "Shared host: GPU jobs of cooperating agents are serialized by a lock, other host work is not; host minima include other processes"
    var method = "Metal currentAllocatedSize and phys_footprint sampled every 50 ms on a separate thread; each window's peak against the value at its start"
    var windows: [ResourceSampler.Window] = []
    var reservedAfterOpenMiB = 0
    var reservedAfterUnloadMiB = 0
    var error: String?
}

/// Opt-in measurement of a Qwen3-VL session's transient memory, for the resource profile's scratch estimates: the
/// encoder's and decoder's opening (load and specialization), the first and a warm text request, the first image
/// request at each grid and a warm one, then unload, each as a window with its sampled peak. The session is opened with
/// GPU admission and the design's resource profile, so the reservations are checked natively too. Run under the shared
/// GPU lease, release build:
///
/// AETHER_NATIVE_VL_RESOURCES=1
/// AETHER_VL_RESOURCES_BUNDLE=<registered Qwen3-VL bundle>
/// AETHER_VL_RESOURCES_OUTPUT=<report .json>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_VL_RESOURCES"] == "1"), .timeLimit(.minutes(30)))
func visionLanguageResourcePeaks() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_VL_RESOURCES_OUTPUT"]))
    let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_VL_RESOURCES_BUNDLE"]))
    var report = PeakReport()
    report.bundle = bundle.lastPathComponent
    let sampler = ResourceSampler()
    defer {
        sampler.stop()
        report.windows = sampler.windows
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    do {
        let manifest = try LanguageModelManifest.load(from: bundle)
        let registration = try #require(manifest.vision)
        let coordinator = try ResourceCoordinator(capacityBytes: 48 << 30)
        let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
        // A private, in-process admission: the native-test lease above already excludes other processes.
        let admission = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        sampler.mark("before open")
        let session = try await VisionLanguageSession.open(bundle: bundle, platform: .macos, gpu: admission.owner("describer"),
                                                           resources: try profile(for: manifest, coordinator: coordinator))
        report.reservedAfterOpenMiB = Int(coordinator.snapshot.reservedBytes >> 20)
        sampler.mark("open (encoder and decoder, load and specialization)", reserved: coordinator.snapshot.reservedBytes)
        let text = try tokenizer.render([.user(String(repeating: "The lighthouse keeper paints the railing a calm teal. ", count: 40))]).tokenIDs
        for label in ["first text request", "warm text request"] {
            _ = try await session.generate(renderedTokenIDs: text, images: [], maximumTokens: 16)
            sampler.mark(label, reserved: coordinator.snapshot.reservedBytes)
        }
        let rendered = try tokenizer.render([.user("Describe the colours in one line.", images: 1)]).tokenIDs
        for (index, grid) in try #require(registration.imageGrids).enumerated() {
            let image = try gradient(width: grid.width * registration.patchSize, height: grid.height * registration.patchSize, seed: index)
            _ = try await session.generate(renderedTokenIDs: rendered, images: [image], maximumTokens: 16)
            sampler.mark("image request, grid \(grid.height)x\(grid.width)", reserved: coordinator.snapshot.reservedBytes)
        }
        let image = try gradient(width: 512, height: 384, seed: 9)
        _ = try await session.generate(renderedTokenIDs: rendered, images: [image], maximumTokens: 16)
        sampler.mark("warm image request", reserved: coordinator.snapshot.reservedBytes)
        await session.unload()
        report.reservedAfterUnloadMiB = Int(coordinator.snapshot.reservedBytes >> 20)
        sampler.mark("unload", reserved: coordinator.snapshot.reservedBytes)
        #expect(report.reservedAfterUnloadMiB == 0)
    } catch {
        report.error = String(describing: error)
        throw error
    }
}

private struct CoexistenceReport: Codable {
    var device = LanguageGenerationSession.deviceArchitectureName
    var bundles: [String] = []
    var guardRule = ""
    var rounds = 0
    var completedRounds = 0
    var abortedBecause: String?
    var windows: [ResourceSampler.Window] = []
    var reservedPeakMiB = 0
    var error: String?
}

/// Opt-in, guarded measurement of a Qwen3-VL describer and a Krea 1024x1024 render sharing one process, one GPU
/// admission and one `ResourceCoordinator`: describer requests and renders alternate, and every window records the
/// sampled peaks and the host's minimum memorystatus level and free memory. If a sample falls below the guard (level 60
/// or 4 GiB free by default), the test stops at the next step boundary (a native call already running finishes first),
/// unloads both sessions and records why. It measures the residency overlap that admission bounds, not the
/// cross-process case. Run only in a window the lead gives, on a quiet host, under the shared GPU lease:
///
/// AETHER_NATIVE_VL_KREA_COEXISTENCE=1
/// AETHER_VL_COEXISTENCE_BUNDLE=<registered one-function Qwen3-VL bundle>
/// AETHER_KREA_BUNDLE=<Krea bundle>
/// AETHER_VL_COEXISTENCE_OUTPUT=<report .json>
/// AETHER_VL_COEXISTENCE_ROUNDS=<default 3>
/// AETHER_VL_COEXISTENCE_MIN_LEVEL=<default 60>; AETHER_VL_COEXISTENCE_MIN_FREE_MIB=<default 4096>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_VL_KREA_COEXISTENCE"] == "1"), .timeLimit(.minutes(40)))
func visionLanguageBesideKreaGuarded() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_VL_COEXISTENCE_OUTPUT"]))
    let vlBundle = URL(filePath: try #require(nativeEnvironment["AETHER_VL_COEXISTENCE_BUNDLE"]))
    let kreaRoot = URL(filePath: try #require(nativeEnvironment["AETHER_KREA_BUNDLE"]))
    let level = Int(nativeEnvironment["AETHER_VL_COEXISTENCE_MIN_LEVEL"] ?? "") ?? 60
    let free = (Int(nativeEnvironment["AETHER_VL_COEXISTENCE_MIN_FREE_MIB"] ?? "") ?? 4096) << 20
    var report = CoexistenceReport()
    report.bundles = [vlBundle.lastPathComponent, kreaRoot.deletingLastPathComponent().lastPathComponent]
    report.guardRule = "stop at the next step if memorystatus level < \(level) or free memory < \(free >> 20) MiB"
    report.rounds = Int(nativeEnvironment["AETHER_VL_COEXISTENCE_ROUNDS"] ?? "") ?? 3
    // Refuse to start on a host already near the guard.
    let sampler = ResourceSampler(abortBelowLevel: level, abortBelowFreeBytes: free)
    defer {
        sampler.stop()
        report.windows = sampler.windows
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    var describer: VisionLanguageSession?
    var krea: KreaGenerationSession?
    do {
        let manifest = try LanguageModelManifest.load(from: vlBundle)
        let registration = try #require(manifest.vision)
        let coordinator = try ResourceCoordinator(capacityBytes: 48 << 30)
        let admission = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let tokenizer = try await LanguageTokenizer.open(bundle: vlBundle)
        func peakReserved() { report.reservedPeakMiB = max(report.reservedPeakMiB, Int(coordinator.snapshot.reservedBytes >> 20)) }
        sampler.mark("before open")
        if let reason = sampler.tripped { report.abortedBecause = "before start: " + reason; return }
        describer = try await VisionLanguageSession.open(bundle: vlBundle, platform: .macos, gpu: admission.owner("describer"),
                                                          resources: try profile(for: manifest, coordinator: coordinator))
        peakReserved()
        sampler.mark("describer open", reserved: coordinator.snapshot.reservedBytes)
        let assets = try KreaAssets.open(at: kreaRoot)
        let kreaTokenizer = try await KreaTokenizer.open(assets: assets)
        krea = KreaGenerationSession(assets: assets, gpu: admission.owner("krea"))
        let kreaPrompt = try kreaTokenizer.prepare("a lighthouse on a cliff at dusk, gulls overhead, warm light in the lantern room")
        let rendered = try tokenizer.render([.user("Describe this picture in one line.", images: 1)]).tokenIDs
        let grid = try #require(registration.imageGrids?.first)
        let image = try gradient(width: grid.width * registration.patchSize, height: grid.height * registration.patchSize, seed: 3)
        for round in 1...report.rounds {
            if let reason = sampler.tripped { report.abortedBecause = reason; break }
            _ = try await describer?.generate(renderedTokenIDs: rendered, images: [image], maximumTokens: 32)
            peakReserved()
            sampler.mark("round \(round): describer image request", reserved: coordinator.snapshot.reservedBytes)
            if let reason = sampler.tripped { report.abortedBecause = reason; break }
            _ = try await krea?.generate(configuration: try KreaConfiguration(width: 1024, height: 1024, seed: UInt64(round)), prompt: kreaPrompt)
            sampler.mark("round \(round): Krea 1024x1024 render", reserved: coordinator.snapshot.reservedBytes)
            report.completedRounds = round
        }
        await krea?.unload(); krea = nil
        await describer?.unload(); describer = nil
        sampler.mark("unloaded", reserved: coordinator.snapshot.reservedBytes)
    } catch {
        report.error = String(describing: error)
        await krea?.unload()
        await describer?.unload()
        throw error
    }
}
