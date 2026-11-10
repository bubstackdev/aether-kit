import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing
@testable import AetherImage

/// Krea 2's residency controls with the fake graphs of ``Krea2Tests``: the retention choice, a release that never waits
/// for or interrupts a generation, the measured resident bytes, and the opt-in pressure gate.
@Suite("Krea 2 residency")
struct Krea2ResidencyTests {
    typealias Calls = Krea2Tests.Calls
    static let request = ImageRequest(prompt: "a cat", steps: 1, width: 32, height: 32)
    static let roles = ["denoiser0", "denoiser1", "denoiser2", "denoiser3", "textEncoder", "vaeDecoder"]

    /// A pipeline over the fakes with `options` (unadmitted GPU, a fixed probe unless given).
    static func pipeline(filter: any ContentFilter = Krea2Tests.RecordingFilter(), options: ImagePipelineOptions) throws -> Krea2Pipeline {
        try Krea2Pipeline(
            manifest: try Krea2Tests.manifest(), device: Krea2Tests.mac, filter: filter, options: options,
            encodePrompt: Krea2Tests.encode,
            load: { role throws(AetherError) in
                let calls = Calls()
                return switch role {
                case "textEncoder": Krea2Tests.FakeText(calls)
                case "denoiser0": Krea2Tests.FakeSegment(0, calls)
                case "denoiser1": Krea2Tests.FakeSegment(1, calls)
                case "denoiser2": Krea2Tests.FakeSegment(2, calls)
                case "denoiser3": Krea2Tests.FakeSegment(3, calls)
                default: Krea2Tests.FakeVAE(calls)
                }
            },
            fileBytes: { Krea2Tests.files[$0] ?? 0 })
    }

    static var options: ImagePipelineOptions { .init(memory: .bytes(1 << 40), gpu: .unadmitted, probe: Krea2Tests.still) }

    /// A filter that holds the prompt review until released, so a generation can be caught in progress.
    final class HoldingFilter: ContentFilter {
        let entered = AsyncStream<Void>.makeStream()
        let release = AsyncStream<Void>.makeStream()
        func review(prompt: String) async throws -> ContentVerdict {
            entered.continuation.yield()
            for await _ in release.stream { break }
            return .allow
        }
        func review(image: GeneratedImage, prompt: String) async throws -> ContentVerdict { .allow }
    }

    /// Whether a generation fails with ``AetherError/memoryPressure``.
    static func refusedForPressure(_ pipeline: Krea2Pipeline) async -> Bool {
        do { _ = try await pipeline.generate(request); return false }
        catch AetherError.memoryPressure { return true }
        catch { return false }
    }

    @Test func keepLoadedIsTheDefaultAndReleaseAfterEachGenerationStartsColdEveryTime() async throws {
        let pipeline = try Self.pipeline(options: Self.options)
        #expect(await pipeline.retention == .keepLoaded)
        _ = try await pipeline.generate(Self.request)
        #expect(await pipeline.residentRoles == Self.roles)
        await pipeline.setRetention(.releaseAfterEachGeneration)
        for _ in 0..<2 {
            _ = try await pipeline.generate(Self.request)
            #expect(await pipeline.residentRoles.isEmpty)
            #expect(await pipeline.plannedResidentBytes == 0)
        }
        // Cold every time: the last generation loaded all six.
        #expect(try await pipeline.generate(Self.request, observing: nil).metrics.loadedRoles == Self.roles)
        await pipeline.setRetention(.keepLoaded)
        _ = try await pipeline.generate(Self.request)
        #expect(try await pipeline.generate(Self.request, observing: nil).metrics.loadedRoles.isEmpty)
        await pipeline.unload()
    }

    @Test func releaseIfIdleNeverTouchesAGenerationInProgress() async throws {
        let filter = HoldingFilter()
        let pipeline = try Self.pipeline(filter: filter, options: Self.options)
        // Warm: the prompt review of the first generation is let through at once.
        filter.release.continuation.yield()
        _ = try await pipeline.generate(Self.request)
        #expect(await pipeline.residentRoles == Self.roles)
        // The second generation is held in its prompt review, so it is in progress.
        let running = Task { try await pipeline.generate(Self.request) }
        for await _ in filter.entered.stream { break }
        for await _ in filter.entered.stream { break }
        #expect(await pipeline.releaseIfIdle() == false)
        #expect(await pipeline.residentRoles == Self.roles, "a generation in progress keeps its graphs")
        filter.release.continuation.yield()
        _ = try await running.value
        #expect(await pipeline.releaseIfIdle() == true)
        #expect(await pipeline.residentRoles.isEmpty)
        #expect(await pipeline.plannedResidentBytes == 0)
        // Released, not unloaded: the pipeline generates again.
        filter.release.continuation.yield()
        _ = try await pipeline.generate(Self.request)
        await pipeline.unload()
    }

    @Test func residentBytesAreTheMeasuredHeldFootprint() async throws {
        // A probe that grows on every read, so each phase measures a positive held footprint.
        let reads = Mutex<Int64>(0)
        var options = Self.options
        options.probe = GPUMemoryProbe { reads.withLock { $0 += 1_000; return $0 } }
        let pipeline = try Self.pipeline(options: options)
        #expect(await pipeline.plannedResidentBytes == 0)
        _ = try await pipeline.generate(Self.request)
        #expect(await pipeline.plannedResidentBytes > 0)
        #expect(await pipeline.releaseIfIdle())
        #expect(await pipeline.plannedResidentBytes == 0)
    }

    @Test func criticalPressureRefusesLoadsButNotResidentGraphs() async throws {
        let monitor = MemoryPressureMonitor(source: .manual)
        var options = Self.options
        options.pressure = monitor
        // Cold under critical pressure: refused before anything loads.
        let cold = try Self.pipeline(options: options)
        await monitor.deliver(.critical)
        #expect(await Self.refusedForPressure(cold))
        #expect(await cold.residentRoles.isEmpty)
        // Warm before the notice: the resident graphs still run under it.
        await monitor.deliver(.normal)
        let warm = try Self.pipeline(options: options)
        _ = try await warm.generate(Self.request)
        await monitor.deliver(.critical)
        _ = try await warm.generate(Self.request)
        // Once released, the next generation needs loads again and is refused.
        #expect(await warm.releaseIfIdle())
        #expect(await Self.refusedForPressure(warm))
        await monitor.deliver(.normal)
        _ = try await warm.generate(Self.request)
        await warm.unload()
    }

    @Test func withoutAMonitorPressureIsNotChecked() async throws {
        let pipeline = try Self.pipeline(options: Self.options)
        #expect(Self.options.pressure == nil)
        _ = try await pipeline.generate(Self.request)
        await pipeline.unload()
    }

    /// A generation whose admission released the graphs (here a fifth row count past `maximumShapes`) reports every
    /// role as loaded. Before: the resident roles were read before admission planned, so it reported none.
    @Test func aReleaseByAdmissionCountsAsLoading() async throws {
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: LatentDiffusionTests.Calls())
        var last: ImageGenerationMetrics?
        for prompt in ["a", "ab", "abc", "abcd", "abcde"] {
            last = try await pipeline.generate(
                ImageRequest(prompt: prompt, seed: 1, steps: 1, width: 32, height: 32), observing: nil).metrics
        }
        #expect(loads.log.count == 12)
        #expect(last?.loadedRoles == ["denoiser0", "denoiser1", "denoiser2", "denoiser3", "textEncoder", "vaeDecoder"])
    }
}
