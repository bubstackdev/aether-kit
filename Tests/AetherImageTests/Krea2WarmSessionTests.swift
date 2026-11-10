import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing
@testable import AetherImage

/// A host that keeps a Krea 2 session warm between pictures (encode, then generate from the conditioning, picture after
/// picture) on the fakes of ``Krea2Tests``: the canvas-aware encode keeps the text encoder resident beside the canvas's
/// denoisers and decoder, and a cancelled generation keeps the graphs it had already measured.
@Suite("Krea 2 warm session")
struct Krea2WarmSessionTests {
    typealias Calls = LatentDiffusionTests.Calls
    static let roles = ["denoiser0", "denoiser1", "denoiser2", "denoiser3", "textEncoder", "vaeDecoder"]
    static func request(_ prompt: String = "a cat", seed: UInt64 = 7) -> ImageRequest {
        ImageRequest(prompt: prompt, seed: seed, steps: 2, width: 48, height: 32)
    }

    struct Boom: Error {}

    /// A progress handler that throws `error` at the first step.
    static func throwing(_ error: any Error) -> @Sendable (ImageProgress) throws -> Void {
        { event in if case .step = event { throw error } }
    }

    // MARK: - The canvas-aware encode

    /// Before: the first encode on a pipeline planned under a conditioning-only profile, so the generation that followed
    /// (another profile) released the text encoder, and the next picture's encode loaded it again.
    @Test func anEncodeForTheCanvasKeepsTheTextEncoderResident() async throws {
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: Calls())
        let first = try await pipeline.encode(prompt: "a cat", width: 48, height: 32)
        let generation = try await pipeline.generate(Self.request(), reusing: first)
        #expect(generation.metrics.loadedRoles == ["denoiser0", "denoiser1", "denoiser2", "denoiser3", "vaeDecoder"])
        #expect(await pipeline.residentRoles == Self.roles)
        let loaded = loads.log.count
        // The next picture: a new prompt, encoded and rendered with nothing loaded again.
        let second = try await pipeline.encode(prompt: "a dog", width: 48, height: 32)
        #expect(try await pipeline.generate(Self.request("a dog"), reusing: second).metrics.loadedRoles.isEmpty)
        #expect(loads.log.count == loaded && loads.log.filter { $0 == "textEncoder" }.count == 1)
    }

    /// The unchanged ``Krea2Pipeline/encode(prompt:job:)``: before any generation it plans a conditioning-only profile,
    /// whose text encoder the first generation's canvas releases (and the second picture's encode loads again).
    @Test func anEncodeWithoutTheCanvasStillPlansAConditioningProfile() async throws {
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: Calls())
        let first = try await pipeline.encode(prompt: "a cat")
        _ = try await pipeline.generate(Self.request(), reusing: first)
        #expect(await !pipeline.residentRoles.contains("textEncoder"))
        let second = try await pipeline.encode(prompt: "a dog")
        _ = try await pipeline.generate(Self.request("a dog"), reusing: second)
        #expect(loads.log.filter { $0 == "textEncoder" }.count == 2)
        #expect(await pipeline.residentRoles == Self.roles)
    }

    @Test func anEncodeForAnotherCanvasStillKeepsWhatThatCanvasLoads() async throws {
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: Calls())
        _ = try await pipeline.generate(Self.request())
        let conditioning = try await pipeline.encode(prompt: "a dog", width: 32, height: 32)
        let small = ImageRequest(prompt: "a dog", seed: 7, steps: 2, width: 32, height: 32)
        // The canvas changed at the encode: the 48x32 graphs went then; the 32x32 generation loads only its denoisers and
        // decoder, and the text encoder the encode loaded stays.
        #expect(try await pipeline.generate(small, reusing: conditioning).metrics.loadedRoles
            == ["denoiser0", "denoiser1", "denoiser2", "denoiser3", "vaeDecoder"])
        #expect(await pipeline.residentRoles == Self.roles)
        #expect(loads.log.filter { $0 == "textEncoder" }.count == 2)
    }

    @Test func aCanvasTheExportCantRenderIsRefusedBeforeAnythingLoads() async throws {
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: Calls())
        #expect(await Krea2ConditioningTests.isInvalidParameters { _ = try await pipeline.encode(prompt: "a cat", width: 33, height: 32) })
        #expect(await Krea2ConditioningTests.isInvalidParameters { _ = try await pipeline.encode(prompt: "a cat", width: 0, height: 32) })
        #expect(loads.log.isEmpty)
    }

    // MARK: - Cancellation

    @Test func aCancelledWarmGenerationKeepsItsGraphs() async throws {
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: Calls())
        _ = try await pipeline.generate(Self.request())
        let planned = await pipeline.plannedResidentBytes
        let loaded = loads.log.count
        for error: any Error in [CancellationError(), AetherError.runtime(underlying: CancellationError())] {
            await #expect(throws: (any Error).self) {
                _ = try await pipeline.generate(Self.request("a dog"), observing: Self.throwing(error))
            }
            #expect(await pipeline.residentRoles == Self.roles)
            #expect(await pipeline.plannedResidentBytes == planned)
        }
        // The next picture starts warm: nothing loads.
        #expect(try await pipeline.generate(Self.request("a cow"), observing: nil).metrics.loadedRoles.isEmpty)
        #expect(loads.log.count == loaded)
    }

    /// The task itself cancelled between phases (after the conditioning, before the steps): the graphs stay too.
    @Test func aCancelledTaskKeepsTheGraphs() async throws {
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: Calls())
        _ = try await pipeline.generate(Self.request())
        let loaded = loads.log.count
        let task = Task {
            try await pipeline.generate(Self.request("a dog")) { event in
                if case .conditioningReady = event { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(await pipeline.residentRoles == Self.roles)
        #expect(try await pipeline.generate(Self.request("a cow"), observing: nil).metrics.loadedRoles.isEmpty)
        #expect(loads.log.count == loaded)
    }

    /// A cancellation while the generation was still loading graphs releases them all: the phase that loaded them was
    /// cut short and never measured what they hold.
    @Test func aCancelledColdGenerationReleasesEveryGraph() async throws {
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls())
        await #expect(throws: CancellationError.self) {
            _ = try await pipeline.generate(Self.request(), observing: Self.throwing(CancellationError()))
        }
        #expect(await pipeline.residentRoles.isEmpty)
        #expect(await pipeline.plannedResidentBytes == 0)
    }

    @Test func aFailureStillReleasesEveryGraph() async throws {
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls())
        _ = try await pipeline.generate(Self.request())
        await #expect(throws: Boom.self) {
            _ = try await pipeline.generate(Self.request("a dog"), observing: Self.throwing(Boom()))
        }
        #expect(await pipeline.residentRoles.isEmpty)
        #expect(await pipeline.plannedResidentBytes == 0)
    }

    @Test func releasingAfterEachGenerationReleasesOnCancellationToo() async throws {
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls())
        _ = try await pipeline.generate(Self.request())
        await pipeline.setRetention(.releaseAfterEachGeneration)
        await #expect(throws: CancellationError.self) {
            _ = try await pipeline.generate(Self.request("a dog"), observing: Self.throwing(CancellationError()))
        }
        #expect(await pipeline.residentRoles.isEmpty)
    }

    @Test func cancellationIsRecognisedBareOrWrapped() {
        #expect(Krea2Pipeline.isCancellation(CancellationError()))
        #expect(Krea2Pipeline.isCancellation(AetherError.runtime(underlying: CancellationError())))
        #expect(!Krea2Pipeline.isCancellation(Boom()))
        #expect(!Krea2Pipeline.isCancellation(AetherError.runtime(underlying: Boom())))
        #expect(!Krea2Pipeline.isCancellation(AetherError.memoryPressure))
    }
}
