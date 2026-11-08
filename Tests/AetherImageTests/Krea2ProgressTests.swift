import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing
@testable import AetherImage

/// Krea 2's progress and metrics with the fake graphs of ``Krea2Tests``: the events in
/// order with each step's latents, a throwing handler stopping the generation, and the metrics' shape.
@Suite("Krea 2 progress")
struct Krea2ProgressTests {
    typealias Calls = Krea2Tests.Calls

    final class Events: Sendable {
        let values = Mutex<[ImageProgress]>([])
        var snapshot: [ImageProgress] { values.withLock { $0 } }
        func append(_ event: ImageProgress) { values.withLock { $0.append(event) } }
    }

    struct Refusal: Error {}

    /// "a cat" renders 8 ids (2 prefix + 5 + 1): 6 conditioning rows. 48×32 is a 2×3 token grid.
    static let request = ImageRequest(prompt: "a cat", seed: 7, steps: 2, width: 48, height: 32)

    @Test func progressReportsEachPhaseInOrderWithEachStepsLatents() async throws {
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls())
        let events = Events()
        let generation = try await pipeline.generate(Self.request, observing: { events.append($0) })
        let log = events.snapshot
        #expect(log.count == 5)
        #expect(log.first == .conditioningReady(rows: 6))
        #expect(log[1] == .denoisersReady)
        #expect(log.last == .decoded)
        let final = try Krea2Tests.expected(prompt: "a cat", seed: 7, steps: 2, tokens: 6)
        for (index, event) in log[2...3].enumerated() {
            guard case .step(let step, let total, let latents) = event else { Issue.record("expected a step, got \(event)"); continue }
            #expect(step == index && total == 2)
            #expect(latents.rows == 2 && latents.columns == 3 && latents.channels == 64 && latents.values.count == 6 * 64)
            if index == 1 { #expect(latents.values == final) }
        }
        // The observed generation is the ordinary one.
        let (plain, _) = try Krea2Tests.pipeline(calls: Calls())
        #expect(try await plain.generate(Self.request) == generation.image)
        await pipeline.unload()
        await plain.unload()
    }

    @Test func metricsRecordShapesStepsAndWhatHadToLoad() async throws {
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls())
        let first = try await pipeline.generate(Self.request, observing: nil).metrics
        #expect(first.conditioningRows == 6 && first.imageTokens == 6)
        #expect(first.stepSeconds.count == 2)
        #expect(first.stepSeconds.allSatisfy { $0 >= 0 })
        #expect(abs(first.samplingSeconds - first.stepSeconds.reduce(0, +)) < 1e-9)
        #expect(first.loadedRoles == ["denoiser0", "denoiser1", "denoiser2", "denoiser3", "textEncoder", "vaeDecoder"])
        let parts = first.waitSeconds + first.promptReviewSeconds + first.conditioningSeconds + first.denoiserPreparationSeconds
            + first.samplingSeconds + first.decodeSeconds + first.imageReviewSeconds
        #expect(parts <= first.totalSeconds + 1e-9)
        // Everything fit the budget and stayed resident: the second generation loads nothing.
        let second = try await pipeline.generate(Self.request, observing: nil).metrics
        #expect(second.loadedRoles.isEmpty)
        await pipeline.unload()
    }

    @Test func perPhaseResidencyReportsTheGraphsItReloads() async throws {
        // Files 10 / 80 / 10 against 95: each phase's graphs are released, so every generation loads all six again.
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls(), budget: 95)
        for _ in 0..<2 {
            let metrics = try await pipeline.generate(.init(prompt: "a cat", steps: 1, width: 32, height: 32), observing: nil).metrics
            #expect(metrics.loadedRoles.count == 6)
        }
    }

    @Test func aThrowingHandlerStopsBeforeTheNextGraphCallAndReleases() async throws {
        let calls = Calls(), filter = Krea2Tests.RecordingFilter()
        let (pipeline, _) = try Krea2Tests.pipeline(calls: calls, filter: filter)
        await #expect(throws: Refusal.self) {
            try await pipeline.generate(Self.request, observing: { event in
                if event == .denoisersReady { throw Refusal() }
            })
        }
        #expect(calls.log == ["text"], "no segment runs after the refusal")
        #expect(filter.log == ["prompt a cat"], "a stopped generation has no image to review")
        #expect(await pipeline.residentRoles.isEmpty)
        // The pipeline still works afterwards.
        _ = try await pipeline.generate(Self.request)
        await pipeline.unload()
    }

    @Test func aHandlerCanStopAfterAStep() async throws {
        let calls = Calls()
        let (pipeline, _) = try Krea2Tests.pipeline(calls: calls)
        await #expect(throws: Refusal.self) {
            try await pipeline.generate(Self.request, observing: { event in
                if case .step(0, _, _) = event { throw Refusal() }
            })
        }
        #expect(calls.log.filter { $0.hasPrefix("segment0") }.count == 1)
        #expect(!calls.log.contains("vae"))
        #expect(await pipeline.residentRoles.isEmpty)
    }
}
