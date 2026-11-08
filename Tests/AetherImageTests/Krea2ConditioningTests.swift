import AetherCore
import AetherHost
import Foundation
import Synchronization
import Testing
@testable import AetherImage
@testable import AetherRuntime

/// Conditioning reuse (`encode(prompt:job:)` and `generate(_:reusing:observing:)`) on the Krea 2 fakes: the gate is
/// pixels identical to encode-and-generate for the same seed, with the text encoder run once.
@Suite("Krea 2 conditioning reuse")
struct Krea2ConditioningTests {
    typealias Calls = LatentDiffusionTests.Calls
    static func request(seed: UInt64, width: Int = 48) -> ImageRequest {
        ImageRequest(prompt: "a cat", seed: seed, steps: 2, width: width, height: 32)
    }

    @Test func reuseRendersTheEncodedImageAtEverySeedAndSize() async throws {
        let (fresh, _) = try Krea2Tests.pipeline(calls: Calls())
        var expected: [GeneratedImage] = []
        for request in [Self.request(seed: 7), Self.request(seed: 9), Self.request(seed: 7, width: 32)] {
            expected.append(try await fresh.generate(request))
        }

        let calls = Calls()
        let (pipeline, _) = try Krea2Tests.pipeline(calls: calls)
        let conditioning = try await pipeline.encode(prompt: "a cat")
        var reused: [GeneratedImage] = []
        for request in [Self.request(seed: 7), Self.request(seed: 9), Self.request(seed: 7, width: 32)] {
            reused.append(try await pipeline.generate(request, reusing: conditioning).image)
        }
        #expect(reused == expected)
        #expect(calls.log.filter { $0 == "text" }.count == 1)
    }

    @Test func theConditioningDescribesThePrompt() async throws {
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls())
        let conditioning = try await pipeline.encode(prompt: "a cat")
        let prepared = try pipeline.prepare(prompt: "a cat")
        #expect(conditioning.prompt == "a cat" && conditioning.rows == prepared.conditioningRows)
        #expect(conditioning.columns == Krea2Tests.columns && !conditioning.truncated)
        // The fake text encoder writes fp16.
        #expect(conditioning.byteCount == conditioning.rows * conditioning.columns * 2)
    }

    @Test func aReuseNeverLoadsTheTextEncoder() async throws {
        let calls = Calls()
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: calls)
        await pipeline.setRetention(.releaseAfterEachGeneration)
        let conditioning = try await pipeline.encode(prompt: "a cat")
        #expect(loads.log == ["textEncoder"] && calls.log == ["text"])
        #expect(await pipeline.residentRoles.isEmpty)

        let progress = Mutex<[String]>([])
        let generation = try await pipeline.generate(Self.request(seed: 7), reusing: conditioning) { event in
            if case .step = event { return }
            progress.withLock { $0.append("\(event)") }
        }
        #expect(!loads.log.dropFirst().contains("textEncoder") && calls.log.filter { $0 == "text" }.count == 1)
        #expect(progress.withLock { $0 } == ["conditioningReady(rows: \(conditioning.rows))", "denoisersReady", "decoded"])
        #expect(generation.metrics.conditioningRows == conditioning.rows)
        #expect(generation.metrics.loadedRoles == ["denoiser0", "denoiser1", "denoiser2", "denoiser3", "vaeDecoder"])
    }

    @Test func encodingKeepsTheLastCanvasDenoisersResident() async throws {
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: Calls())
        _ = try await pipeline.generate(Self.request(seed: 7))
        let loaded = loads.log.count
        let conditioning = try await pipeline.encode(prompt: "a dog")
        _ = try await pipeline.generate(ImageRequest(prompt: "a dog", seed: 7, steps: 2, width: 48, height: 32), reusing: conditioning)
        // Everything stayed resident across the encode: nothing loaded again.
        #expect(loads.log.count == loaded)
    }

    @Test func theFilterReviewsThePromptAtEncodeAndAtEachGeneration() async throws {
        let filter = Krea2Tests.RecordingFilter()
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls(), filter: filter)
        let conditioning = try await pipeline.encode(prompt: "a cat")
        _ = try await pipeline.generate(Self.request(seed: 7), reusing: conditioning)
        #expect(filter.log == ["prompt a cat", "prompt a cat", "image 48x32 a cat"])

        let calls = Calls()
        let (strict, loads) = try Krea2Tests.pipeline(calls: calls, filter: Krea2Tests.RecordingFilter(refusePrompt: true))
        await #expect(throws: AetherError.self) { _ = try await strict.encode(prompt: "a cat") }
        #expect(loads.log.isEmpty && calls.log.isEmpty)
    }

    @Test func anotherExportsOrPromptsConditioningIsRefused() async throws {
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls())
        let conditioning = try await pipeline.encode(prompt: "a cat")
        var other = try Krea2Tests.manifest()
        other.license = "another export"
        let (elsewhere, loads) = try Krea2Tests.pipeline(calls: Calls(), manifest: other)
        #expect(await Self.isInvalidParameters { _ = try await elsewhere.generate(Self.request(seed: 7), reusing: conditioning) })
        #expect(loads.log.isEmpty)
        let dog = ImageRequest(prompt: "a dog", seed: 7, steps: 2, width: 48, height: 32)
        #expect(await Self.isInvalidParameters { _ = try await pipeline.generate(dog, reusing: conditioning) })
    }

    static func isInvalidParameters(_ body: () async throws -> Void) async -> Bool {
        do {
            try await body()
            return false
        } catch AetherError.generation(.invalidParameters) {
            return true
        } catch {
            return false
        }
    }
}
