import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import Foundation
import Testing
@testable import AetherImage

/// `Krea2Pipeline.prepare(prompt:)`: the prompt as `generate` renders it, on the CPU, with the shrunk manifest of
/// ``Krea2Tests`` (2-row prefix, at most 10 tokens, the last 2 kept). Rendered ids are 100, 101, the scalars, 200.
@Suite("Krea 2 prompt preparation")
struct Krea2PromptPreparationTests {
    typealias Calls = Krea2Tests.Calls

    @Test func aShortPromptIsRenderedWhole() throws {
        let filter = Krea2Tests.RecordingFilter()
        let (pipeline, loads) = try Krea2Tests.pipeline(calls: Calls(), filter: filter)
        let prepared = try pipeline.prepare(prompt: "a cat")
        #expect(prepared.tokenIDs == [100, 101] + "a cat".unicodeScalars.map { Int32($0.value) } + [200])
        #expect(prepared.renderedTokenCount == 8 && prepared.conditioningRows == 6 && !prepared.truncated)
        #expect(prepared.maximumTokens == 10)
        #expect(loads.log.isEmpty, "preparation loads no graph")
        #expect(filter.log.isEmpty, "preparation doesn't consult the filter")
    }

    @Test func aLongPromptReportsItsCutAndKeepsTheTail() throws {
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls())
        // 2 + 12 + 1 = 15 rendered ids, at most 10: the first 8 and the last 2.
        let text = "abcdefghijkl"
        let rendered: [Int32] = [100, 101] + text.unicodeScalars.map { Int32($0.value) } + [200]
        let prepared = try pipeline.prepare(prompt: text)
        #expect(prepared.truncated && prepared.renderedTokenCount == 15)
        #expect(prepared.tokenIDs == Array(rendered.prefix(8)) + rendered.suffix(2))
        #expect(prepared.conditioningRows == 8)
    }

    @Test func theRowCountIsTheOneGenerationReports() async throws {
        let (pipeline, _) = try Krea2Tests.pipeline(calls: Calls())
        let request = ImageRequest(prompt: "a cat", steps: 1, width: 32, height: 32)
        let metrics = try await pipeline.generate(request, observing: nil).metrics
        #expect(metrics.conditioningRows == (try pipeline.prepare(prompt: request.prompt)).conditioningRows)
        await pipeline.unload()
    }

    @Test func aPromptGenerationRefusesIsRefusedHereToo() async throws {
        // Every rendering has its two prefix ids, so the fake can't produce too few rows; a negative id is refused.
        let manifest = try Krea2Tests.manifest()
        let pipeline = try Krea2Pipeline(
            manifest: manifest, device: Krea2Tests.mac, filter: Krea2Tests.RecordingFilter(),
            options: .init(gpu: .unadmitted, probe: Krea2Tests.still), encodePrompt: { _ in [100, 101, -1] },
            load: { _ throws(AetherError) in Krea2Tests.FakeVAE(Calls()) }, fileBytes: { _ in 0 })
        #expect(throws: AetherError.self) { try pipeline.prepare(prompt: "x") }
        await #expect(throws: AetherError.self) {
            try await pipeline.generate(.init(prompt: "x", steps: 1, width: 32, height: 32))
        }
    }
}
