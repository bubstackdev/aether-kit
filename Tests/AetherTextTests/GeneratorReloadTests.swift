@_spi(Testing) import AetherHost
import Synchronization
import Testing
@testable import AetherRuntime
@testable import AetherText
import AetherTestSupport

/// The Generator's side of the reload budget (docs/VLM.md, V6): it checks the budget at the end of every request,
/// before the request's stream ends, and a reload changes no reply and keeps the next turn's prefix reuse.
@Suite("Generator reload budget (fake)")
struct GeneratorReloadTests {
    @Test func aRequestPastTheBudgetReloadsBeforeItsStreamEnds() async throws {
        let level = Mutex<Int64>(1 << 30)
        let loads = Mutex(0)
        let fake = FakeLM(FakeLM.Config(predict: FakeChat.scripted(FakeChat.echo)))
        var options = DecoderEngine.Options(memoryBudget: .bytes(1 << 30))
        options.reloadBudgetBytes = 1 << 20
        let engine = try await DecoderEngine(
            step: fake, graph: fake.manifestGraph(), device: FakeChat.device, options: options,
            reloader: { () async throws(AetherError) -> (step: any GraphBackend, prefill: (any GraphBackend)?) in
                loads.withLock { $0 += 1 }
                return (FakeLM(FakeLM.Config(predict: FakeChat.scripted(FakeChat.echo))), nil)
            },
            levelProbe: { level.withLock { $0 } })
        let generator = Generator(engine: engine, decoder: FakeDecoding(), renderer: FakeRenderer(), stopTokens: [Vocab.stop],
                                  profile: .plain, gpu: .unadmitted)
        let chat = ChatSession(generator: generator, configuration: .init(parameters: GenerationParameters(maxTokens: 32)))
        _ = try await FakeChat.collect(await chat.respond(to: "hello there"))
        #expect(await engine.reloads.count == 0)  // the first request sets the reference level
        level.withLock { $0 += 2 << 20 }
        let second = try await FakeChat.collect(await chat.respond(to: "and now"))
        #expect(await engine.reloads.count == 1 && loads.withLock { $0 } == 1)
        let third = try await FakeChat.collect(await chat.respond(to: "once more"))
        // The prefix survives the reload: the third turn reuses everything the engine held.
        let two = try #require(second.summary), three = try #require(third.summary)
        #expect(three.reusedTokens == two.promptTokens + two.tokens.count - 1)

        let fresh = try await FakeChat.generator()
        let replay = try await FakeChat.collect(fresh.generate(Array(await chat.history.prefix(5))))
        #expect(replay.text == third.text && !third.text.isEmpty)
    }
}
