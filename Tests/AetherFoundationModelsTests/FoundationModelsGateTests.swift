import Foundation
import FoundationModels
import Synchronization
import Testing
@testable import AetherFoundationModels

/// M3's gate (DESIGN.md §14), AETHER_GATES=1, on qwen3-0.6b from lab/models: a real `LanguageModelSession` over
/// ``AetherLanguageModel``, greedy and thinking off (no reasoning level):
/// 1. two-turn recall (the second turn answers from the first, by prefix reuse);
/// 2. one `@Generable` guided output, constrained and decoded by FoundationModels;
/// 3. one Hermes tool call round trip (the model calls the tool, FoundationModels runs it, the model answers from it).
@Suite("FoundationModelsGateTests", .serialized, .enabled(if: ProcessInfo.processInfo.environment["AETHER_GATES"] == "1"))
struct FoundationModelsGateTests {
    static var models: URL {
        if let override = ProcessInfo.processInfo.environment["AETHER_LAB_MODELS"] { return URL(fileURLWithPath: override) }
        return labModels(from: #filePath)
    }

    static let mac = Device(platform: .macOS, architecture: "h17c")

    static func model() async throws -> AetherLanguageModel {
        let id = "qwen3-0.6b"
        let manifest = try Manifest.decode(Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appending(path: "AetherCoreTests/Fixtures/zoo/\(id)/aether.json")))
        let bundle = InstalledBundle(ref: try ModelRef.resolve(id, for: mac), url: models.appending(path: id), isStale: false)
        return try await AetherLanguageModel.load(bundle: bundle, manifest: manifest, device: mac, engine: DecoderEngine.Options(capacity: 4096))
    }

    @Generable struct Capital {
        @Guide(description: "The country") var country: String
        @Guide(description: "Its capital city") var capital: String
    }

    struct Weather: Tool {
        final class Calls: Sendable {
            let cities = Mutex<[String]>([])
        }
        let calls = Calls()
        let name = "get_weather"
        let description = "Get the current weather in a city."
        @Generable struct Arguments {
            @Guide(description: "The city name") var city: String
        }
        func call(arguments: Arguments) async throws -> String {
            calls.cities.withLock { $0.append(arguments.city) }
            return "Sunny, 21 degrees Celsius."
        }
    }

    @Test func languageModelSessionOnQwen3() async throws {
        let model = try await Self.model()
        #expect(model.capabilities.contains(.toolCalling) && model.capabilities.contains(.reasoning))
        let options = GenerationOptions(maximumResponseTokens: 96)

        // 1. Two-turn recall.
        let chat = LanguageModelSession(model: model, instructions: "You are a helpful assistant. Answer briefly.")
        let first = try await chat.respond(to: "My name is Mara and my favourite colour is teal. Just say OK.", options: options)
        let second = try await chat.respond(to: "What is my favourite colour? Answer with one word.", options: options)
        print("GATE fm recall: 1: \(first.content.prefix(120).debugDescription)  2: \(second.content.prefix(120).debugDescription)")
        #expect(second.content.lowercased().contains("teal"))

        // 2. @Generable guided output.
        let guided = try await LanguageModelSession(model: model)
            .respond(to: "What is the capital of France?", generating: Capital.self, options: GenerationOptions(maximumResponseTokens: 64))
        print("GATE fm guided: \(guided.content.country) / \(guided.content.capital) \(guided.rawContent.jsonString)")
        #expect(guided.content.capital.contains("Paris"))

        // 3. A Hermes tool call round trip.
        let weather = Weather()
        let tools = LanguageModelSession(model: model, tools: [weather], instructions: "You are a helpful assistant. Use the tools when they help.")
        let answer = try await tools.respond(to: "What's the weather like in Paris right now?", options: GenerationOptions(maximumResponseTokens: 128))
        let cities = weather.calls.cities.withLock { $0 }
        print("GATE fm tool: calls \(cities); reply \(answer.content.prefix(160).debugDescription)")
        #expect(cities.count == 1 && cities.first?.contains("Paris") == true)
        #expect(answer.content.contains("21") || answer.content.lowercased().contains("sunny"))
    }
}

/// `lab/models` beside the repo: the first found walking up from `file`, so the gates find it from a worktree under
/// `.worktrees/` as from the main checkout (`AETHER_LAB_MODELS` overrides, at the call sites).
func labModels(from file: String) -> URL {
    var directory = URL(fileURLWithPath: file).deletingLastPathComponent()
    while directory.path != "/" {
        let candidate = directory.appending(path: "lab/models")
        if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        directory = directory.deletingLastPathComponent()
    }
    return URL(fileURLWithPath: file).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appending(path: "lab/models")
}
