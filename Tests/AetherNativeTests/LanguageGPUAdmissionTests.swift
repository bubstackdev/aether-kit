import AetherKit
import AetherLanguageCoreAI
import Foundation
import Synchronization
import Testing

/// Opt-in native check of `LanguageGenerationSession`'s `gpu:` path on a real model: a request is
/// admitted for exactly its native work, and `unload()` withdraws a request still waiting for
/// admission, so close never waits behind the GPU queue.
///
/// AETHER_NATIVE_LANGUAGE_GPU_ADMISSION=1
/// AETHER_LANGUAGE_BUCKETED_MODEL=<bucketed model.aimodel with export-report.json beside it>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_GPU_ADMISSION"] == "1"), .timeLimit(.minutes(10)))
func languageSessionGPUAdmissionAdmitsAndWithdraws() async throws {
    let process = try await nativeGPUGrant(); defer { process?.release() }
    let model = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_BUCKETED_MODEL"]))
    struct Report: Decodable { let exportBounds: LanguageExportBounds? }
    let bounds = try JSONDecoder().decode(Report.self, from: Data(contentsOf:
        model.deletingLastPathComponent().appendingPathComponent("export-report.json"))).exportBounds
    // An in-process admission local to this test (the process lease above serializes native tests).
    let admission = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
    let prompt = [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]       // a short chat-shaped prompt

    // 1. Admitted for the request's native work only.
    let session = try await LanguageGenerationSession.open(at: model, maximumContextLength: 512, stopTokenIDs: [151935],
                                                           exportBounds: bounds, gpu: admission.owner("language"))
    #expect(admission.snapshot.running == 0)
    let observed = Mutex<[Int]>([])
    let reply = try await session.generate(prompt: prompt, maximumTokens: 8, onToken: { _ in
        observed.withLock { $0.append(admission.snapshot.running) }
    })
    #expect(reply.tokens.count == 8)
    #expect(observed.withLock { $0 }.allSatisfy { $0 == 1 })                 // held throughout decoding
    #expect(admission.snapshot.running == 0 && admission.snapshot.queued == 0)   // released at the end

    // 2. A request queued behind another holder is withdrawn by unload, which does not wait for the GPU.
    let blocker = try await admission.admit(owner: "blocker")
    let waiting = Task { try await session.generate(prompt: prompt, maximumTokens: 8) }
    var queued = false
    for _ in 0..<500 where !queued {
        queued = admission.snapshot.queued == 1
        if !queued { try await Task.sleep(for: .milliseconds(10)) }
    }
    #expect(queued)
    let start = ContinuousClock.now
    await session.unload()
    let closeSeconds = start.duration(to: .now)
    await #expect(throws: LanguageGenerationError.closed) { try await waiting.value }
    #expect(closeSeconds < .seconds(5))
    #expect(admission.snapshot.queued == 0 && admission.snapshot.running == 1)   // only the blocker remains
    #expect(await session.state == .closed)
    blocker.release()
    #expect(admission.snapshot.running == 0)
    await #expect(throws: LanguageGenerationError.closed) { try await session.generate(prompt: prompt, maximumTokens: 1) }
}
