import Foundation
import Metal
import Testing
@_spi(Diagnostics) @testable import AetherRuntime

extension GateTests {
    /// `Options.canaryRecords`: a pass is recorded, and the next engine of the same graph starts at depth 3 without
    /// running the canary; a diagnostic hand-off never trusts the record (the negative control still fails).
    @Test func aRecordedCanaryPassSkipsTheNextCanary() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "aether-canary-records-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        var options = DecoderEngine.Options(capacity: 4096)
        options.canaryAtLoad = true
        options.canaryRecords = folder
        let first = try await Self.engine("qwen3-0.6b", options)
        let ran = try #require(await first.canary)
        #expect(ran.passed && !ran.recorded && ran.seconds > 0)
        #expect(try FileManager.default.contentsOfDirectory(atPath: folder.path).count == 1)
        options.canaryAtLoad = false
        let second = try await Self.engine("qwen3-0.6b", options)
        let recorded = try #require(await second.canary)
        #expect(recorded.passed && recorded.recorded && recorded.seconds == 0)
        #expect(await second.decodeMode == .gpu(depth: 3))
        options.canaryAtLoad = true
        options.staleTokenHandoff = true
        let stale = try await Self.engine("qwen3-0.6b", options)
        #expect(await stale.canary?.passed == false)
        #expect(await stale.decodeMode == .gpu(depth: 1))
    }
}
