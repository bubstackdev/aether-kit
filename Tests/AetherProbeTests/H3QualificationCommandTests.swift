import AetherKit
import Foundation
import Testing
@testable import AetherProbe

@Test func h3QualificationRequiresExplicitModeBeforeCreatingOutput() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    for mode in ["--unknown", "run", ""] {
        await #expect(throws: AetherError.self) {
            try await H3QualificationCommand.run(["qualify-h3", "/missing", "/missing", root.path, mode])
        }
    }
    #expect(!FileManager.default.fileExists(atPath: root.path))
    #expect(try await H3QualificationCommand.run(["other"]) == false)
}

@Test func h3QualificationPreparationFailureCannotLaunchNativeOrClaimReady() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    await #expect(throws: (any Error).self) {
        try await H3QualificationCommand.run(["qualify-h3", "/missing", "/missing", root.path, "--prepare-only"])
    }
    let data = try Data(contentsOf: root.appendingPathComponent("report.json"))
    let report = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(report["passed"] as? Bool == false)
    #expect(report["nativeExecuted"] as? Bool == false)
    #expect(report["stage"] as? String == "failed")
    await #expect(throws: AetherError.self) {
        try await H3QualificationCommand.run(["qualify-h3", "/missing", "/missing", root.path, "--run"])
    }
    #expect(try Data(contentsOf: root.appendingPathComponent("report.json")) == data)
}
