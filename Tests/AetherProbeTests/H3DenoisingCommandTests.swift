import AetherKit
import Foundation
import Testing
@testable import AetherProbe

@Test func h3DenoisingQualificationRequiresExplicitModeBeforeCreatingOutput() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    for mode in ["--unknown", "run", "", "--lifecycle", "--run"] {
        await #expect(throws: AetherError.self) {
            try await H3DenoisingCommand.run(["qualify-h3-denoising", "/missing", "/missing", "/missing", "/missing", "/missing", "-", root.path, mode])
        }
    }
    #expect(!FileManager.default.fileExists(atPath: root.path))
    #expect(try await H3DenoisingCommand.run(["other"]) == false)
}

@Test func h3DenoisingQualificationPreparationFailureCannotLaunchNativeOrClaimReady() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    await #expect(throws: (any Error).self) {
        try await H3DenoisingCommand.run(["qualify-h3-denoising", "/missing", "/missing", "/missing", "/missing", "/missing", "-", root.path, "--prepare-only"])
    }
    let data = try Data(contentsOf: root.appendingPathComponent("report.json"))
    let report = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(report["passed"] as? Bool == false)
    #expect(report["nativeExecuted"] as? Bool == false)
    #expect(report["stage"] as? String == "failed")
    await #expect(throws: AetherError.self) {
        try await H3DenoisingCommand.run(["qualify-h3-denoising", "/missing", "/missing", "/missing", "/missing", "/missing", "/missing-references", root.path, "--run"])
    }
    #expect(try Data(contentsOf: root.appendingPathComponent("report.json")) == data)
}
