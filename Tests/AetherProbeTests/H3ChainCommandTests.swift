import AetherKit
import AetherH3CoreAI
import Foundation
import Testing
@testable import AetherProbe

@Test func h3ChainModesAreExplicitAndFailedPreparationCannotLaunchNative() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    await #expect(throws: AetherError.self) {
        try await H3ChainCommand.run(["qualify-h3-chain", "/missing", "/missing", "/missing", root.path, "--unknown"])
    }
    #expect(!FileManager.default.fileExists(atPath: root.path))
    #expect(try await H3ChainCommand.run(["other"]) == false)
    await #expect(throws: (any Error).self) {
        try await H3ChainCommand.run(["qualify-h3-chain", "/missing", "/missing", "/missing", root.path, "--prepare-only"])
    }
    let data = try Data(contentsOf: root.appendingPathComponent("report.json"))
    let raw = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(raw["nativeExecuted"] as? Bool == false)
    #expect(raw["passed"] as? Bool == false)
    #expect(raw["captureComplete"] as? Bool == false)
    await #expect(throws: AetherError.self) {
        try await H3ChainCommand.run(["qualify-h3-chain", "/missing", "/missing", "/missing", root.path, "--capture"])
    }
    #expect(try Data(contentsOf: root.appendingPathComponent("report.json")) == data)
}

@Test func h3ChainCancellationRejectsLaterBlocksAndIncorrectOwnershipOrder() throws {
    let valid = ["b0.nativeSubmitted", "b0.nativeCompleted", "b1.nativeSubmitted", "cancellationRequested",
                 "unloadStarted", "b1.nativeCompleted", "slotReleased", "runCancelled", "unloaded"]
    try H3ChainTrace.validate(valid, closing: true)
    for invalid in [valid + ["b2.nativeSubmitted"], Array(valid.dropFirst(2)),
                    valid.filter { $0 != "b1.nativeCompleted" },
                    valid.filter { $0 != "runCancelled" } + ["runSucceeded"],
                    ["b0.nativeSubmitted", "b0.nativeCompleted", "b1.nativeSubmitted", "cancellationRequested", "unloadStarted", "slotReleased", "b1.nativeCompleted", "runCancelled", "unloaded"]] {
        #expect(throws: AetherError.self) { try H3ChainTrace.validate(invalid, closing: true) }
    }
}


@Test func h3BackboneCancellationTraceRequiresTheMiddleBoundaryAndNoLaterWork() throws {
    let prefix = (0..<24).flatMap { ["b\($0).nativeSubmitted", "b\($0).nativeCompleted"] }
    let events = prefix + ["b24.nativeSubmitted", "cancellationRequested", "unloadStarted", "b24.nativeCompleted", "unloaded", "slotReleased", "runCancelled"]
    try H3ChainTrace.validate(events, closing: true, target: 24)
    #expect(throws: AetherError.self) { try H3ChainTrace.validate(events + ["b25.nativeSubmitted"], closing: true, target: 24) }
    #expect(throws: AetherError.self) { try H3ChainTrace.validate(Array(events.dropFirst(2)), closing: true, target: 24) }
}
