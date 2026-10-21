import AetherKit
import AetherKrea
import Foundation
import Testing
@testable import AetherProbe

@Test func referenceParserKeepsEncodedAndProvidedBaselinesSeparate() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    let object: [String: Any] = [
        "runs": [["encodedTextSHA256": "encoded-text", "stepLatentsSHA256": Array(repeating: "encoded", count: 8), "pixelsSHA256": "encoded-pixels"]],
        "pairs": [["stepLatentsSHA256": [Array(repeating: "base", count: 8), Array(repeating: "provided", count: 8)], "pixelsSHA256": ["base-pixels", "provided-pixels"]]],
    ]
    try JSONSerialization.data(withJSONObject: object).write(to: url)
    let encoded = try ReferenceHashes.load(url, encoded: true)
    let provided = try ReferenceHashes.load(url, encoded: false)
    #expect(encoded.text == "encoded-text")
    #expect(encoded.steps == Array(repeating: "encoded", count: 8))
    #expect(encoded.pixels == "encoded-pixels")
    #expect(provided.text == nil)
    #expect(provided.steps == Array(repeating: "provided", count: 8))
    #expect(provided.pixels == "provided-pixels")
}

@Test func parityHarnessRejectsIncompleteRunsAndRecordsMismatch() throws {
    #expect(throws: AetherError.self) { try ParityJournal(expected: ReferenceHashes(text: nil, steps: [], pixels: "")) }
    let journal = try ParityJournal(expected: ReferenceHashes(text: "wrong", steps: Array(repeating: "", count: 8), pixels: ""))
    let text = try KreaConditioning(tensor: Float16Tensor(shape: [20, 30720], values: Array(repeating: 0, count: 20 * 30720)))
    #expect(throws: AetherError.self) { try journal.accept(.conditioningReady(text)) }
    #expect(journal.events.first?.passed == false)
    #expect(throws: AetherError.self) { try journal.accept(.decoded) }
    #expect(throws: AetherError.self) {
        try journal.finish(GeneratedImage(pixelsCHW: FloatTensor(shape: [3, 1, 1], values: [0, 0, 0])))
    }
}

@Test func parityPreparationFailureWritesFailedReport() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    await #expect(throws: (any Error).self) {
        try await KreaCommands.run(["parity-krea", "/missing/bundle", "/missing/lab", "/missing/fixtures", "/missing/shape.json", root.path, "provided"])
    }
    let report = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("report.json"))) as! [String: Any]
    #expect(report["passed"] as? Bool == false)
    #expect(report["modelStreamsVerified"] as? Bool == false)
    #expect(report["stage"] as? String == "preparation")
    #expect(report["error"] as? String != nil)
}
