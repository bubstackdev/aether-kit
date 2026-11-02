@testable import AetherProbe
import AetherH3
import AetherKit
import Foundation
import Testing

/// The FL2VA labels' ids under the checkpoint's tokenizer, pinned from the Transformers oracle
/// (`Tests/AetherH3TokenizersTests/Fixtures/token-oracle.json`, checked against the real tokenizer when
/// `AETHER_H3_TOKENIZER` is set).
private let labelIDs: [String: [Int32]] = ["<Picture 1>: ": [21604, 3826, 220, 16, 26818, 220], "<Picture 2>: ": [21604, 3826, 220, 17, 26818, 220]]

@available(macOS 27, iOS 27, *)
@Test func h3DumpRendersTokenizeTheKeyframeLabelsNotThePrompt() throws {
    let prompt: [Int32] = (0..<49).map { 1_000 + $0 }
    let tokenize = H3ProfileCommand.dumpTokenize(prompt) { text in
        guard let ids = labelIDs[text] else { throw AetherError.invalidRequest("unexpected text \(text)") }
        return ids
    }
    #expect(try tokenize("ids") == prompt)
    let presentation = try H3Presentation(prompt: try tokenize("ids"), keyframes: [(960, 544), (960, 544)], tokenize: tokenize)
    // Two labelled 510-token blocks, then the prompt: upstream's 1,085 rows at 960x544 first+last.
    #expect(presentation.count == 1_085)
    #expect(Array(presentation.ids.prefix(7)) == labelIDs["<Picture 1>: "]! + [151_652])
    #expect(Array(presentation.ids[518..<525]) == labelIDs["<Picture 2>: "]! + [151_652])
    #expect(Array(presentation.ids.suffix(49)) == prompt)
}

/// Returns a fixed conditioning.
private struct FixedConditioner: H3PromptConditioner {
    let ids: [Int32], tags: [Int32]
    func encode(_ prompt: String, geometry: H3Geometry, onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3TensorData {
        try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, ids.count, 5120]), bytes: Data(count: ids.count * 5120 * 2))
    }
    func encode(_ prompt: String, keyframes: [H3Keyframe], geometry: H3Geometry,
                onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Conditioning {
        try H3Conditioning(tap: try await encode(prompt, geometry: geometry, onLayer: onLayer), tags: tags, ids: ids)
    }
    func supports(_ geometry: H3Geometry, keyframes: [H3Keyframe]) -> Bool { true }
    func release() async {}
}

@available(macOS 27, iOS 27, *)
@Test func h3DumpConditionerRefusesConditioningTheDumpDidNotLayOut() async throws {
    let dump = FileManager.default.temporaryDirectory.appendingPathComponent("h3-dump-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dump) }
    let out = try H3RawDirectory(new: dump)
    let ids: [Int32] = [5, 6, 7, 8], tags: [Int32] = [1, 0, 0, 1]
    func int32(_ values: [Int32]) throws -> H3TensorData {
        try H3TensorData(spec: H3TensorSpec(scalarType: .int32, shape: [values.count]), bytes: values.withUnsafeBytes { Data($0) })
    }
    out.fields = ["tokens": "4"]
    try out.put("ids", try int32([8])); try out.put("presentation_ids", try int32(ids)); try out.put("presentation_tags", try int32(tags))
    try out.save(["scope": "test"])
    let g = try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)
    let same = try H3DumpConditioner(FixedConditioner(ids: ids, tags: tags), dump: dump)
    #expect(try await same.encode("ids", keyframes: [], geometry: g) { _, _ in }.ids == ids)
    for (badIDs, badTags) in [([5, 6, 7, 9] as [Int32], tags), (ids, [1, 1, 1, 1] as [Int32]), ([5, 6, 7] as [Int32], [1, 0, 0] as [Int32])] {
        let conditioner = try H3DumpConditioner(FixedConditioner(ids: badIDs, tags: badTags), dump: dump)
        await #expect(throws: AetherError.self) { _ = try await conditioner.encode("ids", keyframes: [], geometry: g) { _, _ in } }
    }
}
