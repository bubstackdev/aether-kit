@testable import AetherLanguageCoreAI
import Foundation
import Testing

private struct RopeFixture: Decodable {
    struct Tokens: Decodable { let visionStart: Int; let imagePad: Int; let visionEnd: Int }
    struct Case: Decodable { let name: String; let promptTokens: [Int]; let grid: [Int]; let ropePositions: [[Int]] }
    let imageTokens: Tokens
    let mergeSize: Int
    let cases: [Case]
}

private func features(grid: [Int], merge: Int = 2) throws -> LanguageImageFeatures {
    let tokens = grid[0] * (grid[1] / merge) * (grid[2] / merge)
    return try LanguageImageFeatures(grid: (grid[0], grid[1], grid[2]), mergeSize: merge, hiddenSize: 1, deepstackLevels: 1,
                                     embeddings: Array(repeating: 0, count: tokens), deepstack: Array(repeating: 0, count: tokens))
}

struct LanguageVisionPromptTests {
    private let tokens = LanguageImageTokens(visionStart: 151652, imagePad: 151655, visionEnd: 151653)

    /// Expanding a rendered placeholder reproduces the upstream processor's ids, and the host M-RoPE
    /// rule reproduces the upstream model's own rope index for every prompt token.
    @Test func expansionAndPositionsMatchUpstream() throws {
        let url = try #require(Bundle.module.url(forResource: "qwen3vl-rope", withExtension: "json", subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(RopeFixture.self, from: Data(contentsOf: url))
        for c in fixture.cases {
            // Collapse the processor's expanded pads back to the single rendered placeholder.
            var rendered: [Int] = []
            for id in c.promptTokens where !(id == tokens.imagePad && rendered.last == tokens.imagePad) { rendered.append(id) }
            let plan = try LanguagePromptPlan.expand(LanguageVisionPrompt(renderedTokenIDs: rendered, images: [features(grid: c.grid)]),
                                                     tokens: tokens)
            #expect(plan.ids == c.promptTokens, "ids \(c.name)")
            #expect(plan.rope.map { $0.map(Int.init) } == c.ropePositions, "rope \(c.name)")
            // The next token continues from the last text position + 1, on every axis.
            #expect(Int(plan.rope[0].last!) + 1 == plan.ids.count + plan.ropeDelta)
            #expect(plan.spans.count == 1 && plan.spans[0].count == c.grid[0] * c.grid[1] * c.grid[2] / 4)
        }
    }

    @Test func positionsAdvanceByTheLargestImageSideNotItsTokenCount() throws {
        // [text, start, image 4x6 patches -> 2x3 tokens, end, text]: the image spans rotary positions 2..4.
        let image = try features(grid: [1, 4, 6])
        let plan = try LanguagePromptPlan.expand(LanguageVisionPrompt(renderedTokenIDs: [7, tokens.visionStart, tokens.imagePad,
                                                                                       tokens.visionEnd, 8], images: [image]), tokens: tokens)
        #expect(plan.ids == [7, tokens.visionStart] + Array(repeating: tokens.imagePad, count: 6) + [tokens.visionEnd, 8])
        #expect(plan.rope[0] == [0, 1, 2, 2, 2, 2, 2, 2, 5, 6])
        #expect(plan.rope[1] == [0, 1, 2, 2, 2, 3, 3, 3, 5, 6])
        #expect(plan.rope[2] == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6])
        #expect(plan.ropeDelta == 7 - 10)                 // next token: cache position 10, rotary position 7
    }

    @Test func malformedPromptsAreRefused() throws {
        let image = try features(grid: [1, 2, 2])
        let bare = LanguageVisionPrompt(renderedTokenIDs: [1, tokens.imagePad, 2], images: [image])
        #expect(throws: LanguageGenerationError.self) { try LanguagePromptPlan.expand(bare, tokens: tokens) }
        let missing = LanguageVisionPrompt(renderedTokenIDs: [tokens.visionStart, tokens.imagePad, tokens.visionEnd], images: [])
        #expect(throws: LanguageGenerationError.self) { try LanguagePromptPlan.expand(missing, tokens: tokens) }
        let extra = LanguageVisionPrompt(renderedTokenIDs: [1, 2], images: [image])
        #expect(throws: LanguageGenerationError.self) { try LanguagePromptPlan.expand(extra, tokens: tokens) }
        #expect(throws: LanguageGenerationError.self) {
            try LanguageImageFeatures(grid: (1, 3, 2), mergeSize: 2, hiddenSize: 1, deepstackLevels: 0, embeddings: [], deepstack: [])
        }
        #expect(throws: LanguageGenerationError.self) {
            try LanguageImageFeatures(grid: (1, 2, 2), mergeSize: 2, hiddenSize: 4, deepstackLevels: 1, embeddings: [0], deepstack: [0])
        }
    }

    @Test func imageSpansAreSingleSubmissionsBetweenBucketedTextChunks() throws {
        let image = try features(grid: [1, 8, 8])          // 16 tokens
        let rendered = Array(1...19) + [tokens.visionStart, tokens.imagePad, tokens.visionEnd] + Array(30...34)
        let plan = try LanguagePromptPlan.expand(LanguageVisionPrompt(renderedTokenIDs: rendered, images: [image]), tokens: tokens)
        let submissions = plan.submissions(contract: .bucketedPositions, requestedChunk: 8, maximumChunk: 512)
        // 20 text tokens (19 + vision start) as 8, 8 and a last 8 stepping back to 12; the 16-token image alone; then 6
        // text tokens, shorter than the width, as one chunk. No text chunk reaches back into the image.
        #expect(submissions.map(\.count) == [8, 8, 8, 16, 6])
        #expect(submissions.map(\.image) == [nil, nil, nil, 0, nil])
        #expect(submissions.map(\.start) == [0, 8, 12, 20, 36])
        #expect(submissions.last.map { $0.start + $0.count } == plan.ids.count)
    }
}
