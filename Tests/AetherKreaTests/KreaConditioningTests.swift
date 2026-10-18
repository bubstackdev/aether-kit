import AetherKit
import Testing
@testable import AetherKrea

@Test func promptTruncationPreservesChatSuffixAndInputs() throws {
    let raw = Array(0..<700)
    let prompt = try KreaPrompt(renderedTokenIDs: raw)
    #expect(prompt.wasTruncated)
    #expect(prompt.tokenIDs.count == 546)
    #expect(prompt.conditioningRows == 512)
    #expect(Array(prompt.tokenIDs.suffix(5)) == [695, 696, 697, 698, 699])
    #expect(Array(prompt.tokenIDs.prefix(541)) == (0..<541).map(Int32.init))
    let inputs = try prompt.encoderInputs
    #expect(inputs["input_ids"]?.shape == [1, 546])
    guard case .int32(let positions) = inputs["position_ids"],
          case .int32(let mask) = inputs["attention_mask"] else {
        Issue.record("encoder inputs must be int32")
        return
    }
    #expect(positions.values == (0..<546).map(Int32.init))
    #expect(mask.values == Array(repeating: 1, count: 546))
    #expect(throws: AetherError.self) { try KreaPrompt(tokenIDs: [0]) }
    #expect(throws: AetherError.self) { try KreaPrompt(renderedTokenIDs: Array(repeating: -1, count: 600)) }
    #expect(throws: AetherError.self) { try KreaPrompt(renderedTokenIDs: Array(repeating: Int.max, count: 600)) }
    #expect(KreaPrompt.render("hello").hasSuffix("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"))
}

@Test func conditioningDropsExactlyTheSystemPrefixWithoutReordering() throws {
    let prompt = try KreaPrompt(tokenIDs: Array(repeating: 0, count: 36))
    var rows = [Float16](repeating: 0, count: 36 * KreaConditioning.channels)
    rows[34 * KreaConditioning.channels] = 7
    rows[35 * KreaConditioning.channels] = 9
    rows[rows.count - 1] = 11
    let encoded = try Float16Tensor(shape: [36, KreaConditioning.channels], values: rows)
    let conditioning = try KreaConditioning(encoderOutput: encoded, prompt: prompt)
    #expect(conditioning.tensor.shape == [2, 30720])
    #expect(conditioning.tensor.values[0] == 7)
    #expect(conditioning.tensor.values[30720] == 9)
    #expect(conditioning.tensor.values.last == 11)
    #expect(throws: AetherError.self) {
        try KreaConditioning(encoderOutput: encoded, prompt: KreaPrompt(tokenIDs: Array(repeating: 0, count: 35)))
    }
    let overflow = try FloatTensor(shape: [1, 30720], values: Array(repeating: 70000, count: 30720))
    #expect(throws: AetherError.self) { try KreaConditioning(provided: overflow) }
}

@Test func rotaryUsesSplitHalvesAndCorrectSpatialAxes() throws {
    let tensor = try KreaRotary.image(configuration: KreaConfiguration(width: 32, height: 32))
    #expect(tensor.shape == [4, 2, 128])
    #expect(Array(tensor.values.prefix(128)) == Array(repeating: 1, count: 128))
    #expect(Array(tensor.values[128..<256]) == Array(repeating: 0, count: 128))
    // Time axis stays zero; the first x frequency changes at column 1, y at row 1.
    #expect(tensor.values[256] == 1)
    #expect(tensor.values[256 + 16] == 1)
    #expect(tensor.values[256 + 40] != 1)
    #expect(tensor.values[512 + 16] != 1)
    #expect(tensor.values[512 + 40] == 1)
}
