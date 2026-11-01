import AetherKit
import AetherVision
@testable import AetherLanguageCoreAI
import Foundation
import Testing

/// The hybrid recurrent-state contract's pure rules: declared states, the padded plan and the registration, for
/// Qwen3.5's shape (rotary positions, image rows) and Granite 4.0-H's (neither).
struct HybridLanguageContractTests {
    typealias Port = LanguageHybridContract.Port

    private static let qwenStates = [
        LanguageHybridRegistration.State(name: "convState", scalarType: .float16, shape: [24, 1, 8192, 3]),
        LanguageHybridRegistration.State(name: "recurrentState", scalarType: .float32, shape: [24, 1, 32, 128, 128]),
    ]
    private static let graniteStates = [
        LanguageHybridRegistration.State(name: "convState", scalarType: .float16, shape: [28, 1, 1792, 3]),
        LanguageHybridRegistration.State(name: "ssmState", scalarType: .float32, shape: [28, 1, 48, 32, 128]),
    ]

    private func ports(width: Int, cache: [Int], vocab: Int, rope: Bool, image: Int?) -> (inputs: [String: Port], outputs: [String: Port]) {
        var inputs: [String: Port] = ["input_ids": Port(scalar: .int32, shape: [1, width]),
                                      "cache_positions": Port(scalar: .int32, shape: [1, width]),
                                      "keyCache": Port(scalar: .float16, shape: cache), "valueCache": Port(scalar: .float16, shape: cache)]
        if rope { inputs["rope_positions"] = Port(scalar: .int32, shape: [3, width]) }
        if let image { inputs["image_embeds"] = Port(scalar: .float16, shape: [width, image]) }
        var update = cache
        update[3] = width
        return (inputs, ["logits": Port(scalar: .float16, shape: [1, 1, vocab]), "keyUpdate": Port(scalar: .float16, shape: update),
                         "valueUpdate": Port(scalar: .float16, shape: update)])
    }

    private func states(_ declared: [LanguageHybridRegistration.State]) -> [String: Port] {
        Dictionary(uniqueKeysWithValues: declared.map { ($0.name, Port(scalar: $0.scalarType == .float32 ? .float32 : .float16, shape: $0.shape)) })
    }

    @Test func qwenAndGraniteSignaturesValidate() throws {
        let qwen = LanguageHybridRegistration(staticWidth: 64, states: Self.qwenStates)
        let q = ports(width: 64, cache: [8, 1, 4, -1, 256], vocab: 248_320, rope: true, image: 2560)
        let layout = try LanguageHybridContract.validate(registration: qwen, inputs: q.inputs, outputs: q.outputs,
                                                         states: states(Self.qwenStates))
        #expect(layout == .init(width: 64, vocabularySize: 248_320, cacheShape: [8, 1, 4, -1, 256], takesRopePositions: true,
                                imageHiddenSize: 2560))
        let granite = LanguageHybridRegistration(staticWidth: 64, states: Self.graniteStates)
        let g = ports(width: 64, cache: [4, 1, 4, -1, 64], vocab: 100_352, rope: false, image: nil)
        let graniteLayout = try LanguageHybridContract.validate(registration: granite, inputs: g.inputs, outputs: g.outputs,
                                                                states: states(Self.graniteStates))
        #expect(!graniteLayout.takesRopePositions && graniteLayout.imageHiddenSize == nil)
        // A static capacity (the registered context) is accepted; the session holds it to the context.
        let fixed = ports(width: 64, cache: [8, 1, 4, 4096, 256], vocab: 248_320, rope: true, image: 2560)
        let fixedLayout = try LanguageHybridContract.validate(registration: qwen, inputs: fixed.inputs, outputs: fixed.outputs,
                                                              states: states(Self.qwenStates))
        #expect(fixedLayout.cacheShape[3] == 4096)
    }

    @Test func mismatchedSignaturesAreRefused() {
        let registration = LanguageHybridRegistration(staticWidth: 64, states: Self.qwenStates)
        let good = ports(width: 64, cache: [8, 1, 4, -1, 256], vocab: 1000, rope: true, image: 2560)
        func refuses(inputs: [String: Port]? = nil, outputs: [String: Port]? = nil, states declared: [String: Port]? = nil) -> Bool {
            (try? LanguageHybridContract.validate(registration: registration, inputs: inputs ?? good.inputs,
                                                  outputs: outputs ?? good.outputs, states: declared ?? states(Self.qwenStates))) == nil
        }
        // Another width (a dynamic one included), an unknown input, image rows without rotary positions.
        #expect(refuses(inputs: ports(width: 32, cache: [8, 1, 4, -1, 256], vocab: 1000, rope: true, image: 2560).inputs))
        #expect(refuses(inputs: good.inputs.merging(["input_ids": Port(scalar: .int32, shape: [1, -1])]) { $1 }))
        #expect(refuses(inputs: good.inputs.merging(["deepstack": Port(scalar: .float16, shape: [3, 64, 2560])]) { $1 }))
        #expect(refuses(inputs: good.inputs.filter { $0.key != "rope_positions" }))
        // Update rows of another width, or a cache of another scalar type.
        #expect(refuses(inputs: good.inputs.merging(["valueCache": Port(scalar: .float32, shape: [8, 1, 4, -1, 256])]) { $1 }))
        #expect(refuses(outputs: good.outputs.merging(["keyUpdate": Port(scalar: .float16, shape: [8, 1, 4, 1, 256])]) { $1 }))
        // A state that isn't declared, one missing, or one of another scalar type or shape.
        var extra = states(Self.qwenStates)
        extra["other"] = Port(scalar: .float16, shape: [1])
        #expect(refuses(states: extra))
        #expect(refuses(states: states(Array(Self.qwenStates.prefix(1)))))
        var halved = states(Self.qwenStates)
        halved["recurrentState"] = Port(scalar: .float16, shape: [24, 1, 32, 128, 128])
        #expect(refuses(states: halved))
    }

    @Test func prefillIsPaddedNeverSteppingBack() {
        #expect(LanguageHybridContract.chunks(count: 130, width: 64) == [.init(offset: 0, count: 64), .init(offset: 64, count: 64),
                                                                         .init(offset: 128, count: 2)])
        #expect(LanguageHybridContract.chunks(count: 64, width: 64) == [.init(offset: 0, count: 64)])
        #expect(LanguageHybridContract.chunks(count: 3, width: 64) == [.init(offset: 0, count: 3)])
        let (ids, positions) = LanguageHybridContract.padded(tokens: [7, 8], position: 40, width: 4)
        #expect(ids == [7, 8, 0, 0] && positions == [40, 41, -1, -1])
    }

    @Test func registrationValidatesAndRoundTrips() throws {
        let registration = LanguageHybridRegistration(staticWidth: 64, states: Self.qwenStates)
        try registration.validate()
        #expect(registration.stateBytes == 24 * 8192 * 3 * 2 + 24 * 32 * 128 * 128 * 4)
        let decoded = try JSONDecoder().decode(LanguageHybridRegistration.self, from: JSONEncoder().encode(registration))
        #expect(decoded == registration)
        let json = #"{"layout":"static-width","staticWidth":64,"states":[{"name":"ssmState","role":"recurrent","scalarType":"float32","shape":[28,1,48,32,128]}]}"#
        try JSONDecoder().decode(LanguageHybridRegistration.self, from: Data(json.utf8)).validate()
        for bad in [LanguageHybridRegistration(staticWidth: 0, states: Self.qwenStates),
                    LanguageHybridRegistration(staticWidth: 64, states: []),
                    LanguageHybridRegistration(staticWidth: 64, states: Self.qwenStates + Self.qwenStates.prefix(1)),
                    LanguageHybridRegistration(staticWidth: 64, states: [.init(name: "keyCache", scalarType: .float16, shape: [1])]),
                    LanguageHybridRegistration(staticWidth: 64, states: [.init(name: "s", scalarType: .float16, shape: [0, 4])])] {
            #expect(throws: AetherError.self) { try bad.validate() }
        }
    }

    /// Qwen3.5's encoder has no DeepStack output: the static layout accepts zero levels.
    @Test func encoderWithoutDeepstackHasZeroLevels() throws {
        let grid = Qwen3VLImagePreprocessor.Grid(height: 4, width: 6)
        let inputs = [[24, 1536], [24, 4], [24, 4], [24, 2]]
        let layout = try #require(LanguageVisionContract.staticLayout(grid: grid, mergeSize: 2, inputs: inputs,
                                                                      outputs: [[6, 2560], [0, 6, 2560]]))
        #expect(layout == (1536, 2560, 0))
        #expect(LanguageVisionContract.staticLayout(grid: grid, mergeSize: 2, inputs: inputs, outputs: [[6, 2560], [-1, 6, 2560]]) == nil)
    }
}
