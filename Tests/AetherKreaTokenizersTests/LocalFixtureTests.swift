import AetherKrea
import AetherKreaTokenizers
import CryptoKit
import Foundation
import Testing

private let environment = ProcessInfo.processInfo.environment

@Test(.enabled(if: environment["AETHER_KREA_BUNDLE"] != nil && environment["AETHER_KREA_FIXTURES"] != nil))
func tokenizerMatchesEveryRetainedPrompt() async throws {
    let root = URL(fileURLWithPath: try #require(environment["AETHER_KREA_BUNDLE"]))
    let fixtures = URL(fileURLWithPath: try #require(environment["AETHER_KREA_FIXTURES"]))
    let assets = try KreaAssets.open(at: root)
    let tokenizer = try await KreaTokenizer.open(assets: assets)
    let lines = try String(contentsOf: fixtures.appendingPathComponent("cases.jsonl"), encoding: .utf8)
    var count = 0
    for line in lines.split(separator: "\n") {
        let record = try JSONDecoder().decode(Fixture.self, from: Data(line.utf8))
        let prompt = try tokenizer.prepare(record.prompt)
        let bytes = try Data(contentsOf: fixtures.appendingPathComponent(record.files.ids))
        #expect(bytes.count.isMultiple(of: 4))
        let ids = stride(from: 0, to: bytes.count, by: 4).map { offset in
            Int32(bitPattern: UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8 | UInt32(bytes[offset + 2]) << 16 | UInt32(bytes[offset + 3]) << 24)
        }
        #expect(prompt.tokenIDs == ids, "fixture: \(record.id)")
        #expect(prompt.conditioningRows == record.textShape[0])
        #expect(!prompt.wasTruncated)
        count += 1
    }
    #expect(count == 6)
    let short = try tokenizer.prepare("hello")
    let long = try tokenizer.prepare(String(repeating: "hello ", count: 2000))
    #expect(long.wasTruncated)
    #expect(long.tokenIDs.suffix(5) == short.tokenIDs.suffix(5))
}

@Test(.enabled(if: environment["AETHER_KREA_LAB"] != nil))
func rotaryMatchesEveryRetainedSwiftByte() throws {
    let lab = URL(fileURLWithPath: try #require(environment["AETHER_KREA_LAB"]))
    let reference = try Data(contentsOf: lab.appendingPathComponent("records/round4/block-inputs-tower/rope_image.f32"))
    let tensor = try KreaRotary.image(configuration: KreaConfiguration(width: 768, height: 1344))
    var generated = Data(capacity: tensor.values.count * 4)
    for value in tensor.values {
        let bits = value.bitPattern
        generated.append(contentsOf: [UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8), UInt8(truncatingIfNeeded: bits >> 16), UInt8(truncatingIfNeeded: bits >> 24)])
    }
    #expect(generated == reference)
}

private struct Fixture: Decodable {
    let id: String
    let prompt: String
    let textShape: [Int]
    let files: Files
    struct Files: Decodable { let ids: String }
}
