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

/// Opt-in: the tokenizer's reference prompts against the Krea 2 reference oracle (`krea2-reference` fixture with
/// ids.i32/rope.i32 and image PNGs). Reads tokenizer files only; loads no model.
@Test(.enabled(if: environment["AETHER_KREA_BUNDLE"] != nil && environment["AETHER_KREA_REFERENCE_FIXTURES"] != nil))
func referencePromptsMatchTheOracleTokensAndPositions() async throws {
    let root = URL(fileURLWithPath: try #require(environment["AETHER_KREA_BUNDLE"]))
    let fixtures = URL(fileURLWithPath: try #require(environment["AETHER_KREA_REFERENCE_FIXTURES"]))
    let tokenizer = try await KreaTokenizer.open(assets: KreaAssets.open(at: root))
    func int32s(_ name: String) throws -> [Int32] {
        let bytes = try Data(contentsOf: fixtures.appendingPathComponent(name))
        return stride(from: 0, to: bytes.count, by: 4).map { offset in
            Int32(bitPattern: UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8 | UInt32(bytes[offset + 2]) << 16 | UInt32(bytes[offset + 3]) << 24)
        }
    }
    struct Case: Decodable { let id: String; let prompt: String; let files: Files; struct Files: Decodable { let ids: String; let rope: String; let images: [String] } }
    var count = 0
    for line in try String(contentsOf: fixtures.appendingPathComponent("cases.jsonl"), encoding: .utf8).split(separator: "\n") {
        let record = try JSONDecoder().decode(Case.self, from: Data(line.utf8))
        let images = try record.files.images.map {
            try KreaReferencePreprocessing.prepare(KreaReferenceImage(encoded: Data(contentsOf: fixtures.appendingPathComponent($0))))
        }
        let prompt = try tokenizer.prepare(record.prompt, references: images)
        #expect(prompt.tokenIDs == (try int32s(record.files.ids)), "ids: \(record.id)")
        #expect(prompt.ropePositions == (try int32s(record.files.rope)), "rope: \(record.id)")
        count += 1
    }
    #expect(count == 6)
}

/// Opt-in: the local bundle declares the reference contract, keeps its text-only fingerprint, and gains a separate
/// reference fingerprint. Opening reads headers, hash declarations and tokenizer files only.
@Test(.enabled(if: environment["AETHER_KREA_BUNDLE"] != nil))
func localBundleDeclaresReferencesWithASeparateIdentity() throws {
    let assets = try KreaAssets.open(at: URL(fileURLWithPath: try #require(environment["AETHER_KREA_BUNDLE"])))
    #expect(assets.supportsReferences)
    #expect(assets.maximumTextRows == 1116)
    let reference = try #require(assets.referenceFingerprint)
    #expect(reference != assets.fingerprint && reference.count == 64)
    if let expected = environment["AETHER_KREA_TEXT_FINGERPRINT_PREFIX"] { #expect(assets.fingerprint.hasPrefix(expected)) }
    print("text-only fingerprint \(assets.fingerprint); reference fingerprint \(reference)")
}
