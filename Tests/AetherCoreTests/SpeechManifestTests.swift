import Foundation
import Testing
@testable import AetherCore

/// MANIFEST.md §5.8 (`kokoro@1`, proposed until G5). The fixture is the Kokoro exporter's own output
/// (`Forge/kokoro/export_kokoro.py --sample-manifest`), so a change to what it emits shows up here.
@Suite("Speech manifest")
struct SpeechManifestTests {
    static func fixture() throws -> Data { try Fixtures.data("kokoro/aether.json") }

    @Test func theKokoroExportersManifestDecodes() throws {
        let manifest = try Manifest.decode(Self.fixture())
        #expect(manifest.driver.description == "kokoro@1" && manifest.task == "text-to-speech")
        #expect(manifest.license == "Apache-2.0")
        let durations = try #require(manifest.graphs["durations"])
        let decoder = try #require(manifest.graphs["decoder"])
        #expect(durations.compute == .cpu && durations.expectFrequentReshapes == true)
        #expect(decoder.compute == .gpu && decoder.expectFrequentReshapes == false)
        #expect(manifest.graphs.values.allSatisfy { $0.residency == nil })   // both stay resident
        // One table per voice: float32 [510, 256], each file listed in the asset so the store installs it.
        let tables = try #require(manifest.tables)
        #expect(tables.count == 54)
        let listed = Set(try #require(manifest.assets["main"]?.files).map(\.path))
        for (name, table) in tables {
            #expect(name.hasPrefix("voice.") && table.asset == "main" && table.dtype == .float32)
            #expect(table.shape == [510, 256] && table.byteCount == 510 * 256 * 4)
            #expect(listed.contains(table.file), "\(table.file)")
        }
        #expect(try Manifest.decode(manifest.encoded()) == manifest)
    }

    /// The driver-owned block, read as JSON until `kokoro@1` types it (G5 step 2).
    @Test func theKokoroBlockNamesItsVoicesAndRules() throws {
        let json = try JSONSerialization.jsonObject(with: Self.fixture()) as! [String: Any]
        let block = try #require(json["kokoro"] as? [String: Any])
        #expect(block["sampleRate"] as? Int == 24000 && block["samplesPerFrame"] as? Int == 600)
        let phonemes = try #require(block["phonemes"] as? [String: Any])
        let vocabulary = try #require(phonemes["vocabulary"] as? [String: Int])
        #expect(vocabulary.count == 114 && vocabulary.keys.allSatisfy { $0.unicodeScalars.count == 1 })
        #expect(!vocabulary.values.contains(0))   // 0 is the boundary token only
        #expect(phonemes["maxPhonemes"] as? Int == 510 && phonemes["unknown"] as? String == "refuse")
        let frames = try #require(block["frames"] as? [String: Any])
        #expect(frames["buckets"] as? [Int] == [64, 128, 256, 512, 1024, 2048, 4096])
        #expect(frames["cap"] as? [String: Int] == ["macos": 1024, "ios": 512])
        let tables = try #require(json["tables"] as? [String: Any])
        let voices = try #require(block["voices"] as? [String: [String: String]])
        #expect(voices.count == tables.count)
        for (id, voice) in voices {
            #expect(voice["table"] == "voice.\(id)" && tables[voice["table"]!] != nil)
            #expect(["female", "male"].contains(voice["gender"] ?? ""))
        }
        #expect(voices["af_heart"] == ["table": "voice.af_heart", "language": "en-US", "gender": "female"])
    }
}
