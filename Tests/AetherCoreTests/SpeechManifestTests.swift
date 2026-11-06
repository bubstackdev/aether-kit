import Foundation
import Testing
@testable import AetherCore

/// MANIFEST.md §5.9 (`kokoro@1`). The fixture is the Kokoro exporter's own output
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
        #expect(durations.compute == .gpu && durations.expectFrequentReshapes == true)   // the CPU traps in Swift (export_kokoro.COMPUTE)
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

    /// The typed `kokoro` block.
    @Test func theKokoroBlockNamesItsVoicesAndRules() throws {
        let manifest = try Manifest.decode(Self.fixture())
        let kokoro = try #require(manifest.kokoro)
        #expect(kokoro.sampleRate == 24000 && kokoro.samplesPerFrame == 600 && kokoro.harmonics == 9)
        #expect(kokoro.phonemes.vocabulary.count == 114 && !kokoro.phonemes.vocabulary.values.contains(0))
        #expect(kokoro.phonemes.maxPhonemes == 510 && kokoro.phonemes.unknown == .refuse && kokoro.phonemes.normalization == .none)
        #expect(kokoro.style.dimension == 256 && kokoro.style.row == .phonemeCountMinusOne)
        #expect(kokoro.durations.frames == .roundHalfToEven && kokoro.durations.alignment == .repeat)
        #expect(kokoro.frames.buckets == [64, 128, 256, 512, 1024, 2048, 4096] && kokoro.frames.cap == ["macos": 1024, "ios": 512])
        #expect(kokoro.frames.bucket(for: 98) == 128 && kokoro.frames.bucket(for: 4097) == nil)
        #expect(kokoro.noise.kind == .splitmix64BoxMuller && kokoro.noise.order == .rowMajor)
        #expect(kokoro.voices.count == manifest.tables?.count)
        #expect(kokoro.voices["af_heart"] == .init(table: "voice.af_heart", language: "en-US", gender: .female))
    }

    /// A rule the driver doesn't implement, or an inconsistent block, is refused when the manifest decodes, naming its field.
    @Test func anUnsupportedKokoroBlockIsRefused() throws {
        func refused(_ mutate: (inout [String: Any]) -> Void) throws -> String? {
            var json = try JSONSerialization.jsonObject(with: Self.fixture()) as! [String: Any]
            var block = json["kokoro"] as! [String: Any]
            mutate(&block)
            json["kokoro"] = block
            do {
                _ = try Manifest.decode(try JSONSerialization.data(withJSONObject: json))
                return nil
            } catch let error as AetherError {
                return error.description
            }
        }
        func edit(_ key: String, _ field: String, _ value: Any) -> (inout [String: Any]) -> Void {
            { block in
                var inner = block[key] as! [String: Any]
                inner[field] = value
                block[key] = inner
            }
        }
        #expect(try refused(edit("phonemes", "normalization", "nfc")) != nil)
        #expect(try refused(edit("phonemes", "unknown", "drop")) != nil)
        #expect(try refused(edit("phonemes", "vocabulary", ["ab": 1]))?.contains("kokoro.phonemes.vocabulary.ab") == true)
        #expect(try refused(edit("phonemes", "vocabulary", ["a": 1, "b": 1])) != nil)
        #expect(try refused(edit("phonemes", "vocabulary", ["a": 0])) != nil)
        #expect(try refused(edit("durations", "frames", "round(duration)")) != nil)
        #expect(try refused(edit("frames", "buckets", [128, 64]))?.contains("kokoro.frames.buckets") == true)
        #expect(try refused(edit("frames", "cap", ["macos": 9000]))?.contains("kokoro.frames.cap.macos") == true)
        #expect(try refused(edit("style", "dimension", 128))?.contains("tables.voice.") == true)
        #expect(try refused { block in
            var voices = block["voices"] as! [String: Any]
            voices["af_heart"] = ["table": "voice.nope", "language": "en-US", "gender": "female"]
            block["voices"] = voices
        }?.contains("kokoro.voices.af_heart.table") == true)
        #expect(try refused { block in
            var voices = block["voices"] as! [String: Any]
            voices["xx"] = voices["af_heart"]
            block["voices"] = voices
        }?.contains("kokoro.voices.xx") == true)
    }
}
