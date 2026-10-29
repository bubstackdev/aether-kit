@testable import AetherAssets
import Foundation
import Testing

struct ModelCatalogEntryTests {
    private let revision = "0123456789abcdef0123456789abcdef01234567"

    private func manifest() throws -> AssetManifest {
        try AssetManifest(assetIdentity: "example/1:abc", executionContract: "example.bundle/1", files: [
            AssetFile(path: "model.aimodel/main.mlirb", byteCount: 10, sha256: String(repeating: "a", count: 64)),
            AssetFile(path: "language-model.json", byteCount: 2, sha256: String(repeating: "b", count: 64)),
        ])
    }

    @Test func entryRoundTripsThroughJSONAndKeepsItsInstallationID() throws {
        for source in [ModelSource.https(base: URL(string: "https://models.example.com/kit/qwen3")!),
                       .huggingFace(repository: "org/model-name_1.0", revision: revision, subdirectory: "bundles/macos")] {
            let entry = try ModelCatalogEntry(manifest: manifest(), source: source, displayName: "Example")
            let decoded = try JSONDecoder().decode(ModelCatalogEntry.self, from: JSONEncoder().encode(entry))
            #expect(decoded == entry)
            #expect(try decoded.installationID == manifest().installationID)
            #expect(decoded.byteCount == 12)
        }
    }

    @Test func sourceJSONNamesItsKind() throws {
        let json = try JSONEncoder().encode(ModelSource.huggingFace(repository: "org/model", revision: revision, subdirectory: nil))
        let object = try #require(try JSONSerialization.jsonObject(with: json) as? [String: String])
        #expect(object == ["kind": "huggingFace", "repository": "org/model", "revision": revision])
        let unknown = Data(#"{"kind":"ftp","base":"ftp://example.com"}"#.utf8)
        #expect(throws: AssetStoreError.self) { try JSONDecoder().decode(ModelSource.self, from: unknown) }
    }

    @Test func huggingFaceNeedsAFullCommitRevisionAndOwnerSlashName() throws {
        for bad in ["main", "v1.0", String(revision.dropLast()), revision.uppercased(), revision + "0"] {
            #expect(throws: AssetStoreError.self) {
                try ModelSource.huggingFace(repository: "org/model", revision: bad, subdirectory: nil).validate()
            }
        }
        for bad in ["model", "org/model/extra", "/model", "org/", "../model", "org/mo del", "org/mödel"] {
            #expect(throws: AssetStoreError.self) {
                try ModelSource.huggingFace(repository: bad, revision: revision, subdirectory: nil).validate()
            }
        }
        for bad in ["/abs", "a/../b", "a//b", ""] {
            #expect(throws: AssetStoreError.self) {
                try ModelSource.huggingFace(repository: "org/model", revision: revision, subdirectory: bad).validate()
            }
        }
    }

    @Test func httpsSourcesRefuseOtherSchemesAndCredentials() {
        for bad in ["http://example.com/m", "file:///tmp/m", "https://user:pw@example.com/m", "https://example.com/m?sig=1",
                    "https://example.com/m#frag"] {
            #expect(throws: AssetStoreError.self) { try ModelSource.https(base: URL(string: bad)!).validate() }
        }
    }

    @Test func downloadURLsJoinTheSourceAndThePath() throws {
        let hub = ModelSource.huggingFace(repository: "org/model", revision: revision, subdirectory: "bundles/macos")
        #expect(try hub.url(for: "model.aimodel/main.mlirb").absoluteString
            == "https://huggingface.co/org/model/resolve/\(revision)/bundles/macos/model.aimodel/main.mlirb")
        let root = ModelSource.huggingFace(repository: "org/model", revision: revision, subdirectory: nil)
        #expect(try root.url(for: "language-model.json").absoluteString
            == "https://huggingface.co/org/model/resolve/\(revision)/language-model.json")
        let cdn = ModelSource.https(base: URL(string: "https://cdn.example.com/kit/qwen3/")!)
        #expect(try cdn.url(for: "tokenizer/tokenizer.json").absoluteString
            == "https://cdn.example.com/kit/qwen3/tokenizer/tokenizer.json")
        #expect(throws: AssetStoreError.self) { try cdn.url(for: "../escape") }
    }

    @Test func anEntryOnlyHandsOutURLsForItsOwnFiles() throws {
        let entry = try ModelCatalogEntry(manifest: manifest(), source: .https(base: URL(string: "https://example.com/m")!))
        #expect(try entry.url(for: entry.manifest.files[0]).absoluteString == "https://example.com/m/language-model.json")
        let stranger = AssetFile(path: "other.bin", byteCount: 1, sha256: String(repeating: "c", count: 64))
        #expect(throws: AssetStoreError.self) { try entry.url(for: stranger) }
    }

    @Test func decodingRefusesAnInvalidEntry() throws {
        let entry = try ModelCatalogEntry(manifest: manifest(), source: .https(base: URL(string: "https://example.com/m")!))
        var object = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(entry)) as? [String: Any])
        object["schemaVersion"] = 2
        #expect(throws: AssetStoreError.self) {
            try JSONDecoder().decode(ModelCatalogEntry.self, from: JSONSerialization.data(withJSONObject: object))
        }
    }

    @Test func requirementsRoundTripAndAreOptional() throws {
        let source = ModelSource.https(base: URL(string: "https://example.com/m")!)
        let requirements = try ModelRequirements(platforms: [.ios], memoryBytes: 4 << 30)
        let entry = try ModelCatalogEntry(manifest: manifest(), source: source, requirements: requirements)
        let json = try JSONEncoder().encode(entry)
        #expect(try JSONDecoder().decode(ModelCatalogEntry.self, from: json).requirements == requirements)
        let object = try #require(try JSONSerialization.jsonObject(with: json) as? [String: Any])
        #expect((object["requirements"] as? [String: Any])?["platforms"] as? [String] == ["ios"])
        // An entry written before requirements existed has none, and encodes none.
        let plain = try ModelCatalogEntry(manifest: manifest(), source: source)
        let plainObject = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(plain)) as? [String: Any])
        #expect(plainObject["requirements"] == nil)
        #expect(try JSONDecoder().decode(ModelCatalogEntry.self, from: JSONEncoder().encode(plain)).requirements == nil)
    }

    @Test func invalidRequirementsAreRefused() throws {
        #expect(throws: AssetStoreError.self) { try ModelRequirements(platforms: []) }
        #expect(throws: AssetStoreError.self) { try ModelRequirements(platforms: [.macos, .macos]) }
        #expect(throws: AssetStoreError.self) { try ModelRequirements(memoryBytes: 0) }
        let entry = try ModelCatalogEntry(manifest: manifest(), source: .https(base: URL(string: "https://example.com/m")!))
        var object = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(entry)) as? [String: Any])
        object["requirements"] = ["platforms": [String]()]
        #expect(throws: AssetStoreError.self) {
            try JSONDecoder().decode(ModelCatalogEntry.self, from: JSONSerialization.data(withJSONObject: object))
        }
        object["requirements"] = ["platforms": ["iOS"]]
        #expect(throws: AssetStoreError.self) {
            try JSONDecoder().decode(ModelCatalogEntry.self, from: JSONSerialization.data(withJSONObject: object))
        }
        // A platform this Kit doesn't know still decodes (it matches no device), so a newer entry stays readable.
        object["requirements"] = ["platforms": ["visionos", "ios"]]
        let newer = try JSONDecoder().decode(ModelCatalogEntry.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(newer.requirements?.platforms == [ModelRequirements.Platform(rawValue: "visionos"), .ios])
    }

    @Test func huggingFaceAuthorizationOnlyGoesToTheHub() {
        let headers = ModelPreparation.huggingFaceAuthorization(token: "hf_x")
        #expect(headers(URL(string: "https://huggingface.co/org/m/resolve/abc/f")!) == ["Authorization": "Bearer hf_x"])
        #expect(headers(URL(string: "https://cdn.example.com/f")!).isEmpty)
    }
}
