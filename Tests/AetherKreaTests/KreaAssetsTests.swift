import AetherKit
import AetherAssets
import AetherKreaCoreAI
import CryptoKit
import Foundation
import Testing
@testable import AetherKrea

struct BundleFixture {
    let root: URL

    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let recipe = try #require(Bundle.module.url(forResource: "krea-recipe", withExtension: "json", subdirectory: "Fixtures"))
        try FileManager.default.copyItem(at: recipe, to: root.appendingPathComponent("aether.json"))
        var assets: [String: String] = [:]
        for name in ["textEncoder", "vaeDecoder", "denoiser0", "denoiser1", "denoiser2", "denoiser3"] {
            let directory = root.appendingPathComponent(name + ".aimodel")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let data = Data("fixture model: \(name)".utf8)
            try data.write(to: directory.appendingPathComponent("main.mlirb"))
            try Data(SHA256.hash(data: data)).write(to: directory.appendingPathComponent("main.hash"))
            try Data("{\"assetVersion\":\"2.0\"}".utf8).write(to: directory.appendingPathComponent("metadata.json"))
            assets[name] = name + ".aimodel"
        }
        let metadata: [String: Any] = ["name": "krea_2_turbo_realism", "assets": assets]
        try JSONSerialization.data(withJSONObject: metadata).write(to: root.appendingPathComponent("metadata.json"))
        let tokenizer = root.appendingPathComponent("tokenizer")
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try Data("{}".utf8).write(to: tokenizer.appendingPathComponent(name))
        }
    }

    func cleanup() { try? FileManager.default.removeItem(at: root) }

    func edit(_ name: String, _ edit: (inout [String: Any]) -> Void) throws {
        let url = root.appendingPathComponent(name)
        var json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        edit(&json)
        try JSONSerialization.data(withJSONObject: json).write(to: url)
    }
}

@Test func bundleVerificationDetectsModelAndTokenizerMutation() throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    #expect(assets.denoisers.count == 4)
    #expect(assets.fingerprint.count == 64)
    try assets.verify()
    let metadata = assets.decoder.url.appendingPathComponent("metadata.json")
    let original = try Data(contentsOf: metadata)
    try Data("{}".utf8).write(to: metadata)
    #expect(throws: AetherError.self) { try assets.verify() }
    try original.write(to: metadata)
    try Data("corrupt".utf8).write(to: assets.denoisers[0].url.appendingPathComponent("main.mlirb"))
    #expect(throws: AetherError.self) { try assets.verify() }
    try Data("changed".utf8).write(to: assets.tokenizerDirectory.appendingPathComponent("tokenizer.json"))
    #expect(throws: AetherError.self) { try assets.verifyTokenizer() }
    // Opening is deliberately not full model verification; identity changes with tokenizer bytes.
    let changed = try KreaAssets.open(at: fixture.root)
    #expect(changed.fingerprint != assets.fingerprint)
}

@Test func bundleRejectsUnsupportedRecipeBeforeModelsAreLoaded() throws {
    for change in ["schema", "shift", "prompt", "target", "binding"] {
        let fixture = try BundleFixture()
        defer { fixture.cleanup() }
        try fixture.edit("aether.json") { json in
            switch change {
            case "schema": json["schema"] = 2
            case "target":
                var target = json["target"] as! [String: Any]
                target["compiled"] = true
                json["target"] = target
            case "binding":
                var inputs = json["inputs"] as! [String: Any]
                inputs["input_ids"] = ["bind": "tokens", "dtype": "float32"]
                json["inputs"] = inputs
            default:
                var diffusion = json["diffusion"] as! [String: Any]
                if change == "shift" { diffusion["sigmaShift"] = ["rule": "constant", "mu": 2.0] }
                else {
                    var prompt = diffusion["prompt"] as! [String: Any]
                    prompt["dropLeadingRows"] = 0
                    diffusion["prompt"] = prompt
                }
                json["diffusion"] = diffusion
            }
        }
        #expect(throws: AetherError.self) { try KreaAssets.open(at: fixture.root) }
    }
}

@Test func bundleAcceptsTheVanillaExportAndRejectsOtherNames() throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let realism = try KreaAssets.open(at: fixture.root)
    try fixture.edit("metadata.json") { $0["name"] = "krea_2_turbo" }
    let vanilla = try KreaAssets.open(at: fixture.root)
    #expect(vanilla.denoisers.count == 4)
    // The metadata (and so the name) is part of the identity: approvals for one export never cover the other.
    #expect(vanilla.fingerprint != realism.fingerprint)
    for name in ["krea_2_raw", "krea_2_turbo_realism_v2", ""] {
        try fixture.edit("metadata.json") { $0["name"] = name }
        #expect(throws: AetherError.self) { try KreaAssets.open(at: fixture.root) }
    }
}

@Test func bundleRejectsTraversalSymlinkEscapeAndAliasedModels() throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    for relative in ["../outside.aimodel", "/private/tmp/outside.aimodel", "textEncoder.aimodel"] {
        try fixture.edit("metadata.json") { metadata in
            var assets = metadata["assets"] as! [String: Any]
            assets["denoiser0"] = relative
            metadata["assets"] = assets
        }
        #expect(throws: AetherError.self) { try KreaAssets.open(at: fixture.root) }
    }
    try fixture.edit("metadata.json") { metadata in
        var assets = metadata["assets"] as! [String: Any]
        assets["denoiser0"] = "denoiser0.aimodel"
        metadata["assets"] = assets
    }
    let file = fixture.root.appendingPathComponent("denoiser0.aimodel/main.mlirb")
    try FileManager.default.removeItem(at: file)
    try FileManager.default.createSymbolicLink(at: file, withDestinationURL: URL(fileURLWithPath: "/etc/hosts"))
    #expect(throws: AetherError.self) { try KreaAssets.open(at: fixture.root) }
}

@Test func bundleRejectsProvenanceMismatchAndMalformedDeclarations() throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    try fixture.edit("aether.json") { manifest in
        manifest["provenance"] = ["files": ["tokenizer/tokenizer.json": "sha256:" + String(repeating: "0", count: 64)]]
    }
    #expect(throws: AetherError.self) { try KreaAssets.open(at: fixture.root) }
    try fixture.edit("aether.json") { $0.removeValue(forKey: "provenance") }
    try Data([0]).write(to: fixture.root.appendingPathComponent("textEncoder.aimodel/main.hash"))
    #expect(throws: AetherError.self) { try KreaAssets.open(at: fixture.root) }
}


@Test func portableKreaInstallationPreservesFingerprintAndSessionReaderProtection() async throws {
    let fixture = try BundleFixture(); defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    let manifest = try assets.installationManifest()
    #expect(manifest.files.count == 22)
    let storage = fixture.root.appendingPathComponent("store")
    let store = try LocalAssetStore(root: storage)
    let id = try await store.install(manifest, from: assets.root)
    do {
        let lease = try await store.acquire(id)
        let installed = try KreaAssets.open(in: lease)
        #expect(installed.fingerprint == assets.fingerprint)
        #expect(installed.referenceFingerprint == assets.referenceFingerprint)
        try installed.verify()
    }
    let session = try KreaGenerationSession(installed: try await store.acquire(id))
    #expect(throws: AssetStoreError.busy) { try store.remove(id) }
    await session.unload()
    #expect(await session.state == .closed)
    try store.remove(id)
}

@Test func portableKreaManifestDetectsDeclarationDriftAndCopiedModelCorruption() async throws {
    let fixture = try BundleFixture(); defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    try fixture.edit("metadata.json") { $0["extra"] = "changed" }
    #expect(throws: AetherError.self) { try assets.installationManifest() }
    let updated = try KreaAssets.open(at: fixture.root)
    let manifest = try updated.installationManifest()
    try Data("wrong".utf8).write(to: assets.decoder.url.appendingPathComponent("main.mlirb"))
    let store = try LocalAssetStore(root: fixture.root.appendingPathComponent("store"))
    await #expect(throws: AssetStoreError.integrityMismatch("vaeDecoder.aimodel/main.mlirb")) {
        try await store.install(manifest, from: updated.root)
    }
}

@Test func installedKreaRejectsARepackagedOrIncompleteContract() async throws {
    let fixture = try BundleFixture(); defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    let manifest = try assets.installationManifest()
    let repackaged = try AssetManifest(assetIdentity: "another-model", executionContract: "other/1", files: manifest.files)
    let store = try LocalAssetStore(root: fixture.root.appendingPathComponent("store"))
    let id = try await store.install(repackaged, from: assets.root)
    let lease = try await store.acquire(id)
    #expect(throws: AetherError.self) { try KreaAssets.open(in: lease) }
    #expect(throws: AetherError.self) { try KreaGenerationSession(installed: lease) }
}


@Test func portableKreaPreservesOptionalTokenizerFilesAndRefusesInternalAliases() async throws {
    let fixture = try BundleFixture(); defer { fixture.cleanup() }
    let optional = ["config.json": "{\"model_type\":\"qwen3\"}", "chat_template.jinja": "{{ messages }}", "chat_template.json": "{}"]
    for (name, text) in optional { try Data(text.utf8).write(to: fixture.root.appendingPathComponent("tokenizer/" + name)) }
    let assets = try KreaAssets.open(at: fixture.root)
    let manifest = try assets.installationManifest()
    #expect(manifest.files.count == 25)
    let store = try LocalAssetStore(root: fixture.root.appendingPathComponent("store"))
    let id = try await store.install(manifest, from: assets.root)
    let lease = try await store.acquire(id)
    for (name, text) in optional {
        #expect(try String(contentsOf: lease.directory.appendingPathComponent("tokenizer/" + name), encoding: .utf8) == text)
    }
    let model = fixture.root.appendingPathComponent("textEncoder.aimodel")
    let real = fixture.root.appendingPathComponent("real.aimodel")
    try FileManager.default.moveItem(at: model, to: real)
    try FileManager.default.createSymbolicLink(at: model, withDestinationURL: real)
    let aliased = try KreaAssets.open(at: fixture.root)
    #expect(throws: AetherError.self) { try aliased.installationManifest() }
}

@Test func streamVerificationSpansChunksAndReportsFirstFailureInSerialOrder() throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    // Two complete 4 MiB chunks and a partial tail.
    var raw = Data(repeating: 0x5a, count: 8 << 20)
    raw.append(contentsOf: 0..<17)
    let stream = fixture.root.appendingPathComponent("denoiser2.aimodel")
    try raw.write(to: stream.appendingPathComponent("main.mlirb"))
    try Data(SHA256.hash(data: raw)).write(to: stream.appendingPathComponent("main.hash"))
    let assets = try KreaAssets.open(at: fixture.root)
    try assets.verify()
    raw[(4 << 20) + 3] ^= 1
    try raw.write(to: stream.appendingPathComponent("main.mlirb"))
    #expect(throws: AetherError.invalidModel("model hash mismatch: denoiser2.aimodel")) { try assets.verify() }
    // A later failure must not displace an earlier asset's error, however threads interleave.
    for name in ["denoiser3", "denoiser1"] {
        try Data("corrupt".utf8).write(to: fixture.root.appendingPathComponent("\(name).aimodel/main.mlirb"))
    }
    for _ in 0..<20 {
        #expect(throws: AetherError.invalidModel("model hash mismatch: denoiser1.aimodel")) { try assets.verify() }
    }
}

@Test func streamVerificationObservesCallerCancellation() async throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    let task = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        try assets.verify()
    }
    await #expect(throws: CancellationError.self) { try await task.value }
}
