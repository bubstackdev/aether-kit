import AetherKit
import AetherKrea
import CryptoKit
import Foundation
import Testing

func shapeRegistration(fixture: BundleFixture, assets: KreaAssets) throws -> URL {
    var paths: [String] = []
    var hashes: [String] = []
    for index in 0..<4 {
        let relative = "fixed\(index).aimodel"
        let url = fixture.root.appendingPathComponent(relative)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let bytes = Data("specialized \(index)".utf8)
        let digest = Data(SHA256.hash(data: bytes))
        try bytes.write(to: url.appendingPathComponent("main.mlirb"))
        try digest.write(to: url.appendingPathComponent("main.hash"))
        try Data("{\"assetVersion\":\"2.0\"}".utf8).write(to: url.appendingPathComponent("metadata.json"))
        paths.append(relative)
        hashes.append(digest.map { String(format: "%02x", $0) }.joined())
    }
    let json: [String: Any] = [
        "image_tokens": 1, "text_rows": 1, "source_sha256": assets.denoisers.map(\.declaredSHA256),
        "specialized_sha256": hashes, "segments": paths,
    ]
    let url = fixture.root.appendingPathComponent("shape.json")
    try JSONSerialization.data(withJSONObject: json).write(to: url)
    return url
}

@Test func specializationMatchesBothShapesAndSourceIdentity() throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let original = try KreaAssets.open(at: fixture.root)
    let registration = try shapeRegistration(fixture: fixture, assets: original)
    let assets = try original.registeringSpecialization(at: registration)
    let selected = try assets.denoiserSelection(imageTokens: 1, textRows: 1)
    #expect(selected.specialized)
    #expect(selected.assets[0].url.lastPathComponent == "fixed0.aimodel")
    #expect(try !assets.denoiserSelection(imageTokens: 2, textRows: 1).specialized)
    #expect(try !assets.denoiserSelection(imageTokens: 1, textRows: 2).specialized)
    #expect(try !original.denoiserSelection(imageTokens: 1, textRows: 1).specialized)
    #expect(throws: AetherError.self) { try assets.registeringSpecialization(at: registration) }
    try fixture.edit("shape.json") { $0["source_sha256"] = Array(repeating: String(repeating: "0", count: 64), count: 4) }
    #expect(throws: AetherError.self) { try original.registeringSpecialization(at: registration) }
}

@Test func specializationRejectsTamperingAndUnknownFields() throws {
    let fixture = try BundleFixture()
    defer { fixture.cleanup() }
    let assets = try KreaAssets.open(at: fixture.root)
    let registration = try shapeRegistration(fixture: fixture, assets: assets)
    try fixture.edit("shape.json") { $0["unknown"] = true }
    #expect(throws: AetherError.self) { try assets.registeringSpecialization(at: registration) }
    try fixture.edit("shape.json") { $0.removeValue(forKey: "unknown") }
    try Data("tampered".utf8).write(to: fixture.root.appendingPathComponent("fixed2.aimodel/main.mlirb"))
    #expect(throws: AetherError.self) { try assets.registeringSpecialization(at: registration) }
    try Data("tampered".utf8).write(to: assets.denoisers[0].url.appendingPathComponent("main.mlirb"))
    #expect(throws: AetherError.self) { try assets.registeringSpecialization(at: registration) }
}
