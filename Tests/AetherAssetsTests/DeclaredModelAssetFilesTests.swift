import AetherAssets
import CryptoKit
import Foundation
import Testing

/// `verify(files:)` hashes exactly the files it's given (a cache claim's resolved files), and only inside the asset.
@Test func declaredAssetVerifiesTheGivenFilesInsideItsDirectory() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    let directory = root.appendingPathComponent("model.aimodel")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let stream = Data("stream".utf8), metadata = Data("{}".utf8)
    try stream.write(to: directory.appendingPathComponent("main.mlirb"))
    try Data(SHA256.hash(data: stream)).write(to: directory.appendingPathComponent("main.hash"))
    try metadata.write(to: directory.appendingPathComponent("metadata.json"))
    let hex = { (data: Data) in SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    let asset = DeclaredModelAsset(url: directory, declaredSHA256: hex(stream), metadataSHA256: hex(metadata))
    let files = try asset.files()
    #expect(files.map(\.lastPathComponent) == ["main.hash", "main.mlirb", "metadata.json"])
    try asset.verify(files: files, checkStop: {})
    // A file outside the asset directory, even with the right bytes, is refused.
    let outside = root.appendingPathComponent("main.mlirb")
    try stream.write(to: outside)
    #expect(throws: AssetStoreError.self) { try asset.verify(files: [files[0], outside, files[2]], checkStop: {}) }
    #expect(throws: AssetStoreError.self) { try asset.verify(files: [files[0], files[1]], checkStop: {}) }
    try DeclaredModelAsset.verifyConcurrently([asset, asset], files: [files, nil])
}
