import CryptoKit
import Foundation
import Synchronization
import Testing
@testable import AetherCore

/// A bundle's status (not prepared, ready, blocked with a reason), cheap and verified, on synthetic bundles: each
/// driver fixture's manifest with its file list replaced by small files written next to it.
@Suite("Bundle status")
struct BundleStatusTests {
    /// One manifest fixture per driver family, so the check is shown to be the same for every driver.
    static let fixtures = [
        "h3/minimax-h3.json", "krea2/vanilla.json", "flux2/flux2-klein-4b.json", "kokoro/aether.json",
        "sd-turbo/aether.json", "m4-ios/clip-vit-b32/aether.json", "m4-ios/yolox-s/aether.json",
    ]

    static let files: [(path: String, data: Data)] = [
        ("models/body.aimodel/main.mlirb", Data(repeating: 1, count: 3 << 20)),
        ("weights/body/block-00.bin", Data(repeating: 2, count: 1 << 20)),
        ("tokenizer/tokenizer.json", Data("{\"model\": {}}".utf8)),
    ]

    static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    static func declarations(_ files: [(path: String, data: Data)] = files) -> [Manifest.Asset.File] {
        files.map { Manifest.Asset.File(path: $0.path, bytes: Int64($0.data.count), sha256: sha256($0.data)) }
    }

    /// Writes `files` and the fixture manifest (its main asset declaring `declared`) under `root`.
    @discardableResult
    static func writeLocal(
        at root: URL, fixture: String = "h3/minimax-h3.json", files: [(path: String, data: Data)] = files,
        declared: [Manifest.Asset.File]? = nil, edit: (inout Manifest) -> Void = { _ in }
    ) throws -> Manifest {
        var manifest = try Manifest.decode(Fixtures.data(fixture))
        manifest.assets["main"]?.files = declared ?? declarations(files)
        edit(&manifest)
        for file in files {
            let url = root.appending(path: file.path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try file.data.write(to: url)
        }
        try manifest.write(toBundle: root)
        return manifest
    }

    /// A reference as a local bundle's host builds it: the manifest's driver, no declared files.
    static func handMadeRef(_ manifest: Manifest) -> ModelRef {
        ModelRef(
            catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: "macos", driver: manifest.driver,
            assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
    }

    static func writeManifestText(_ text: String, at root: URL) throws {
        let url = Manifest.url(inBundle: root)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    // MARK: - Not prepared, incomplete, ready

    @Test func anAbsentOrEmptyFolderIsNotPrepared() async throws {
        try await withTemporaryDirectory { dir in
            #expect(InstalledBundle.status(ofLocalBundleAt: dir.appending(path: "absent")) == .notPrepared)
            #expect(InstalledBundle.status(ofLocalBundleAt: dir) == .notPrepared)
            try Data().write(to: dir.appending(path: ".DS_Store"))
            #expect(InstalledBundle.status(ofLocalBundleAt: dir) == .notPrepared)
            #expect(try InstalledBundle.verifiedStatus(ofLocalBundleAt: dir) == .notPrepared)
        }
    }

    @Test func filesWithoutAManifestAreIncomplete() async throws {
        try await withTemporaryDirectory { dir in
            let weights = dir.appending(path: "weights/block-00.bin")
            try FileManager.default.createDirectory(at: weights.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([1, 2, 3]).write(to: weights)
            #expect(InstalledBundle.status(ofLocalBundleAt: dir) == .blocked(.incomplete))
            let manifest = try Manifest.decode(Fixtures.data("h3/minimax-h3.json"))
            let bundle = InstalledBundle(ref: Self.handMadeRef(manifest), url: dir, isStale: false)
            #expect(try bundle.status() == .blocked(.incomplete))
        }
    }

    @Test(arguments: fixtures)
    func everyDriversCompleteBundleIsReady(fixture: String) async throws {
        try await withTemporaryDirectory { dir in
            let manifest = try Self.writeLocal(at: dir, fixture: fixture)
            #expect(InstalledBundle.status(ofLocalBundleAt: dir) == .ready)
            #expect(InstalledBundle.status(ofLocalBundleAt: dir, supportedDrivers: [manifest.driver]) == .ready)
            #expect(try InstalledBundle.verifiedStatus(ofLocalBundleAt: dir) == .ready)
            let bundle = InstalledBundle(ref: Self.handMadeRef(manifest), url: dir, isStale: false)
            #expect(try bundle.status() == .ready)
            #expect(try bundle.verifiedStatus() == .ready)
        }
    }

    // MARK: - Missing and corrupt files

    @Test func missingDeclaredFilesBlockWithTheirPathsAndBytes() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir)
            for path in ["weights/body/block-00.bin", "models/body.aimodel/main.mlirb"] {
                try FileManager.default.removeItem(at: dir.appending(path: path))
            }
            let expected = BundleStatus.blocked(.missingFiles(
                files: ["models/body.aimodel/main.mlirb", "weights/body/block-00.bin"], bytes: (3 << 20) + (1 << 20)))
            #expect(InstalledBundle.status(ofLocalBundleAt: dir) == expected)
            let hashed = Mutex(false)
            #expect(try InstalledBundle.verifiedStatus(ofLocalBundleAt: dir) { _ in hashed.withLock { $0 = true } } == expected)
            #expect(hashed.withLock { !$0 }, "nothing is hashed once the cheap check blocks")
        }
    }

    @Test func aDirectoryWhereAFileIsDeclaredIsMissing() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir)
            let path = "tokenizer/tokenizer.json"
            try FileManager.default.removeItem(at: dir.appending(path: path))
            try FileManager.default.createDirectory(at: dir.appending(path: path), withIntermediateDirectories: true)
            guard case .blocked(.missingFiles(let files, _)) = InstalledBundle.status(ofLocalBundleAt: dir) else {
                Issue.record("expected missing files")
                return
            }
            #expect(files == [path])
        }
    }

    @Test func aWrongSizeIsCorruptWithoutHashing() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir)
            try Data(repeating: 2, count: 10).write(to: dir.appending(path: "weights/body/block-00.bin"))
            let expected = BundleStatus.blocked(.corruptFile(
                file: "weights/body/block-00.bin", expected: "size \(1 << 20)", actual: "size 10"))
            #expect(InstalledBundle.status(ofLocalBundleAt: dir) == expected)
            #expect(try InstalledBundle.verifiedStatus(ofLocalBundleAt: dir) == expected)
        }
    }

    @Test func aChangedByteIsFoundOnlyByTheVerifiedStatus() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir)
            var altered = Self.files[0].data
            altered[altered.count - 1] = 9
            try altered.write(to: dir.appending(path: Self.files[0].path))
            #expect(InstalledBundle.status(ofLocalBundleAt: dir) == .ready, "status reads sizes, not bytes")
            #expect(try InstalledBundle.verifiedStatus(ofLocalBundleAt: dir) == .blocked(.corruptFile(
                file: Self.files[0].path, expected: Self.sha256(Self.files[0].data), actual: Self.sha256(altered))))
        }
    }

    @Test func theVerifiedStatusReportsProgressOverEveryDeclaredFile() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir)
            let seen = Mutex<[BundleVerificationProgress]>([])
            #expect(try InstalledBundle.verifiedStatus(ofLocalBundleAt: dir) { p in seen.withLock { $0.append(p) } } == .ready)
            let updates = seen.withLock { $0 }
            let total = Int64(Self.files.reduce(0) { $0 + $1.data.count })
            #expect(updates.first == BundleVerificationProgress(completedBytes: 0, totalBytes: total, file: nil))
            #expect(updates.last?.completedBytes == total && updates.last?.fractionCompleted == 1)
            #expect(Set(updates.compactMap(\.file)) == Set(Self.files.map(\.path)))
            #expect(zip(updates, updates.dropFirst()).allSatisfy { $0.completedBytes <= $1.completedBytes })
        }
    }

    @Test func theVerifiedStatusStopsOnceCancelled() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir)
            let task = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try InstalledBundle.verifiedStatus(ofLocalBundleAt: dir)
            }
            await #expect(throws: CancellationError.self) { try await task.value }
        }
    }

    // MARK: - Versions and drivers

    @Test func aNewerSchemaIsUnsupported() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir) { $0.schema = 2 }
            #expect(InstalledBundle.status(ofLocalBundleAt: dir) == .blocked(.unsupportedSchema(found: 2, supported: 1)))
        }
    }

    @Test func aNewerDriverVersionIsUnknown() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir) { $0.driver = DriverID(validated: "minimax-h3@2") }
            #expect(InstalledBundle.status(ofLocalBundleAt: dir) == .blocked(.unknownDriver(DriverID(validated: "minimax-h3@2"))))
        }
    }

    @Test func aDriverTheCallerCantLoadIsUnsupported() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir)
            #expect(InstalledBundle.status(ofLocalBundleAt: dir, supportedDrivers: [.krea2])
                == .blocked(.unsupportedDriver(.minimaxH3)))
            #expect(InstalledBundle.status(ofLocalBundleAt: dir, supportedDrivers: [.krea2, .minimaxH3]) == .ready)
        }
    }

    @Test func aMalformedManifestNamesItsField() async throws {
        try await withTemporaryDirectory { dir in
            try Self.writeLocal(at: dir)
            try Self.writeManifestText("{\"schema\": 1, \"driver\": \"minimax-h3@1\", \"id\"", at: dir)
            #expect(InstalledBundle.status(ofLocalBundleAt: dir)
                == .blocked(.invalidManifest(field: "", reason: "not a JSON object with an integer schema")))

            var manifest = try Manifest.decode(Fixtures.data("h3/minimax-h3.json"))
            manifest.task = ""
            try Self.writeManifestText(String(decoding: try manifest.encoded(), as: UTF8.self), at: dir)
            guard case .blocked(.invalidManifest(let field, _)) = InstalledBundle.status(ofLocalBundleAt: dir) else {
                Issue.record("expected an invalid manifest")
                return
            }
            #expect(field == "task")
        }
    }

    // MARK: - With a reference

    @Test func aReferenceForAnotherDriverIsRefused() async throws {
        try await withTemporaryDirectory { dir in
            let manifest = try Self.writeLocal(at: dir)
            var ref = Self.handMadeRef(manifest)
            ref.driver = .krea2
            guard case .blocked(.invalidManifest(let field, _)) = try InstalledBundle(ref: ref, url: dir, isStale: false).status()
            else {
                Issue.record("expected a driver mismatch")
                return
            }
            #expect(field == "driver")
        }
    }

    /// A catalog install: the reference declares the files (and the manifest itself); the manifest declares the same
    /// files. Each is checked once.
    static func catalogRef(_ manifest: Manifest, at root: URL, extra: [CatalogFile] = []) throws -> ModelRef {
        let manifestData = try Data(contentsOf: Manifest.url(inBundle: root))
        let files = files.map { CatalogFile(path: $0.path, size: Int64($0.data.count), sha256: sha256($0.data)) }
            + [CatalogFile(path: ".aether/aether.json", size: Int64(manifestData.count), sha256: sha256(manifestData))] + extra
        return ModelRef(
            catalogID: manifest.id, repo: "owner/model", revision: "r1", variant: "macos", driver: manifest.driver,
            assets: [.init(name: "main", repo: "owner/model", revision: "r1", subtree: "", files: files)])
    }

    @Test func aCatalogInstallChecksTheReferencesFilesAndTheManifestsOnce() async throws {
        try await withTemporaryDirectory { dir in
            let manifest = try Self.writeLocal(at: dir)
            let license = Data("license".utf8)
            try license.write(to: dir.appending(path: "LICENSE"))
            let ref = try Self.catalogRef(manifest, at: dir, extra: [
                CatalogFile(path: "LICENSE", size: Int64(license.count), sha256: Self.sha256(license)),
            ])
            let bundle = InstalledBundle(ref: ref, url: dir, isStale: false)
            #expect(try bundle.status() == .ready)
            let seen = Mutex<[BundleVerificationProgress]>([])
            #expect(try bundle.verifiedStatus { p in seen.withLock { $0.append(p) } } == .ready)
            let total = seen.withLock { $0.first?.totalBytes }
            #expect(total == ref.bytes, "a file both declare is read once")

            try FileManager.default.removeItem(at: dir.appending(path: "LICENSE"))
            #expect(try bundle.status() == .blocked(.missingFiles(files: ["LICENSE"], bytes: Int64(license.count))))
        }
    }

    @Test func aFileTheReferenceAndManifestDeclareDifferentlyIsRefused() async throws {
        try await withTemporaryDirectory { dir in
            var declared = Self.declarations()
            declared[1].sha256 = String(repeating: "0", count: 64)
            let manifest = try Self.writeLocal(at: dir, declared: declared)
            let bundle = InstalledBundle(ref: try Self.catalogRef(manifest, at: dir), url: dir, isStale: false)
            guard case .blocked(.invalidManifest(let field, _)) = try bundle.status() else {
                Issue.record("expected conflicting declarations")
                return
            }
            #expect(field == "assets.main.files.1")
        }
    }

    @Test func aZooInstallWithoutAManifestIsJudgedByItsReference() async throws {
        try await withTemporaryDirectory { dir in
            let weights = Data(repeating: 4, count: 1024)
            try weights.write(to: dir.appending(path: "model.bin"))
            let ref = ModelRef(
                catalogID: "zoo", repo: "owner/zoo", revision: "r1", variant: "macos", driver: .decoderLM,
                assets: [.init(name: "main", repo: "owner/zoo", revision: "r1", subtree: "", files: [
                    CatalogFile(path: "model.bin", size: 1024, sha256: Self.sha256(weights)),
                    CatalogFile(path: "metadata.json", size: 2, sha256: Self.sha256(Data("{}".utf8))),
                ])])
            let bundle = InstalledBundle(ref: ref, url: dir, isStale: false)
            #expect(try bundle.status() == .blocked(.missingFiles(files: ["metadata.json"], bytes: 2)))
            try Data("{}".utf8).write(to: dir.appending(path: "metadata.json"))
            #expect(try bundle.status() == .ready)
            #expect(try bundle.verifiedStatus() == .ready)
            #expect(try bundle.status(supportedDrivers: [.krea2]) == .blocked(.unsupportedDriver(.decoderLM)))
        }
    }

    @Test func anUnsafeReferenceThrows() async throws {
        let ref = ModelRef(
            catalogID: "x", repo: "owner/model", revision: "..", variant: "macos", driver: .krea2,
            assets: [.init(name: "main", repo: "owner/model", revision: "r1", subtree: "", files: [])])
        #expect(throws: AetherError.self) { try InstalledBundle(ref: ref, url: URL(filePath: "/nonexistent"), isStale: false).status() }
    }
}
