import CryptoKit
public import Foundation

// A bundle's status for a host to show (DESIGN.md §5): not prepared, ready, or blocked with a typed reason. Two depths
// of one check: `status` reads the manifest and stats every declared file (no model bytes are read), and
// `verifiedStatus` adds the SHA-256 of every declared file, with progress. Both work for every driver's bundles, from a
// store or from any folder.

/// Where a bundle stands. ``InstalledBundle/status(supportedDrivers:)`` decides it cheaply (sizes, no hashing);
/// ``InstalledBundle/verifiedStatus(supportedDrivers:progress:)`` decides it fully (every declared file's SHA-256).
public enum BundleStatus: Sendable, Hashable, CustomStringConvertible {
    /// Nothing is there yet: the folder is absent or empty.
    case notPrepared
    /// The manifest (when the bundle has one) decodes with a schema and driver this build runs, and every declared file
    /// is present at its declared size (and, from ``InstalledBundle/verifiedStatus(supportedDrivers:progress:)``, its
    /// declared SHA-256).
    case ready
    /// Something is there, but it can't run as it is.
    case blocked(BundleBlockReason)

    public var isReady: Bool { self == .ready }

    public var description: String {
        switch self {
        case .notPrepared: "not prepared"
        case .ready: "ready"
        case .blocked(let reason): "blocked: \(reason)"
        }
    }
}

/// Why a bundle is ``BundleStatus/blocked(_:)``. File paths are relative to the bundle's root.
public enum BundleBlockReason: Sendable, Hashable, CustomStringConvertible {
    /// The folder has files but no manifest, and nothing else declares its files (an interrupted export or copy).
    case incomplete
    /// Declared files are absent: weights, graphs, tables or any other file the reference or the manifest declares.
    /// `files` is sorted; `bytes` is their declared total.
    case missingFiles(files: [String], bytes: Int64)
    /// The manifest's `schema` is one this build doesn't read.
    case unsupportedSchema(found: Int, supported: Int)
    /// The manifest names a driver id (or a version of one) outside this build's driver catalog (MANIFEST.md §1).
    case unknownDriver(DriverID)
    /// The driver is a known id, but not among the drivers the caller can load (for example a `Registry`'s ids).
    case unsupportedDriver(DriverID)
    /// The manifest doesn't decode or validate, or disagrees with the reference it was opened with. `field` is the
    /// dotted manifest path (empty when the file isn't JSON at all).
    case invalidManifest(field: String, reason: String)
    /// A declared file is there but isn't the declared bytes: `expected` and `actual` are `size N` for a wrong size,
    /// SHA-256s for a wrong hash (found only by a verified status), and `actual` is `unreadable` for a file that can't
    /// be read.
    case corruptFile(file: String, expected: String, actual: String)

    public var description: String {
        switch self {
        case .incomplete:
            "the bundle has no manifest"
        case .missingFiles(let files, let bytes):
            "\(files.count) declared file\(files.count == 1 ? "" : "s") (\(bytes) bytes) missing, first '\(files.first ?? "")'"
        case .unsupportedSchema(let found, let supported):
            "manifest schema \(found); this build reads \(supported)"
        case .unknownDriver(let driver):
            "unknown driver '\(driver)'"
        case .unsupportedDriver(let driver):
            "driver '\(driver)' is not supported here"
        case .invalidManifest(let field, let reason):
            "invalid manifest field '\(field)': \(reason)"
        case .corruptFile(let file, let expected, let actual):
            "'\(file)' is not the declared file: expected \(expected), found \(actual)"
        }
    }
}

extension InstalledBundle {
    /// The bundle's status without reading model bytes: whether the folder exists and holds anything, the manifest's
    /// schema and driver, and that every declared file exists at its declared size. Declared files are those of the
    /// reference (a catalog install) and of the manifest's `assets.*.files` (a local export), each once; a file both
    /// declare differently blocks as ``BundleBlockReason/invalidManifest(field:reason:)``.
    ///
    /// `supportedDrivers` are the drivers the caller can load (`Set(registry.ids)`); nil checks only that the driver
    /// is one this build knows. A bundle without a manifest whose reference declares files (a zoo install, whose
    /// manifest the first load synthesizes) is judged by those files and the reference's driver.
    ///
    /// Throws only ``AetherError/invalidReference(field:reason:)``, for a reference with an unsafe path.
    public func status(supportedDrivers: Set<DriverID>? = nil) throws(AetherError) -> BundleStatus {
        try ref.validateForStore()
        return switch BundleInspection.inspect(url: url, ref: ref, supportedDrivers: supportedDrivers) {
        case .settled(let status): status
        case .files: .ready
        }
    }

    /// ``status(supportedDrivers:)``, then the SHA-256 of every declared file: the full check, for a bundle opened from
    /// any folder. Tens of GB for a large model: run it off the main actor. Nothing is hashed when the cheap check
    /// already blocks, and hashing stops at the first file that differs.
    ///
    /// `progress` is called as bytes are read (about every 64 MiB and at each file's end). Throws `CancellationError`
    /// between reads once the task is cancelled, and ``AetherError/invalidReference(field:reason:)`` for a reference
    /// with an unsafe path.
    public func verifiedStatus(
        supportedDrivers: Set<DriverID>? = nil, progress: ((BundleVerificationProgress) -> Void)? = nil
    ) throws -> BundleStatus {
        try ref.validateForStore()
        return try BundleInspection.verify(
            BundleInspection.inspect(url: url, ref: ref, supportedDrivers: supportedDrivers), in: url, progress: progress)
    }

    /// The status of a folder meant to hold a bundle outside any store (a local export, a restored archive), before
    /// there is a reference for it: ``BundleStatus/notPrepared`` when the folder is absent or empty,
    /// ``BundleBlockReason/incomplete`` when it has files but no manifest, and otherwise
    /// ``status(supportedDrivers:)`` with the manifest's own `assets.*.files` as the declared files.
    public static func status(ofLocalBundleAt url: URL, supportedDrivers: Set<DriverID>? = nil) -> BundleStatus {
        switch BundleInspection.inspect(url: url, ref: nil, supportedDrivers: supportedDrivers) {
        case .settled(let status): status
        case .files: .ready
        }
    }

    /// ``status(ofLocalBundleAt:supportedDrivers:)``, then the SHA-256 of every file the manifest declares, as
    /// ``verifiedStatus(supportedDrivers:progress:)`` does. Throws `CancellationError` once the task is cancelled.
    public static func verifiedStatus(
        ofLocalBundleAt url: URL, supportedDrivers: Set<DriverID>? = nil,
        progress: ((BundleVerificationProgress) -> Void)? = nil
    ) throws -> BundleStatus {
        try BundleInspection.verify(
            BundleInspection.inspect(url: url, ref: nil, supportedDrivers: supportedDrivers), in: url, progress: progress)
    }
}

/// One declared file: its path under the bundle root, size and SHA-256 (lowercase).
struct DeclaredFile: Sendable, Hashable {
    var local: String
    var size: Int64
    var sha256: String
}

/// A value, or why the bundle is blocked.
enum Checked<Value> {
    case success(Value)
    case failure(BundleBlockReason)
}

enum BundleInspection {
    enum Outcome {
        /// Decided without reading any file's bytes.
        case settled(BundleStatus)
        /// Every declared file is present at its size: ready once their hashes agree.
        case files([DeclaredFile])
    }

    static func inspect(url: URL, ref: ModelRef?, supportedDrivers: Set<DriverID>?) -> Outcome {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else { return .settled(.notPrepared) }
        guard isDirectory.boolValue else { return .settled(.blocked(.incomplete)) }
        let entries = (try? fm.contentsOfDirectory(atPath: url.path)) ?? []
        guard entries.contains(where: { $0 != ".DS_Store" }) else { return .settled(.notPrepared) }

        let manifestURL = Manifest.url(inBundle: url)
        var manifest: Manifest?
        if fm.fileExists(atPath: manifestURL.path) {
            let data: Data
            do { data = try Data(contentsOf: manifestURL) } catch {
                return .settled(.blocked(.corruptFile(
                    file: "\(Manifest.directory)/\(Manifest.fileName)", expected: "a readable manifest", actual: "unreadable")))
            }
            switch decode(data, supportedDrivers: supportedDrivers) {
            case .failure(let reason): return .settled(.blocked(reason))
            case .success(let decoded): manifest = decoded
            }
        } else if ref?.assets.allSatisfy({ $0.files.isEmpty }) ?? true {
            return .settled(.blocked(.incomplete))
        }

        guard let driver = manifest?.driver ?? ref?.driver else { return .settled(.blocked(.incomplete)) }
        if let ref, let manifest, ref.driver != manifest.driver {
            return .settled(.blocked(.invalidManifest(
                field: "driver", reason: "the manifest names \(manifest.driver); the reference names \(ref.driver)")))
        }
        if manifest == nil, let reason = driverReason(driver, supportedDrivers: supportedDrivers) {
            return .settled(.blocked(reason))
        }

        let files: [DeclaredFile]
        switch declaredFiles(ref: ref, manifest: manifest) {
        case .failure(let reason): return .settled(.blocked(reason))
        case .success(let list): files = list
        }
        var missing: [DeclaredFile] = []
        var wrongSize: (DeclaredFile, Int64)?
        for file in files {
            guard let size = InstalledBundle.fileSize(url.appending(path: file.local)) else {
                missing.append(file)
                continue
            }
            if size != file.size, wrongSize == nil { wrongSize = (file, size) }
        }
        if !missing.isEmpty {
            return .settled(.blocked(.missingFiles(
                files: missing.map(\.local).sorted(), bytes: saturatingSum(missing.map(\.size)))))
        }
        if let (file, size) = wrongSize {
            return .settled(.blocked(.corruptFile(file: file.local, expected: "size \(file.size)", actual: "size \(size)")))
        }
        return .files(files)
    }

    /// The schema first, then the driver, then the whole manifest, so a newer manifest is reported by what makes it
    /// newer rather than by the first field this build can't decode.
    static func decode(_ data: Data, supportedDrivers: Set<DriverID>?) -> Checked<Manifest> {
        struct Head: Decodable {
            var schema: Int?
            var driver: String?
        }
        guard let head = try? JSONDecoder().decode(Head.self, from: data) else {
            return .failure(.invalidManifest(field: "", reason: "not a JSON object with an integer schema"))
        }
        if let schema = head.schema, schema != Manifest.currentSchema {
            return .failure(.unsupportedSchema(found: schema, supported: Manifest.currentSchema))
        }
        if let text = head.driver, let driver = try? DriverID(parsing: text),
           let reason = driverReason(driver, supportedDrivers: supportedDrivers) {
            return .failure(reason)
        }
        do {
            return .success(try Manifest.decode(data))
        } catch .manifestInvalid(let field, let reason) {
            return .failure(.invalidManifest(field: field, reason: reason))
        } catch {
            return .failure(.invalidManifest(field: "", reason: error.description))
        }
    }

    static func driverReason(_ driver: DriverID, supportedDrivers: Set<DriverID>?) -> BundleBlockReason? {
        if !DriverCatalog.drivers.contains(driver) { return .unknownDriver(driver) }
        if let supportedDrivers, !supportedDrivers.contains(driver) { return .unsupportedDriver(driver) }
        return nil
    }

    /// Every file the reference and the manifest declare, once per path under the bundle root, in declaration order
    /// (the reference's first). The manifest's assets are laid out as the reference lays out the asset of the same
    /// name; an asset the reference doesn't have (or every asset, without a reference) as the store would: its
    /// subtree under the root, or `.aether/assets/<name>/` for a foreign repo.
    static func declaredFiles(ref: ModelRef?, manifest: Manifest?) -> Checked<[DeclaredFile]> {
        var list: [DeclaredFile] = []
        var index: [String: Int] = [:]
        let slash = URL(fileURLWithPath: "/")
        if let ref {
            for item in InstalledBundle.verificationList(ref) where index[item.local] == nil {
                index[item.local] = list.count
                list.append(DeclaredFile(local: item.local, size: item.file.size, sha256: item.file.sha256.lowercased()))
            }
        }
        for (name, asset) in (manifest?.assets ?? [:]).sorted(by: { $0.key < $1.key }) {
            let root: URL
            if let ref, let refAsset = ref.asset(name) {
                root = InstalledLayout.assetRoot(refAsset, mainRepo: ref.repo, in: slash)
            } else if asset.repo != nil, asset.repo != ref?.repo {
                root = slash.appending(path: Manifest.directory).appending(path: "assets").appending(path: name)
            } else {
                root = asset.subtree.isEmpty ? slash : InstalledLayout.appendingComponents(slash, asset.subtree)
            }
            for (i, file) in (asset.files ?? []).enumerated() {
                let local = String(root.appending(path: file.path).path.dropFirst())
                let declared = DeclaredFile(local: local, size: file.bytes, sha256: file.sha256.lowercased())
                if let existing = index[local] {
                    guard list[existing].size == declared.size, list[existing].sha256 == declared.sha256 else {
                        return .failure(.invalidManifest(
                            field: "assets.\(name).files.\(i)",
                            reason: "declares '\(local)' as size \(declared.size), sha256 \(declared.sha256); the reference as size \(list[existing].size), sha256 \(list[existing].sha256)"))
                    }
                    continue
                }
                index[local] = list.count
                list.append(declared)
            }
        }
        return .success(list)
    }

    /// Hashes `outcome`'s files (when it has any to hash) and returns the status they give.
    static func verify(_ outcome: Outcome, in root: URL, progress: ((BundleVerificationProgress) -> Void)?) throws -> BundleStatus {
        guard case .files(let files) = outcome else {
            if case .settled(let status) = outcome { return status }
            return .ready
        }
        if let (file, actual) = try firstMismatch(files, in: root, progress: progress) {
            return .blocked(.corruptFile(file: file.local, expected: file.sha256, actual: actual))
        }
        return .ready
    }

    /// Reads `files` in order and returns the first whose SHA-256 differs, with what was found (`unreadable` for a
    /// file that can't be opened or read). Progress about every 64 MiB and at each file's end. `CancellationError`
    /// between reads once the task is cancelled.
    static func firstMismatch(
        _ files: [DeclaredFile], in root: URL, progress: ((BundleVerificationProgress) -> Void)?
    ) throws -> (DeclaredFile, String)? {
        let total = saturatingSum(files.map(\.size))
        var completed: Int64 = 0
        var reported: Int64 = 0
        progress?(BundleVerificationProgress(completedBytes: 0, totalBytes: total, file: nil))
        for file in files {
            try Task.checkCancellation()
            guard let handle = try? FileHandle(forReadingFrom: root.appending(path: file.local)) else {
                return (file, "unreadable")
            }
            defer { try? handle.close() }
            var hasher = SHA256()
            do {
                try FileChunks.forEach(handle) { chunk in
                    try Task.checkCancellation()
                    hasher.update(data: chunk)
                    completed += Int64(chunk.count)
                    if completed - reported >= 64 << 20 {
                        reported = completed
                        progress?(BundleVerificationProgress(completedBytes: completed, totalBytes: total, file: file.local))
                    }
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                return (file, "unreadable")
            }
            let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            guard actual == file.sha256 else { return (file, actual) }
            reported = completed
            progress?(BundleVerificationProgress(completedBytes: completed, totalBytes: total, file: file.local))
        }
        return nil
    }
}
