import Foundation

/// Where a `ModelStore`'s entries live under its root, and the maintenance that keeps them tidy.
///
/// ```
/// <root>/<owner>/<name>/<revision>/   installed bundles (complete exactly when .aether/receipt.json is present)
/// <root>/.staging/<owner~name@revision@variant>/   resumable downloads
/// <root>/.locks/<owner~name@revision>   one advisory lock per install location (StoreLock)
/// <root>/.trash/<uuid>/   bundles and stages being deleted
/// ```
struct StoreLayout: Sendable {
    let root: URL

    var stagingRoot: URL { root.appending(path: ".staging", directoryHint: .isDirectory) }
    var locks: URL { root.appending(path: ".locks", directoryHint: .isDirectory) }
    var trash: URL { root.appending(path: ".trash", directoryHint: .isDirectory) }

    func bundle(for ref: ModelRef) -> URL { bundle(repo: ref.repo, revision: ref.revision) }

    func bundle(repo: String, revision: String) -> URL {
        InstalledLayout.appendingComponents(InstalledLayout.appendingComponents(root, repo), revision)
    }

    func staging(for ref: ModelRef) -> URL {
        let key = [ref.repo, ref.revision, ref.variant].joined(separator: "@").replacingOccurrences(of: "/", with: "~")
        return stagingRoot.appending(path: key)
    }

    func lock(for ref: ModelRef) -> URL { lock(repo: ref.repo, revision: ref.revision) }

    func lock(repo: String, revision: String) -> URL {
        locks.appending(path: "\(repo.replacingOccurrences(of: "/", with: "~"))@\(revision)")
    }

    /// `owner~name@revision@variant` → its parts.
    static func parseStagingKey(_ name: String) -> (repo: String, revision: String, variant: String)? {
        let parts = name.split(separator: "@", omittingEmptySubsequences: false)
        guard parts.count == 3, parts.allSatisfy({ !$0.isEmpty }) else { return nil }
        return (parts[0].replacingOccurrences(of: "~", with: "/"), String(parts[1]), String(parts[2]))
    }

    /// Moves `url` into the trash in one rename, then deletes it there: an interrupted delete never leaves a partial
    /// bundle at its install location.
    func discard(_ url: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: trash, withIntermediateDirectories: true)
        let aside = trash.appending(path: UUID().uuidString)
        try fm.moveItem(at: url, to: aside)
        try? fm.removeItem(at: aside)
    }

    /// See ``ModelStore/recoverAbandonedStages()``.
    func recover() -> Int {
        var removed = 0
        let fm = FileManager.default

        for name in Self.entries(trash) {
            if (try? fm.removeItem(at: trash.appending(path: name))) != nil { removed += 1 }
        }

        for name in Self.entries(stagingRoot) {
            guard let key = Self.parseStagingKey(name) else { continue }
            let stage = stagingRoot.appending(path: name)
            guard Self.isDirectory(stage) else { continue }
            guard let lock = try? StoreLock.attempt(lock(repo: key.repo, revision: key.revision), exclusive: true) else {
                continue  // being installed, removed or read
            }
            defer { withExtendedLifetime(lock) {} }
            guard let receipt = InstallReceipt.read(at: bundle(repo: key.repo, revision: key.revision)),
                  receipt.ref.variant == key.variant
            else { continue }  // an unfinished install: kept for resume
            if (try? discard(stage)) != nil { removed += 1 }
        }

        // <owner>/<name>/<revision> without a receipt. Only names that look like a pinned revision are considered.
        for owner in Self.entries(root) where !owner.hasPrefix(".") {
            let ownerDir = root.appending(path: owner)
            guard Self.isDirectory(ownerDir) else { continue }
            for name in Self.entries(ownerDir) where !name.hasPrefix(".") {
                let repoDir = ownerDir.appending(path: name)
                guard Self.isDirectory(repoDir) else { continue }
                for revision in Self.entries(repoDir) where Self.isPinnedRevision(revision) {
                    let revisionDir = repoDir.appending(path: revision)
                    guard Self.isDirectory(revisionDir), InstallReceipt.read(at: revisionDir) == nil,
                          let lock = try? StoreLock.attempt(lock(repo: "\(owner)/\(name)", revision: revision), exclusive: true)
                    else { continue }
                    defer { withExtendedLifetime(lock) {} }
                    guard InstallReceipt.read(at: revisionDir) == nil else { continue }
                    if (try? discard(revisionDir)) != nil { removed += 1 }
                }
            }
        }
        return removed
    }

    private static func entries(_ directory: URL) -> [String] {
        guard isDirectory(directory) else { return [] }
        return ((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []).sorted()
    }

    /// A real directory: never a symlink to one.
    private static func isDirectory(_ url: URL) -> Bool {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType) == .typeDirectory
    }

    private static func isPinnedRevision(_ name: String) -> Bool {
        name.utf8.count == 40 && name.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}
