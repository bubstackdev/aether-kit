import Foundation
import CryptoKit

extension Manifest {
    /// The SHA-256 (lowercase hex) of the manifest's canonical JSON (the decoded manifest re-encoded with sorted
    /// keys): the export's identity. Every manifest has one, and it changes with anything an export can change that
    /// matters to how its graphs run: the graph files and their hashes, precision, compute and reshape hints, the
    /// weights' provenance. (A manifest edit that changes nothing of that, such as a description, changes it too.)
    public var contentDigest: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        // A decoded manifest always re-encodes; the fallback only keeps this total.
        let data = (try? encoder.encode(self)) ?? Data(id.utf8)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
