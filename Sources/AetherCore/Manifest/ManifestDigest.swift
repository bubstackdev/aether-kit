import Foundation
import CryptoKit

extension Manifest {
    /// The SHA-256 (lowercase hex) of the manifest's canonical JSON (the decoded manifest re-encoded with sorted
    /// keys), its measurements left out: the export's identity. Every manifest has one, and it changes with anything
    /// an export can change that matters to how its graphs run: the graph files and their hashes, precision, compute
    /// and reshape hints, the weights' provenance. (A manifest edit that changes nothing of that, such as a
    /// description, changes it too.)
    ///
    /// The model's content only: the memory envelopes a gate measures on the export (`h3.envelopes`, MANIFEST.md
    /// §5.10) are not part of it, so recording or re-measuring them (on another chip, say) keeps the export's identity,
    /// and the bundle's (``InstalledBundle/contentIdentity()``), and the envelopes name the export they were measured
    /// on by this same digest. A manifest without envelopes has the digest it always had.
    public var contentDigest: String {
        var content = self
        content.h3?.envelopes = nil
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        // A decoded manifest always re-encodes; the fallback only keeps this total.
        let data = (try? encoder.encode(content)) ?? Data(id.utf8)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
