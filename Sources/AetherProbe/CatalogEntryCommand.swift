import AetherAssets
import AetherFlux2
import AetherKit
import AetherKrea
import AetherLanguage
import AetherSpeech
import Foundation

/// `catalog-entry FAMILY BUNDLE SOURCE [--name NAME]`: prints the pinned catalog entry an app prepares the bundle from
/// (docs/model-preparation.md). The manifest is the family's own installation contract, so the entry's installation
/// ID equals the one a local install of the same bundle gets. CPU only: model streams are not reread.
@available(macOS 27, iOS 27, *)
enum CatalogEntryCommand {
    static let usage = """
        aether-probe catalog-entry language|kokoro|flux2|krea BUNDLE --https BASE_URL [--name NAME]
        aether-probe catalog-entry language|kokoro|flux2|krea BUNDLE --huggingface OWNER/NAME@REVISION[/SUBDIRECTORY] [--name NAME]
        """

    static func run(_ arguments: [String]) throws -> Bool {
        guard arguments.first == "catalog-entry" else { return false }
        guard arguments.count == 5 || (arguments.count == 7 && arguments[5] == "--name") else {
            throw AetherError.invalidRequest("usage:\n" + usage)
        }
        let bundle = URL(fileURLWithPath: arguments[2])
        let manifest: AssetManifest
        switch arguments[1] {
        case "language": manifest = try LanguageModelManifest.installationManifest(in: bundle)
        case "kokoro": manifest = try KokoroAssets.open(at: bundle).installationManifest()
        case "flux2": manifest = try Flux2Assets.open(at: bundle).installationManifest()
        case "krea": manifest = try KreaAssets.open(at: bundle).installationManifest()
        default: throw AetherError.invalidRequest("unknown family \(arguments[1]); usage:\n" + usage)
        }
        let entry = try ModelCatalogEntry(manifest: manifest, source: source(arguments[3], arguments[4]),
                                          displayName: arguments.count == 7 ? arguments[6] : nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        FileHandle.standardOutput.write(try encoder.encode(entry) + Data("\n".utf8))
        return true
    }

    static func source(_ flag: String, _ value: String) throws -> ModelSource {
        switch flag {
        case "--https":
            guard let base = URL(string: value) else { throw AetherError.invalidRequest("invalid base URL") }
            return .https(base: base)
        case "--huggingface":
            // OWNER/NAME@REVISION[/SUBDIRECTORY]
            let halves = value.split(separator: "@", maxSplits: 1).map(String.init)
            guard halves.count == 2 else { throw AetherError.invalidRequest("expected OWNER/NAME@REVISION[/SUBDIRECTORY]") }
            let tail = halves[1].split(separator: "/", maxSplits: 1).map(String.init)
            return .huggingFace(repository: halves[0], revision: tail[0], subdirectory: tail.count == 2 ? tail[1] : nil)
        default:
            throw AetherError.invalidRequest("expected --https or --huggingface; usage:\n" + usage)
        }
    }
}
