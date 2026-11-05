import Foundation
import Testing
@testable import AetherCore

@Suite("Catalog and ModelRef")
struct CatalogTests {
    @Test func generatedSwiftMatchesCatalogJSON() throws {
        let json = try Data(contentsOf: Fixtures.repoRoot.appending(path: "catalog/catalog.json"))
        let decoded = try Catalog.decode(json)
        #expect(decoded == Catalog.builtin, "run scripts/gen-catalog emit")
    }

    @Test func builtinEntriesAreComplete() throws {
        for entry in Catalog.builtin.entries {
            #expect(entry.revision.count == 40, "\(entry.id): pin a full commit SHA")
            #expect(!entry.variants.isEmpty)
            for (key, variant) in entry.variants {
                #expect(!variant.files.isEmpty, "\(entry.id)[\(key)]")
                #expect(variant.bytes == variant.files.reduce(0) { $0 + $1.size })
                for file in variant.files {
                    #expect(file.sha256.count == 64 && file.sha256.allSatisfy(\.isHexDigit), "\(file.path)")
                }
            }
        }
        #expect(Set(Catalog.builtin.entries.map(\.id)) == ["qwen3-0.6b", "qwen3.5-0.8b"])
    }

    @Test func unknownIDIsReported() {
        #expect {
            _ = try Catalog.builtin.entry("nope")
        } throws: { error in
            if case AetherError.unknownModel(let id) = error { return id == "nope" }
            return false
        }
    }

    static let entry = CatalogEntry(
        id: "toy", task: "text-generation", driver: .decoderLM, repo: "owner/toy",
        revision: String(repeating: "a", count: 40), license: "mit",
        variants: [
            "macos": CatalogVariant(subtree: "macos", bytes: 0, files: []),
            "ios": CatalogVariant(subtree: "ios", bytes: 0, files: []),
            "ios-h18p": CatalogVariant(subtree: "ios-h18p", bytes: 0, files: [], architectures: ["h18p"]),
            // A variant keyed by architecture but restricted to others is never chosen.
            "ios-h16g": CatalogVariant(subtree: "ios-h16g", bytes: 0, files: [], architectures: ["h99"]),
        ])

    @Test(arguments: [
        (Device.Platform.iOS, "h18p", "ios-h18p"),
        (.iOS, "h17p", "ios"),
        (.iOS, "h16g", "ios"),
        (.macOS, "h17c", "macos"),
    ])
    func resolvesVariantForFakedDevice(_ platform: Device.Platform, _ arch: String, _ expected: String) throws {
        let device = Device(platform: platform, architecture: arch, physicalMemory: 8 << 30)
        let (ref, _) = try ModelRef.resolve(Self.entry, for: device)
        #expect(ref.variant == expected)
        #expect(ref.subtree == expected)
        #expect(ref.repo == "owner/toy" && ref.catalogID == "toy")
    }

    @Test func noVariantForPlatformIsUnsupportedDevice() {
        var entry = Self.entry
        entry.variants = ["ios": entry.variants["ios"]!]
        #expect {
            _ = try ModelRef.resolve(entry, for: Device(platform: .macOS, architecture: "h17c"))
        } throws: { error in
            if case AetherError.unsupportedDevice(let id, let platform, let arch) = error {
                return id == "toy" && platform == "macos" && arch == "h17c"
            }
            return false
        }
    }

    @Test func builtinResolvesOnMac() throws {
        let (ref, variant) = try ModelRef.resolve("qwen3.5-0.8b", for: Device(platform: .macOS, architecture: "h17c"))
        #expect(ref.subtree == "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_block32_sym")
        #expect(variant.files.contains { $0.path.hasSuffix("main.mlirb") })
    }
}
