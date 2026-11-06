import Foundation
import Testing
@testable import AetherCore

@Suite("Catalog, overrides and ModelRef")
struct CatalogTests {
    @Test func generatedSwiftMatchesCatalogJSON() throws {
        let json = try Data(contentsOf: Fixtures.repoRoot.appending(path: "catalog/catalog.json"))
        #expect(try Catalog.decode(json) == Catalog.builtin, "run scripts/gen-catalog emit")
    }

    @Test func generatedOverridesMatchTheTable() throws {
        let json = try Data(contentsOf: Fixtures.repoRoot.appending(path: "catalog/zoo-overrides.json"))
        #expect(try ZooOverrides.decode(json) == ZooOverrides.builtin, "run scripts/gen-catalog emit")
    }

    @Test func everyOverrideIsPinnedToItsCatalogRevision() throws {
        for (id, item) in ZooOverrides.builtin.entries {
            let entry = try Catalog.builtin.entry(id)
            #expect(item.revision == entry.revision, "\(id)")
            for key in item.variants.keys where key != "*" {
                #expect(entry.variants[key] != nil, "\(id): override for unknown variant \(key)")
            }
        }
    }

    @Test func builtinEntriesAreComplete() throws {
        for entry in Catalog.builtin.entries {
            #expect(entry.revision.count == 40, "\(entry.id): pin a full commit SHA")
            #expect(!entry.variants.isEmpty)
            for (key, variant) in entry.variants {
                #expect(variant.assets["main"] != nil, "\(entry.id)[\(key)]")
                #expect(DriverCatalog.drivers.contains(variant.driver), "\(entry.id)[\(key)] \(variant.driver)")
                for (name, asset) in variant.assets {
                    #expect(!asset.files.isEmpty, "\(entry.id)[\(key)].\(name)")
                    #expect(asset.repo == nil || asset.revision?.count == 40, "\(entry.id)[\(key)].\(name): pin the foreign repo")
                    for file in asset.files {
                        #expect(file.sha256.count == 64 && file.sha256.allSatisfy(\.isHexDigit), "\(file.path)")
                    }
                }
            }
        }
        #expect(Set(Catalog.builtin.entries.map(\.id)) == Set(Fixtures.goldenIDs + ["mistral-7b-v0.3", "sd-turbo", "flux2-klein-4b", "kokoro-82m"]))
    }

    /// The generative exports (aether-models): SD-Turbo is the one image model on iOS in 0.1.0; FLUX.2 is macOS-only.
    @Test func generativeEntriesResolvePerPlatform() throws {
        let mac = Device(platform: .macOS, architecture: "h17c")
        let phone = Device(platform: .iOS, architecture: "h18p")
        for device in [mac, phone] {
            let ref = try ModelRef.resolve("sd-turbo", for: device)
            #expect(ref.asset("main")?.files.contains { $0.path == ".aether/aether.json" } == true)
        }
        // iOS takes the mixed-precision export (fp32 text encoder, fp16 U-Net and VAE; docs/DEVICE.md), a pinned repo.
        let phoneTurbo = try #require(try ModelRef.resolve("sd-turbo", for: phone).asset("main"))
        #expect(phoneTurbo.repo == "aether-models/aether-kit-sd-turbo-ios" && phoneTurbo.revision.count == 40)
        #expect(try ModelRef.resolve("sd-turbo", for: mac).asset("main")?.repo == "aether-models/aether-kit-sd-turbo")
        for device in [mac, phone] { _ = try ModelRef.resolve("kokoro-82m", for: device) }
        _ = try ModelRef.resolve("flux2-klein-4b", for: mac)
        #expect(throws: AetherError.self) { _ = try ModelRef.resolve("flux2-klein-4b", for: phone) }
    }

    @Test func mistralTokenizerIsAPinnedForeignAsset() throws {
        let ref = try ModelRef.resolve("mistral-7b-v0.3", for: Device(platform: .macOS, architecture: "h17c"))
        let tokenizer = try #require(ref.asset("tokenizer"))
        #expect(tokenizer.repo == "mistralai/Mistral-7B-Instruct-v0.3")
        #expect(tokenizer.revision == "c170c708c41dac9275d15a8fff4eca08d52bab71")
        #expect(tokenizer.subtree == "")
        #expect(Set(tokenizer.files.map(\.path)) == ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"])
        let main = try #require(ref.asset("main"))
        #expect(!main.files.contains { $0.path.hasPrefix("tokenizer/") }, "the bundle ships no tokenizer/")
        #expect(ref.assets.first?.name == "main")
    }

    @Test func unknownIDIsReported() {
        #expect {
            _ = try Catalog.builtin.entry("nope")
        } throws: { error in
            if case AetherError.unknownModel(let id) = error { return id == "nope" }
            return false
        }
    }

    static func variant(_ subtree: String, archs: [String]? = nil) -> CatalogVariant {
        CatalogVariant(driver: .decoderLM, assets: ["main": CatalogAsset(subtree: subtree, files: [])], architectures: archs)
    }

    static let entry = CatalogEntry(
        id: "toy", task: "text-generation", repo: "owner/toy", revision: String(repeating: "a", count: 40),
        license: "mit",
        variants: [
            "macos": variant("macos"),
            "ios": variant("ios"),
            "ios-h18p": variant("ios-h18p", archs: ["h18p"]),
            // Keyed by an architecture but restricted to another: never chosen.
            "ios-h16g": variant("ios-h16g", archs: ["h99"]),
        ])

    @Test(arguments: [
        (Device.Platform.iOS, "h18p", "ios-h18p"),
        (.iOS, "h17p", "ios"),
        (.iOS, "h16g", "ios"),
        (.macOS, "h17c", "macos"),
    ])
    func resolvesVariantForFakedDevice(_ platform: Device.Platform, _ arch: String, _ expected: String) throws {
        let ref = try ModelRef.resolve(Self.entry, for: Device(platform: platform, architecture: arch, physicalMemory: 8 << 30))
        #expect(ref.variant == expected)
        #expect(ref.asset("main")?.subtree == expected)
        #expect(ref.repo == "owner/toy" && ref.catalogID == "toy")
    }

    @Test func perArchOnlyVariantsAreUnsupportedElsewhere() throws {
        // nemotron and gemma-4 ship iOS only as h18p AOT: another iPhone gets unsupportedDevice before download.
        for id in ["nemotron-3-nano-4b", "gemma-4-e2b"] {
            #expect(try ModelRef.resolve(id, for: Device(platform: .iOS, architecture: "h18p")).variant == "ios-h18p")
            #expect {
                _ = try ModelRef.resolve(id, for: Device(platform: .iOS, architecture: "h17p"))
            } throws: { error in
                if case AetherError.unsupportedDevice(let cid, "ios", "h17p") = error { return cid == id }
                return false
            }
        }
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

    @Test func gemma4ResolvesWithItsTables() throws {
        let ref = try ModelRef.resolve("gemma-4-e2b", for: Device(platform: .macOS, architecture: "h17c"))
        let tables = try #require(ref.asset("tables"))
        #expect(tables.subtree == "ios-frontend/gemma4_qat_gather_raw")
        #expect(Set(tables.files.map(\.path)) == ["embed_per_layer.i8", "embed_per_layer.scale.f32"])
        #expect(tables.repo == ref.repo && tables.revision == ref.revision)
    }
}
