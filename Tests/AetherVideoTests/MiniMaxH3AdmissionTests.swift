import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing
@testable import AetherVideo

/// Measured admission per profile (MANIFEST.md §5.10, `h3.envelopes`): the export's profiles with their recipes,
/// memory class and shapes; an unmeasured key refused before anything loads; an envelope only for the export and chip
/// it was measured on; keyframed clips under their own keys; the physical-memory minimum; the gate's measuring policy.
@Suite("MiniMax H3 measured admission")
struct MiniMaxH3AdmissionTests {
    static let gib: Int64 = 1 << 30

    static func json() throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: Data(contentsOf: MiniMaxH3Tests.manifestURL)) as! [String: Any]
    }

    static func decode(_ json: [String: Any]) throws -> Manifest {
        try Manifest.decode(try JSONSerialization.data(withJSONObject: json))
    }

    static func envelope(
        _ key: String, recipes: [String] = ["turbo8"], architecture: String = "h17c", export: String? = nil,
        peak: Int64 = 10 << 30
    ) throws -> [String: Any] {
        let phase: [String: Any] = ["peakBytes": peak, "heldBytes": peak / 2]
        return ["key": key, "recipes": recipes, "architecture": architecture,
                "export": try export ?? MiniMaxH3Tests.manifest().measurementDigest, "marginBytes": 256 << 20,
                "perPrompt": phase, "perStep": phase, "perImage": phase, "renders": 1, "record": "test.json"]
    }

    /// The fixture with `envelopes` in `h3.envelopes`.
    static func manifest(envelopes: [[String: Any]]) throws -> Manifest {
        var json = try Self.json()
        var block = json["h3"] as! [String: Any]
        block["envelopes"] = envelopes
        json["h3"] = block
        return try decode(json)
    }

    final class Loads: Sendable {
        let count = Mutex(0)
    }

    static func pipeline(
        _ manifest: Manifest, device: Device = Device(platform: .macOS, architecture: "h17c", physicalMemory: 64 << 30),
        policy: VideoAdmissionPolicy = .measuredOnly, ledger: FootprintLedger = FootprintLedger(), loads: Loads = Loads()
    ) throws -> MiniMaxH3Pipeline {
        try MiniMaxH3Pipeline(
            manifest: manifest, device: device, bundle: nil,
            options: .init(gpu: .unadmitted, ledger: ledger, probe: GPUMemoryProbe { 0 }), admissionPolicy: policy,
            tokenize: { $0.unicodeScalars.map { Int32($0.value) } },
            load: { role throws(AetherError) in
                loads.count.withLock { $0 += 1 }
                throw .manifestInvalid(field: role, reason: "no graphs in this test")
            })
    }

    /// The error a generation ends with (none expected to succeed: the test has no graphs).
    static func failure(_ pipeline: MiniMaxH3Pipeline, _ request: VideoRequest) async -> AetherError? {
        do {
            _ = try await pipeline.generate(request)
            return nil
        } catch let error as AetherError {
            return error
        } catch {
            return nil
        }
    }

    static func isUnmeasured(_ error: AetherError?) -> Bool {
        if case .unsupportedOperation(let why)? = error { return why.contains("no measured memory envelope") }
        return false
    }

    static func reachedLoading(_ error: AetherError?) -> Bool {
        if case .manifestInvalid(_, let why)? = error { return why == "no graphs in this test" }
        return false
    }

    @Test func everyProfileCarriesItsRecipesMemoryClassAndShapes() throws {
        let h3 = try #require(try MiniMaxH3Tests.manifest().h3)
        let profiles = Dictionary(uniqueKeysWithValues: h3.profiles.map { ($0.key, $0) })
        #expect(profiles["960x544x124"]?.memory == .init(barBytes: 18 * Self.gib, minimumPhysicalMemoryBytes: 0))
        #expect(profiles["704x704x124"]?.memory == .init(barBytes: 18 * Self.gib, minimumPhysicalMemoryBytes: 0))
        for key in ["960x544x243", "544x960x243", "1280x704x124", "704x1280x124"] {
            #expect(profiles[key]?.memory == .init(barBytes: 24 * Self.gib, minimumPhysicalMemoryBytes: 64 * Self.gib), "\(key)")
        }
        #expect(profiles["960x544x243"]?.recipes == ["turbo8", "turbo4"] && profiles["1280x704x124"]?.recipes == ["turbo8"])
        #expect(profiles["960x544x243"]?.audioDecoder == "audioDecoder405" && profiles["960x544x243"]?.videoRows == 36720)
        #expect(profiles["1280x704x124"]?.keyframeEncoder == "videoEncoderTail1280x704")
        #expect(profiles["704x704x124"]?.shapes?["body"]?["table"] == [[6, 3, 5376]])
        #expect(h3.envelopes == nil)
        let shapes = MiniMaxH3Pipeline.bodyShapes(profiles["704x704x124"]!, rows: 19000, cardinality: 1,
                                                  inputs: ["hidden": "hidden", "table": "table"])
        #expect(shapes.allows("body", input: "table", shape: [6, 3, 5376]) && !shapes.allows("body", input: "table", shape: [6, 6, 5376]))
        #expect(shapes.allows("body", input: "hidden", shape: [1, 19000, 5376]) && !shapes.allows("body", input: "hidden", shape: [1, 19001, 5376]))
    }

    @Test func anUnmeasuredProfileIsRefusedBeforeAnythingLoads() async throws {
        let loads = Loads()
        let pipeline = try Self.pipeline(try MiniMaxH3Tests.manifest(), loads: loads)
        #expect(pipeline.envelopes.isEmpty && pipeline.envelope(for: VideoRequest(prompt: "a fox")) == nil)
        for request in [VideoRequest(prompt: "a fox in snow"), VideoRequest(prompt: "a fox in snow", profile: "704x704x124")] {
            #expect(Self.isUnmeasured(await Self.failure(pipeline, request)))
        }
        let resident = await pipeline.residentRoles
        #expect(loads.count.withLock { $0 } == 0 && resident.isEmpty)
    }

    @Test func aMeasuredKeyIsAdmittedAndSeedsTheLedger() async throws {
        let manifest = try Self.manifest(envelopes: [try Self.envelope("960x544x124", recipes: ["turbo8", "turbo4"])])
        let ledger = FootprintLedger(), loads = Loads()
        let device = Device(platform: .macOS, architecture: "h17c", physicalMemory: 64 << 30)
        let pipeline = try Self.pipeline(manifest, device: device, ledger: ledger, loads: loads)
        let envelope = try #require(pipeline.envelope(for: VideoRequest(prompt: "a fox")))
        #expect(envelope.key == "960x544x124" && envelope.barBytes == 18 * Self.gib && envelope.stagePeakBytes == 10 * Self.gib)
        #expect(envelope.minimumPhysicalMemoryBytes == 0 && envelope.marginBytes == 256 << 20)
        #expect(Self.reachedLoading(await Self.failure(pipeline, VideoRequest(prompt: "a fox in snow"))))
        #expect(Self.reachedLoading(await Self.failure(pipeline, VideoRequest(prompt: "a fox in snow", recipe: "turbo4"))))
        #expect(loads.count.withLock { $0 } > 0)
        let model = ResidentAdmission.modelKey(manifest: manifest, device: device)
        let seeded = ledger.snapshot.keys.filter { $0.model == model && $0.profile == "960x544x124" }
        #expect(seeded.count == 3)
        #expect(ledger.footprint(.init(model: model, profile: "960x544x124", phase: .perStep))?.peakBytes == 10 * Self.gib)
        // The other orientation and the keyframed keys aren't measured.
        #expect(Self.isUnmeasured(await Self.failure(pipeline, VideoRequest(prompt: "a fox in snow", profile: "544x960x124"))))
    }

    @Test func anEnvelopeHoldsOnlyForItsRecipeExportAndChip() async throws {
        let manifest = try Self.manifest(envelopes: [
            try Self.envelope("960x544x124"),
            try Self.envelope("544x960x124", architecture: "h16s"),
            try Self.envelope("704x704x124", export: String(repeating: "a", count: 64)),
        ])
        let pipeline = try Self.pipeline(manifest)
        #expect(pipeline.envelopes.keys.sorted() == ["960x544x124"])
        #expect(Self.isUnmeasured(await Self.failure(pipeline, VideoRequest(prompt: "a fox in snow", recipe: "turbo4"))))
        #expect(Self.isUnmeasured(await Self.failure(pipeline, VideoRequest(prompt: "a fox in snow", profile: "544x960x124"))))
        #expect(Self.isUnmeasured(await Self.failure(pipeline, VideoRequest(prompt: "a fox in snow", profile: "704x704x124"))))
        // On another chip nothing holds.
        let other = try Self.pipeline(manifest, device: Device(platform: .macOS, architecture: "h16s", physicalMemory: 64 << 30))
        #expect(other.envelopes.keys.sorted() == ["544x960x124"])
    }

    @Test func keyframedClipsAreAdmittedUnderTheirOwnKeys() async throws {
        let manifest = try Self.manifest(envelopes: [try Self.envelope("960x544x124/fl2va1")])
        let pipeline = try Self.pipeline(manifest)
        let frame = try VideoKeyframe(rgb: [UInt8](repeating: 128, count: 960 * 544 * 3), width: 960, height: 544)
        #expect(pipeline.envelope(for: VideoRequest(prompt: "a ball", keyframes: .first(frame)))?.key == "960x544x124/fl2va1")
        #expect(pipeline.envelope(for: VideoRequest(prompt: "a ball", keyframes: .last(frame)))?.key == "960x544x124/fl2va1")
        #expect(pipeline.envelope(for: VideoRequest(prompt: "a ball", keyframes: .firstAndLast(frame, frame))) == nil)
        #expect(pipeline.envelope(for: VideoRequest(prompt: "a ball")) == nil)
        #expect(Self.isUnmeasured(await Self.failure(pipeline, VideoRequest(prompt: "a ball"))))
    }

    @Test func anExtendedProfileNeedsItsPhysicalMemory() async throws {
        let manifest = try Self.manifest(envelopes: [try Self.envelope("960x544x243", peak: 20 << 30)])
        let small = try Self.pipeline(manifest, device: Device(platform: .macOS, architecture: "h17c", physicalMemory: 48 << 30))
        let error = await Self.failure(small, VideoRequest(prompt: "a fox in snow", profile: "960x544x243"))
        guard case .capacity(let what, let needed, _)? = error else { Issue.record("\(String(describing: error))"); return }
        #expect(what.contains("physical memory") && needed == 64 * Self.gib)
        let large = try Self.pipeline(manifest, device: Device(platform: .macOS, architecture: "h17c", physicalMemory: 64 << 30))
        #expect(Self.reachedLoading(await Self.failure(large, VideoRequest(prompt: "a fox in snow", profile: "960x544x243"))))
    }

    @Test func aRecipeTheProfileDoesNotOfferIsRefusedEvenWhenMeasuring() async throws {
        let pipeline = try Self.pipeline(try MiniMaxH3Tests.manifest(), policy: .measuring)
        let error = await Self.failure(pipeline, VideoRequest(prompt: "a fox in snow", profile: "1280x704x124", recipe: "turbo4"))
        guard case .unsupportedOperation(let why)? = error else { Issue.record("\(String(describing: error))"); return }
        #expect(why.contains("offers turbo8"))
        #expect(Self.reachedLoading(await Self.failure(pipeline, VideoRequest(prompt: "a fox in snow", profile: "1280x704x124"))))
    }

    @Test func theMeasuringPolicyAdmitsEverythingAndSeedsNothing() async throws {
        let manifest = try Self.manifest(envelopes: [try Self.envelope("960x544x124")])
        let ledger = FootprintLedger()
        let pipeline = try Self.pipeline(manifest, policy: .measuring, ledger: ledger)
        #expect(ledger.snapshot.isEmpty)
        for profile in ["960x544x124", "544x960x243", "704x704x124"] {
            #expect(Self.reachedLoading(await Self.failure(pipeline, VideoRequest(prompt: "a fox in snow", profile: profile))))
        }
    }

    @Test func recordingAnEnvelopeKeepsTheMeasurementDigest() throws {
        let plain = try MiniMaxH3Tests.manifest()
        let measured = try Self.manifest(envelopes: [try Self.envelope("960x544x124")])
        #expect(plain.measurementDigest == plain.contentDigest)
        #expect(measured.measurementDigest == plain.measurementDigest && measured.contentDigest != plain.contentDigest)
    }

    @Test func malformedEnvelopesAndProfilesAreRefusedNamingTheField() throws {
        func refused(_ edit: (inout [String: Any]) throws -> Void) throws -> String? {
            var json = try Self.json()
            var block = json["h3"] as! [String: Any]
            try edit(&block)
            json["h3"] = block
            do {
                _ = try Self.decode(json)
                return nil
            } catch let error as AetherError {
                if case .manifestInvalid(let field, _) = error { return field }
                return "\(error)"
            }
        }
        #expect(try refused { $0["envelopes"] = [try Self.envelope("960x544x124", peak: 18 << 30)] } == "h3.envelopes.0")
        #expect(try refused { $0["envelopes"] = [try Self.envelope("960x544x124/fl2va3")] } == "h3.envelopes.0.key")
        #expect(try refused { $0["envelopes"] = [try Self.envelope("640x480x124")] } == "h3.envelopes.0.key")
        #expect(try refused { $0["envelopes"] = [try Self.envelope("1280x704x124", recipes: ["turbo4"])] } == "h3.envelopes.0.recipes.0")
        #expect(try refused { $0["envelopes"] = [try Self.envelope("960x544x124", export: "xyz")] } == "h3.envelopes.0.export")
        #expect(try refused { $0["envelopes"] = [try Self.envelope("960x544x124"), try Self.envelope("960x544x124")] } == "h3.envelopes.1")
        #expect(try refused { block in
            var profiles = block["profiles"] as! [[String: Any]]
            var shapes = profiles[0]["shapes"] as! [String: Any]
            shapes["videoEncoderTail640x480"] = [String: Any]()
            profiles[0]["shapes"] = shapes
            block["profiles"] = profiles
        } == "h3.profiles.0.shapes.videoEncoderTail640x480")
        #expect(try refused { block in
            var profiles = block["profiles"] as! [[String: Any]]
            profiles[0]["recipes"] = ["turbo3"]
            block["profiles"] = profiles
        } == "h3.profiles.0.recipes.0")
        #expect(try refused { block in
            var profiles = block["profiles"] as! [[String: Any]]
            profiles[0]["memory"] = ["barBytes": 0, "minimumPhysicalMemoryBytes": 0]
            block["profiles"] = profiles
        } == "h3.profiles.0.memory")
    }
}
