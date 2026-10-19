import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import CryptoKit
import Foundation
import Synchronization

enum KreaCommands {
    static func run(_ arguments: [String]) async throws -> Bool {
        guard let command = arguments.first, ["render-krea", "parity-krea"].contains(command) else { return false }
        if command == "parity-krea" { try await parity(arguments); return true }
        guard (7...8).contains(arguments.count), let width = Int(arguments[4]),
              let height = Int(arguments[5]), let seed = UInt64(arguments[6]) else {
            throw AetherError.invalidRequest("use render-krea BUNDLE OUTPUT.png PROMPT WIDTH HEIGHT SEED [SHAPE.json]")
        }
        let configuration = try KreaConfiguration(width: width, height: height, seed: seed)
        var assets = try KreaAssets.open(at: URL(fileURLWithPath: arguments[1]))
        if arguments.count == 8 { assets = try assets.registeringSpecialization(at: URL(fileURLWithPath: arguments[7])) }
        let tokenizer = try await KreaTokenizer.open(assets: assets)
        let prompt = try tokenizer.prepare(arguments[3])
        let session = KreaGenerationSession(assets: assets)
        do {
            let image = try await session.generate(configuration: configuration, prompt: prompt) { event in
                if case .step(let index, let total, _) = event { log("step \(index + 1)/\(total)") }
            }
            await session.unload()
            try image.savePNG(to: URL(fileURLWithPath: arguments[2]))
            try AetherProbe.printJSON(["output": arguments[2], "pixelsSHA256": floatHash(image.pixelsCHW.values)])
        } catch {
            await session.unload()
            throw error
        }
        return true
    }

    private static func parity(_ arguments: [String]) async throws {
        guard arguments.count == 7, ["provided", "encoded"].contains(arguments[6]) else {
            throw AetherError.invalidRequest("use parity-krea BUNDLE LAB FIXTURES SHAPE.json OUTPUT_DIRECTORY provided|encoded")
        }
        let out = URL(fileURLWithPath: arguments[5], isDirectory: true)
        guard !FileManager.default.fileExists(atPath: out.path) else {
            throw AetherError.invalidRequest("parity output directory already exists; choose a new directory")
        }
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let encoded = arguments[6] == "encoded"
        var report = ParityReport(mode: arguments[6])
        let started = ContinuousClock.now
        var session: KreaGenerationSession?
        var journal: ParityJournal?
        func writeReport() throws {
            report.seconds = AetherProbe.seconds(started.duration(to: .now))
            if let journal { report.events = journal.events }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: out.appendingPathComponent("report.json"), options: .atomic)
        }
        try writeReport()
        do {
            let fixtures = URL(fileURLWithPath: arguments[3], isDirectory: true)
            let lines = try String(contentsOf: fixtures.appendingPathComponent("cases.jsonl"), encoding: .utf8)
            let cases = try lines.split(separator: "\n").map { try JSONDecoder().decode(ParityCase.self, from: Data($0.utf8)) }
            guard let fixture = cases.first(where: { $0.id == "tower-768x1344" }), fixture.width == 768,
                  fixture.height == 1344, fixture.steps == 8, fixture.seed == 71344, fixture.textShape == [20, 30720] else {
                throw AetherError.invalidRequest("missing or incompatible tower fixture")
            }
            let configuration = try KreaConfiguration(width: fixture.width, height: fixture.height, steps: fixture.steps, seed: fixture.seed)
            let reference = URL(fileURLWithPath: arguments[2], isDirectory: true).appendingPathComponent(encoded
                ? "records/round4/fixed-encoded-tower/record.json" : "records/round4/fixed-paired-tower/record.json")
            let expected = try ReferenceHashes.load(reference, encoded: encoded)
            report.reference = reference.path
            report.referenceSHA256 = byteHash(try Data(contentsOf: reference))
            let recorder = try ParityJournal(expected: expected)
            journal = recorder
            let noise = try FloatTensor(shape: [configuration.imageTokenCount, 64], values: floats(fixtureFile(fixture.files.noise, root: fixtures)))
            let text = try FloatTensor(shape: fixture.textShape, values: floats(fixtureFile(fixture.files.text, root: fixtures)))
            let ids = try integers(fixtureFile(fixture.files.ids, root: fixtures))
            let masks = try integers(fixtureFile(fixture.files.mask, root: fixtures))
            for (name, relative) in ["noise": fixture.files.noise, "text": fixture.files.text, "ids": fixture.files.ids, "mask": fixture.files.mask] {
                report.fixtureSHA256[name] = byteHash(try Data(contentsOf: fixtureFile(relative, root: fixtures)))
            }
            guard masks.count == ids.count, masks.allSatisfy({ $0 == 1 }) else {
                throw AetherError.invalidRequest("fixture mask must contain one per token")
            }
            var assets = try KreaAssets.open(at: URL(fileURLWithPath: arguments[1]))
            report.bundleFingerprint = assets.fingerprint
            log("Verifying all model streams; this can read tens of GB.")
            try assets.verify()
            report.modelStreamsVerified = true
            log("Verifying source-linked shape specialization.")
            assets = try assets.registeringSpecialization(at: URL(fileURLWithPath: arguments[4]))
            report.registrationSHA256 = byteHash(try Data(contentsOf: URL(fileURLWithPath: arguments[4])))
            report.shapeRegistrationVerified = true
            let tokenizer = try await KreaTokenizer.open(assets: assets)
            let prompt = try tokenizer.prepare(fixture.prompt)
            guard prompt.tokenIDs == ids else { throw AetherError.invalidRequest("prompt tokens differ from fixture") }
            report.tokenIDsMatch = true
            report.stage = "generation"
            try writeReport()
            let pipeline = KreaGenerationSession(assets: assets)
            session = pipeline
            let progress: @Sendable (KreaGenerationEvent) throws -> Void = { event in
                try recorder.accept(event)
                log("gate event: \(recorder.events.last?.event ?? "unknown")")
            }
            let image: GeneratedImage
            if encoded {
                image = try await pipeline.generate(configuration: configuration, prompt: prompt, noise: noise, progress: progress)
            } else {
                image = try await pipeline.generate(configuration: configuration, conditioning: KreaConditioning(provided: text), noise: noise, progress: progress)
            }
            await pipeline.unload()
            try recorder.finish(image)
            try image.savePNG(to: out.appendingPathComponent("image.png"))
            report.passed = true
            report.stage = "complete"
            try writeReport()
            try AetherProbe.printJSON(report)
        } catch {
            if let session { await session.unload() }
            report.passed = false
            report.error = String(describing: error)
            try writeReport()
            throw error
        }
    }
}

private struct ParityCase: Decodable {
    let id: String
    let prompt: String
    let width: Int
    let height: Int
    let steps: Int
    let seed: UInt64
    let textShape: [Int]
    let files: Files
    struct Files: Decodable { let ids: String; let mask: String; let noise: String; let text: String }
}

struct ReferenceHashes {
    let text: String?
    let steps: [String]
    let pixels: String

    static func load(_ url: URL, encoded: Bool) throws -> Self {
        struct Record: Decodable {
            let runs: [Encoded]?
            let pairs: [Pair]?
            struct Encoded: Decodable { let encodedTextSHA256: String; let stepLatentsSHA256: [String]; let pixelsSHA256: String }
            struct Pair: Decodable { let stepLatentsSHA256: [[String]]; let pixelsSHA256: [String] }
        }
        let record = try JSONDecoder().decode(Record.self, from: Data(contentsOf: url))
        if encoded, let row = record.runs?.first { return Self(text: row.encodedTextSHA256, steps: row.stepLatentsSHA256, pixels: row.pixelsSHA256) }
        if !encoded, let row = record.pairs?.first, row.stepLatentsSHA256.count == 2, row.pixelsSHA256.count == 2 {
            return Self(text: nil, steps: row.stepLatentsSHA256[1], pixels: row.pixelsSHA256[1])
        }
        throw AetherError.invalidRequest("reference is missing the selected baseline")
    }
}

struct ParityEvent: Codable, Sendable {
    let event: String
    var step: Int? = nil
    var sha256: String? = nil
    var passed: Bool? = nil
    var selectionIdentity: String? = nil
}

private struct ParityReport: Codable {
    var passed = false
    let mode: String
    var stage = "preparation"
    var seconds: Double = 0
    var reference: String? = nil
    var referenceSHA256: String? = nil
    var registrationSHA256: String? = nil
    var fixtureSHA256: [String: String] = [:]
    var bundleFingerprint: String? = nil
    var modelStreamsVerified = false
    var shapeRegistrationVerified = false
    var tokenIDsMatch = false
    var events: [ParityEvent] = []
    var error: String? = nil
}

final class ParityJournal: Sendable {
    private struct State: Sendable {
        var events: [ParityEvent] = []
        var conditioning = false
        var denoisers = false
        var nextStep = 0
        var decoded = false
    }
    private let state = Mutex(State())
    private let expected: ReferenceHashes

    init(expected: ReferenceHashes) throws {
        guard expected.steps.count == 8 else { throw AetherError.invalidRequest("reference must have eight step hashes") }
        self.expected = expected
    }

    var events: [ParityEvent] { state.withLock { $0.events } }

    func accept(_ event: KreaGenerationEvent) throws {
        try state.withLock { state in
            switch event {
            case .conditioningReady(let text):
                guard !state.conditioning, !state.denoisers, text.rows == 20 else { throw AetherError.invalidTensor("unexpected conditioning event") }
                let hash = floatHash(text.tensor.values.map(Float.init))
                let passed = expected.text == nil || hash == expected.text
                state.events.append(ParityEvent(event: "conditioning", sha256: hash, passed: passed))
                guard passed else { throw AetherError.invalidTensor("encoded text hash mismatch") }
                state.conditioning = true
            case .denoisersReady(let selection):
                guard state.conditioning, !state.denoisers, selection.specialized,
                      selection.imageTokens == 4032, selection.textRows == 20 else {
                    throw AetherError.invalidModel("parity requires the tower specialization")
                }
                state.events.append(ParityEvent(event: "denoisers", passed: true, selectionIdentity: selection.identity))
                state.denoisers = true
            case .step(let index, let total, let latents):
                guard state.denoisers, !state.decoded, index == state.nextStep, index < 8, total == 8, latents.shape == [4032, 64] else {
                    throw AetherError.invalidTensor("unexpected parity step or latent shape")
                }
                let hash = floatHash(latents.values)
                let passed = hash == expected.steps[index]
                state.events.append(ParityEvent(event: "step", step: index, sha256: hash, passed: passed))
                guard passed else { throw AetherError.invalidTensor("latent hash mismatch at step \(index)") }
                state.nextStep += 1
            case .decoded:
                guard state.nextStep == 8, !state.decoded else { throw AetherError.invalidTensor("unexpected decode event") }
                state.events.append(ParityEvent(event: "decoded"))
                state.decoded = true
            }
        }
    }

    func finish(_ image: GeneratedImage) throws {
        try state.withLock { state in
            guard state.decoded, state.nextStep == 8, image.pixelsCHW.shape == [3, 1344, 768] else {
                throw AetherError.invalidTensor("parity ended before complete decode")
            }
            let hash = floatHash(image.pixelsCHW.values)
            let passed = hash == expected.pixels
            state.events.append(ParityEvent(event: "pixels", sha256: hash, passed: passed))
            guard passed else { throw AetherError.invalidTensor("decoded pixel hash mismatch") }
        }
    }
}

// Apple silicon is little-endian; these are the fixture's float32 byte hashes.
func floatHash(_ values: [Float]) -> String {
    values.withUnsafeBytes { byteHash(Data($0)) }
}

private func byteHash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

private func integers(_ url: URL) throws -> [Int32] {
    let data = try Data(contentsOf: url)
    guard data.count.isMultiple(of: 4) else { throw AetherError.invalidRequest("partial 32-bit fixture") }
    return data.withUnsafeBytes { bytes in
        stride(from: 0, to: data.count, by: 4).map { Int32(bitPattern: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian) }
    }
}

private func floats(_ url: URL) throws -> [Float] { try integers(url).map { Float(bitPattern: UInt32(bitPattern: $0)) } }

private func fixtureFile(_ relative: String, root: URL) throws -> URL {
    let canonicalRoot = root.standardizedFileURL.resolvingSymlinksInPath()
    let url = root.appendingPathComponent(relative).standardizedFileURL.resolvingSymlinksInPath()
    guard !relative.hasPrefix("/"), url.pathComponents.starts(with: canonicalRoot.pathComponents),
          url.pathComponents.count > canonicalRoot.pathComponents.count else {
        throw AetherError.invalidRequest("fixture file escapes its root")
    }
    return url
}

private func log(_ message: String) { FileHandle.standardError.write(Data((message + "\n").utf8)) }
