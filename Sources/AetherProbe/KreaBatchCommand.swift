import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import CryptoKit
import Darwin
import Foundation

struct KreaBatchOptions: Sendable {
    static let usage = "use render-krea-batch BUNDLE NEW_OUTPUT_DIRECTORY PROMPT WIDTH HEIGHT SEED[,SEED...] [SHAPE.json] [--steps=N]"
    let bundle: URL
    let output: URL
    let prompt: String
    let width: Int
    let height: Int
    let seeds: [UInt64]
    let steps: Int
    let registration: URL?

    /// `--steps=N` (1...50, default 8) may appear once anywhere after the command name.
    init(_ arguments: [String]) throws {
        let stepFlags = arguments.dropFirst().filter { $0.hasPrefix("--steps=") }
        guard stepFlags.count <= 1 else { throw AetherError.invalidRequest(Self.usage) }
        var steps = 8
        if let flag = stepFlags.first {
            let value = flag.dropFirst("--steps=".count)
            guard !value.isEmpty, value.utf8.allSatisfy({ (48...57).contains($0) }), let parsed = Int(value) else {
                throw AetherError.invalidRequest("batch --steps must be a decimal integer")
            }
            steps = parsed
        }
        let arguments = arguments.filter { !$0.hasPrefix("--steps=") }
        guard (7...8).contains(arguments.count), arguments[0] == "render-krea-batch",
              let width = Int(arguments[4]), let height = Int(arguments[5]),
              !arguments[3].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AetherError.invalidRequest(Self.usage)
        }
        let parts = arguments[6].split(separator: ",", omittingEmptySubsequences: false)
        guard (1...64).contains(parts.count) else { throw AetherError.invalidRequest("batch requires 1...64 seeds") }
        let seeds = try parts.map { part -> UInt64 in
            guard !part.isEmpty, part.utf8.allSatisfy({ (48...57).contains($0) }), let seed = UInt64(part) else {
                throw AetherError.invalidRequest("batch seeds must be comma-separated UInt64 decimal integers")
            }
            return seed
        }
        _ = try KreaConfiguration(width: width, height: height, steps: steps)
        self.bundle = URL(fileURLWithPath: arguments[1], isDirectory: true)
        self.output = URL(fileURLWithPath: arguments[2], isDirectory: true).standardizedFileURL
        self.prompt = arguments[3]
        self.width = width
        self.height = height
        self.seeds = seeds
        self.steps = steps
        self.registration = arguments.count == 8 ? URL(fileURLWithPath: arguments[7]) : nil
    }
}

struct BatchPreparation: Codable, Sendable {
    let bundleFingerprint: String
    let conditioningSHA256: String
    let conditioningByteCount: Int
    let conditioningRows: Int
    let wasTruncated: Bool
}

protocol BatchRenderer: Sendable {
    func prepare() async throws -> BatchPreparation
    func generate(_ configuration: KreaConfiguration) async throws -> GeneratedImage
    func unload() async
}

private actor NativeBatchRenderer: BatchRenderer {
    let session: KreaGenerationSession
    let prompt: KreaPrompt
    var prepared: KreaPreparedPrompt?

    init(assets: KreaAssets, prompt: KreaPrompt) {
        session = KreaGenerationSession(assets: assets, compute: .preferGPU, residency: .balanced)
        self.prompt = prompt
    }
    func prepare() async throws -> BatchPreparation {
        let value = try await session.prepare(prompt: prompt)
        prepared = value
        return BatchPreparation(bundleFingerprint: value.bundleFingerprint,
            conditioningSHA256: floatHash(value.conditioning.tensor.values.map(Float.init)),
            conditioningByteCount: value.conditioningByteCount, conditioningRows: value.conditioning.rows,
            wasTruncated: prompt.wasTruncated)
    }
    func generate(_ configuration: KreaConfiguration) async throws -> GeneratedImage {
        guard let prepared else { throw AetherError.invalidRequest("batch prompt has not been prepared") }
        return try await session.generate(configuration: configuration, preparedPrompt: prepared) { event in
            if case .step(let index, let total, _) = event { log("step \(index + 1)/\(total)") }
        }
    }
    func unload() async { await session.unload(); prepared = nil }
}

struct BatchImageRecord: Codable, Sendable {
    let index: Int
    let seed: UInt64
    let png: String
    var status = "generating"
    var generationSeconds: Double?
    var pixelsSHA256: String?
    var pngSHA256: String?
}

struct KreaBatchReport: Encodable, Sendable {
    let schema = 1
    let startedAt = ISO8601DateFormatter().string(from: Date())
    let os = ProcessInfo.processInfo.operatingSystemVersionString
    let physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
    var completed = false
    var stage = "preparation"
    let bundlePath: String
    let prompt: String
    let width: Int
    let height: Int
    let steps: Int
    let seeds: [UInt64]
    let registrationPath: String?
    var registrationSHA256: String?
    let computePolicy = "preferGPU"
    let residency = "balanced"
    let modelStreamsVerified = false
    let timingScope = "Wall clock, not a benchmark. Setup includes assets/tokenizer/optional specialization, prompt encoding, hashing and encoder unload. Generation includes model acquisition and progress logging; excludes final hashes, PNG/report IO and unload. No warm-ups."
    var preparation: BatchPreparation?
    var setupSeconds: Double?
    var totalSeconds: Double = 0
    var images: [BatchImageRecord] = []
    var error: String?
}

enum KreaBatchCommand {
    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "render-krea-batch" else { return false }
        let options = try KreaBatchOptions(arguments)
        let report = try await execute(options: options) {
            var assets = try KreaAssets.open(at: options.bundle)
            if let registration = options.registration { assets = try assets.registeringSpecialization(at: registration) }
            let tokenizer = try await KreaTokenizer.open(assets: assets)
            return NativeBatchRenderer(assets: assets, prompt: try tokenizer.prepare(options.prompt))
        }
        try AetherProbe.printJSON(report)
        return true
    }

    static func execute(options: KreaBatchOptions,
                        makeRenderer: () async throws -> any BatchRenderer) async throws -> KreaBatchReport {
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: options.output.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(options.output.path, 0o755) == 0 else {
            throw AetherError.invalidRequest("batch output must be a new directory (mkdir errno \(errno))")
        }
        let started = ContinuousClock.now
        var report = KreaBatchReport(bundlePath: options.bundle.path, prompt: options.prompt,
            width: options.width, height: options.height, steps: options.steps, seeds: options.seeds,
            registrationPath: options.registration?.path)
        var renderer: (any BatchRenderer)?
        func write() throws {
            report.totalSeconds = AetherProbe.seconds(started.duration(to: .now))
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: options.output.appendingPathComponent("report.json"), options: .atomic)
        }
        do {
            try write()
            if let registration = options.registration { report.registrationSHA256 = try fileSHA256(registration) }
            let setupStarted = ContinuousClock.now
            let pipeline = try await makeRenderer()
            renderer = pipeline
            report.preparation = try await pipeline.prepare()
            try Task.checkCancellation()
            report.setupSeconds = AetherProbe.seconds(setupStarted.duration(to: .now))
            for (index, seed) in options.seeds.enumerated() {
                try Task.checkCancellation()
                let name = String(format: "%03d", index + 1) + "-seed-\(seed).png"
                report.stage = "generation"
                report.images.append(BatchImageRecord(index: index + 1, seed: seed, png: name))
                try write()
                log("image \(index + 1)/\(options.seeds.count), seed \(seed)")
                let configuration = try KreaConfiguration(width: options.width, height: options.height, steps: options.steps, seed: seed)
                let generationStarted = ContinuousClock.now
                let image = try await pipeline.generate(configuration)
                report.images[index].generationSeconds = AetherProbe.seconds(generationStarted.duration(to: .now))
                try Task.checkCancellation()
                guard image.width == options.width, image.height == options.height else {
                    throw AetherError.invalidTensor("batch image dimensions differ from request")
                }
                report.images[index].pixelsSHA256 = floatHash(image.pixelsCHW.values)
                report.stage = "saving"
                try write()
                let temporary = options.output.appendingPathComponent(".\(name).tmp.png")
                do {
                    defer { try? FileManager.default.removeItem(at: temporary) }
                    try image.savePNG(to: temporary)
                    report.images[index].pngSHA256 = try fileSHA256(temporary)
                    try FileManager.default.moveItem(at: temporary, to: options.output.appendingPathComponent(name))
                }
                report.images[index].status = "complete"
                try write()
            }
            await pipeline.unload()
            renderer = nil
            report.stage = "complete"
            report.completed = true
            try write()
            return report
        } catch {
            if let renderer { await renderer.unload() }
            report.completed = false
            report.stage = "failed"
            report.error = String(describing: error)
            if let last = report.images.indices.last, report.images[last].status != "complete" {
                report.images[last].status = "failed"
            }
            // Preserve the original failure if the destination itself is unwritable.
            do { try write() } catch { log("could not persist batch failure report: \(error)") }
            throw error
        }
    }
}

private func fileSHA256(_ url: URL) throws -> String {
    SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
}

private func log(_ message: String) { FileHandle.standardError.write(Data((message + "\n").utf8)) }
