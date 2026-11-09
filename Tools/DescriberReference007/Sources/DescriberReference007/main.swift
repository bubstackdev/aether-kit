// Records the 0.0.7 describer's greedy replies on the gate's synthetic cases (Forge/qwen3vl/synthetic_pictures.py):
//
//     AETHER_KIT_007=<0.0.7 checkout> swift run -c release --package-path Tools/DescriberReference007 \
//         DescriberReference007 --bundle <0.0.x language-model.json export> --pictures <pictures dir> --output <new dir>
//
// For each case it renders the turn as a 0.0.7 host does (`LanguageTokenizer.render` of an optional system message
// and `.user(question, images: n)`), runs `VisionLanguageSession.generate(renderedTokenIDs:images:maximumTokens:)`
// (greedy, the describer's path), then re-feeds the reply through the same decoder with teacher forcing for each
// step's top five tokens and their probabilities: the near-tie rule's evidence. It also writes each picture's
// preprocessed patch rows (fp16, as 0.0.7 feeds its tower), so the new driver's preprocessing can be compared.
// GPU work: run it under scripts/gpu-lease.
import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision
import Foundation

struct Cases: Decodable {
    struct Case: Decodable {
        let name: String
        let instructions: String
        let question: String
        let pictures: [String]
        let maximumTokens: Int
    }
    let files: [String: String]
    let cases: [Case]
}

struct Step: Encodable {
    let top: [Int]
    let probabilities: [Double]
}

struct Record: Encodable {
    let name: String
    let renderedTokenIDs: [Int]
    let tokens: [Int]
    let stopReason: String
    let text: String
    let timeToFirstTokenSeconds: Double
    let totalSeconds: Double
    var steps: [Step] = []
    var teacherArgmaxAgrees = true
}

struct Picture: Encodable {
    let file: String
    let grid: [Int]
    let pixelsFile: String
}

func arguments() -> [String: String] {
    var out: [String: String] = [:]
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let key = it.next() { if key.hasPrefix("--"), let value = it.next() { out[String(key.dropFirst(2))] = value } }
    return out
}

func top(_ row: [Float], _ k: Int) -> Step {
    let m = row.max() ?? 0
    var sum = 0.0
    for v in row { sum += exp(Double(v - m)) }
    let order = row.indices.sorted { row[$0] > row[$1] }.prefix(k)
    return Step(top: Array(order), probabilities: order.map { exp(Double(row[$0] - m)) / sum })
}

@main
struct Main {
    static func main() async throws {
        let args = arguments()
        guard let bundlePath = args["bundle"], let picturesPath = args["pictures"], let outputPath = args["output"] else {
            print("usage: DescriberReference007 --bundle <dir> --pictures <dir> --output <new dir>")
            exit(2)
        }
        let bundle = URL(fileURLWithPath: bundlePath), pictures = URL(fileURLWithPath: picturesPath)
        let output = URL(fileURLWithPath: outputPath)
        guard !FileManager.default.fileExists(atPath: output.path) else { print("output exists"); exit(2) }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let cases = try JSONDecoder().decode(Cases.self, from: Data(contentsOf: pictures.appending(path: "cases.json")))
        let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
        let manifest = try LanguageModelManifest.load(from: bundle)
        guard let registration = manifest.vision else { print("not a vision bundle"); exit(2) }

        // 1. The describer itself, every case on one opened session (as a host's several turns are).
        var records: [Record] = []
        let session = try await VisionLanguageSession.open(bundle: bundle, platform: .macos)
        for c in cases.cases {
            let messages: [LanguageChatMessage] = (c.instructions.isEmpty ? [] : [.system(c.instructions)])
                + [.user(c.question, images: c.pictures.count)]
            let prompt = try tokenizer.render(messages)
            let images = try c.pictures.map { try VisionRGBImage(contentsOf: pictures.appending(path: $0)) }
            let reply = try await session.generate(renderedTokenIDs: prompt.tokenIDs, images: images, maximumTokens: c.maximumTokens)
            let text = tokenizer.decode(reply.tokens)
            print("[007] \(c.name): \(reply.tokens.count) tokens, \(reply.stopReason), \(String(format: "%.2f", reply.totalSeconds)) s: \(text.prefix(120))")
            records.append(Record(name: c.name, renderedTokenIDs: prompt.tokenIDs, tokens: reply.tokens, stopReason: reply.stopReason,
                                  text: text, timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds, totalSeconds: reply.totalSeconds))
        }
        await session.unload()

        // 2. Teacher forcing on the same decoder and tower: each step's top five, and the preprocessed pixels.
        let configuration = VisionLanguageSession.preprocessing(for: registration)
        let encoder = try await LanguageVisionEncoder.open(
            at: bundle.appending(path: registration.modelDirectory),
            configuration: .init(mergeSize: registration.mergeSize, positionTableSide: registration.positionTableSide,
                                 imageGrids: (registration.imageGrids ?? []).map { Qwen3VLImagePreprocessor.Grid(height: $0.height, width: $0.width) }),
            expectFrequentReshapes: try registration.encoderExpectsFrequentReshapes())
        let decoder = try await LanguageGenerationSession.open(bundle: bundle, platform: .macos)
        let imageTokens = LanguageImageTokens(visionStart: registration.visionStartTokenID, imagePad: registration.imageTokenID,
                                              visionEnd: registration.visionEndTokenID)
        var pictureRecords: [String: Picture] = [:]
        for (index, c) in cases.cases.enumerated() {
            var features: [LanguageImageFeatures] = []
            for file in c.pictures {
                let image = try VisionRGBImage(contentsOf: pictures.appending(path: file))
                let processed = try Qwen3VLImagePreprocessor.process(rgb: image.rgb, width: image.width, height: image.height,
                                                                     configuration: configuration)
                let halves = processed.pixels.map { Float16($0) }
                if pictureRecords[file] == nil {
                    let name = file.replacingOccurrences(of: ".png", with: ".patches.f16")
                    try halves.withUnsafeBytes { try Data($0).write(to: output.appending(path: name)) }
                    pictureRecords[file] = Picture(file: file, grid: [processed.grid.t, processed.grid.h, processed.grid.w], pixelsFile: name)
                }
                features.append(try await encoder.encode(pixels: halves, grid: processed.grid))
            }
            var record = records[index]
            let continuation = Array(record.tokens.dropLast())
            let rows: [[Float]] = c.pictures.isEmpty
                ? try await decoder.teacherForcedLogits(prompt: record.renderedTokenIDs, continuation: continuation)
                : try await decoder.teacherForcedLogits(
                    prompt: LanguageVisionPrompt(renderedTokenIDs: record.renderedTokenIDs, images: features),
                    imageTokens: imageTokens, continuation: continuation)
            record.steps = rows.map { top($0, 5) }
            record.teacherArgmaxAgrees = zip(record.steps, record.tokens).allSatisfy { $0.top[0] == $1 }
            print("[007] \(c.name): teacher-forced \(rows.count) rows, argmax agrees: \(record.teacherArgmaxAgrees), "
                  + "smallest margin \(String(format: "%.4f", record.steps.map { $0.probabilities[0] - $0.probabilities[1] }.min() ?? 1))")
            records[index] = record
        }
        await decoder.unload()
        await encoder.unload()
        let encoderJSON = JSONEncoder()
        encoderJSON.outputFormatting = [.prettyPrinted, .sortedKeys]
        struct Output: Encodable {
            let schema = 1
            let bundle: String
            let modelSHA256: String
            let visionModelSHA256: String
            let sourceRepository: String?
            let sourceRevision: String?
            let pictures: [String: String]
            let preprocessed: [Picture]
            let cases: [Record]
        }
        let out = Output(bundle: bundle.lastPathComponent, modelSHA256: manifest.modelSHA256, visionModelSHA256: registration.modelSHA256,
                         sourceRepository: manifest.sourceRepository, sourceRevision: manifest.sourceRevision, pictures: cases.files,
                         preprocessed: pictureRecords.values.sorted { $0.file < $1.file }, cases: records)
        try encoderJSON.encode(out).write(to: output.appending(path: "reference.json"))
        print("[007] wrote \(output.appending(path: "reference.json").path)")
    }
}
