// The MLX spike's qualification run, against the same references and criteria as Kit's Core AI backend
// (Tests/AetherNativeTests/LanguageDecodeQualificationTests.swift): per teacher-forced row, cosine >= 0.995 against the
// float32 reference and argmax equal unless at a reference near-tie (top-1 minus top-2 probability < 0.1, native
// token in the reference's top 5); the greedy continuation identical, or first diverging at such a near-tie. Then
// time to first token at several prompt lengths, decode speed, and MLX's peak memory.
//
//   mlx-qwen3-check <hf snapshot dir> <reference dir> <report.json> [float16|bfloat16|float32]

import Foundation
import MLX
import MLXQwen3

struct Reference: Decodable {
    let promptTokens: [Int]
    let greedyTokens: [Int]
    let forcedTokens: [Int]
    let forcedLogits: Logits
    struct Logits: Decodable { let file: String; let rows: Int; let vocab: Int }
}

struct Row: Encodable {
    let cosine: Double, maxAbs: Double, argmaxEqual: Bool, referenceMargin: Double, nativeRankInReference: Int
    var nearTie: Bool { !argmaxEqual && referenceMargin < 0.1 && nativeRankInReference < 5 }
}

struct Report: Encodable {
    var model = "", dtype = "", device = "", loadSeconds = 0.0
    var forcedMinimumCosine = 0.0, forcedMaximumAbs = 0.0, forcedArgmaxMismatches = 0, forcedNearTies = 0
    var forcedPassed = false
    var greedyTokens: [Int] = [], greedyFirstDivergence: Int?, greedyAccepted = false
    var timeToFirstToken: [String: Double] = [:]
    var decodeTokensPerSecond = 0.0, decodeTokens = 0
    var peakMemoryMiB = 0, activeMemoryMiB = 0
}

func compare(_ native: [Float], _ reference: ArraySlice<Float>) -> Row {
    var dot = 0.0, nn = 0.0, rr = 0.0, maxAbs = 0.0
    for (n, r) in zip(native, reference) {
        let a = Double(n), b = Double(r)
        dot += a * b; nn += a * a; rr += b * b; maxAbs = max(maxAbs, abs(a - b))
    }
    let referenceBest = reference.indices.max { reference[$0] < reference[$1] }! - reference.startIndex
    let nativeBest = native.indices.max { native[$0] < native[$1] }!
    let sorted = reference.map(Double.init).sorted(by: >)
    let denominator = reference.reduce(0.0) { $0 + exp(Double($1) - sorted[0]) }
    let margin = (1 - exp(sorted[1] - sorted[0])) / denominator
    let nativeValue = Double(reference[reference.startIndex + nativeBest])
    return Row(cosine: dot / (nn.squareRoot() * rr.squareRoot()), maxAbs: maxAbs, argmaxEqual: nativeBest == referenceBest,
               referenceMargin: margin, nativeRankInReference: reference.filter { Double($0) > nativeValue }.count)
}

let arguments = CommandLine.arguments
guard arguments.count >= 4 else { print("usage: mlx-qwen3-check <snapshot> <reference dir> <report.json> [dtype]"); exit(2) }
let snapshot = URL(filePath: arguments[1]), referenceDirectory = URL(filePath: arguments[2]), output = URL(filePath: arguments[3])
let dtype: DType = switch arguments.count > 4 ? arguments[4] : "float16" {
case "bfloat16": .bfloat16
case "float32": .float32
default: .float16
}
var report = Report()
report.model = snapshot.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent
report.dtype = "\(dtype)"
report.device = GPU.deviceInfo().architecture
let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: referenceDirectory.appendingPathComponent("reference.json")))
let forced = try Data(contentsOf: referenceDirectory.appendingPathComponent(reference.forcedLogits.file))
    .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
let vocab = reference.forcedLogits.vocab

var start = ContinuousClock.now
let runner = try MLXQwen3Runner(directory: snapshot, dtype: dtype)
func seconds(_ from: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - from
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}
report.loadSeconds = seconds(start)

// Teacher-forced rows.
let rows = runner.teacherForcedLogits(prompt: reference.promptTokens, continuation: Array(reference.forcedTokens.dropLast()))
let comparisons = rows.enumerated().map { index, row in compare(Array(row.prefix(vocab)), forced[(index * vocab)..<((index + 1) * vocab)]) }
report.forcedMinimumCosine = comparisons.map(\.cosine).min() ?? 0
report.forcedMaximumAbs = comparisons.map(\.maxAbs).max() ?? 0
report.forcedArgmaxMismatches = comparisons.filter { !$0.argmaxEqual }.count
report.forcedNearTies = comparisons.filter(\.nearTie).count
report.forcedPassed = report.forcedMinimumCosine >= 0.995 && comparisons.allSatisfy { $0.argmaxEqual || $0.nearTie }

// Greedy continuation (no stop token, as the reference's).
let greedy = runner.greedy(prompt: reference.promptTokens, maximumTokens: reference.greedyTokens.count, stopTokenIDs: [])
report.greedyTokens = greedy.tokens
report.greedyFirstDivergence = zip(greedy.tokens, reference.greedyTokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
if let divergence = report.greedyFirstDivergence, divergence < comparisons.count,
   reference.forcedTokens.prefix(divergence) == reference.greedyTokens.prefix(divergence) {
    var chosen = [Float](repeating: 0, count: vocab)
    chosen[greedy.tokens[divergence]] = 1
    let row = compare(chosen, forced[(divergence * vocab)..<((divergence + 1) * vocab)])
    report.greedyAccepted = row.referenceMargin < 0.1 && row.nativeRankInReference < 5
} else {
    report.greedyAccepted = report.greedyFirstDivergence == nil
}

// Time to first token (warm: second of two requests) and decode speed.
for length in [64, 256, 1024, 2048] {
    let prompt = (0..<length).map { 1000 + ($0 * 37) % 5000 }
    _ = runner.greedy(prompt: prompt, maximumTokens: 1, stopTokenIDs: [])
    report.timeToFirstToken[String(length)] = runner.greedy(prompt: prompt, maximumTokens: 1, stopTokenIDs: []).timeToFirstTokenSeconds
}
let decode = runner.greedy(prompt: reference.promptTokens, maximumTokens: 256, stopTokenIDs: [])
report.decodeTokens = decode.tokens.count
report.decodeTokensPerSecond = Double(decode.tokens.count - 1) / (decode.totalSeconds - decode.timeToFirstTokenSeconds)
report.peakMemoryMiB = Memory.peakMemory >> 20
report.activeMemoryMiB = Memory.activeMemory >> 20

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
try encoder.encode(report).write(to: output)
print("forced: min cosine \(report.forcedMinimumCosine), argmax mismatches \(report.forcedArgmaxMismatches) (near-ties \(report.forcedNearTies)), passed \(report.forcedPassed)")
print("greedy: first divergence \(report.greedyFirstDivergence.map(String.init) ?? "none"), accepted \(report.greedyAccepted)")
print("ttft \(report.timeToFirstToken.sorted { Int($0.key)! < Int($1.key)! }), decode \(report.decodeTokensPerSecond) tok/s, peak \(report.peakMemoryMiB) MiB")
