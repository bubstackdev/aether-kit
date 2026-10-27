import AetherCoreAI
import AetherKit
import AetherSpeech
import CryptoKit
import Foundation
import Synchronization
import Testing
@testable import AetherSpeechCoreAI

struct CPUFixture: Decodable {
    struct Tokenization: Decodable { let phonemes: String; let ids: [Int32] }
    struct Noise: Decodable { let seed: UInt64; let count: Int; let first: [Float]; let sha256: String }
    struct Durations: Decodable {
        struct Case: Decodable { let speed: Float; let frames: [Int] }
        let sums: [Float]
        let cases: [Case]
    }
    let vocabulary: [String: Int]
    let tokenization: [Tokenization]
    let noise: Noise
    let durations: Durations

    static func load() throws -> Self {
        let url = try #require(Bundle.module.url(forResource: "kokoro-cpu-fixture", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

func sha256Hex(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

/// Writes a weight-free bundle with the real kokoro.json layout. Model streams are placeholder bytes.
struct FakeBundle {
    let root: URL
    let voices: [String]

    static func make(voices: [String] = ["af_heart", "bm_george"], maximumFrames: Int = 4096, decoderPayload: String = "decoder",
                     decoderBuckets: [Int]? = nil,
                     mutate: (inout [String: Any]) -> Void = { _ in }) throws -> FakeBundle {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("kokoro-fake-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("voices"), withIntermediateDirectories: true)
        func model(_ name: String, payload: String) throws -> [String: Any] {
            let directory = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let stream = Data(payload.utf8)
            let declaration = Data(SHA256.hash(data: stream))
            let metadata = Data(#"{"assetVersion":"2.0"}"#.utf8)
            try stream.write(to: directory.appendingPathComponent("main.mlirb"))
            try declaration.write(to: directory.appendingPathComponent("main.hash"))
            try metadata.write(to: directory.appendingPathComponent("metadata.json"))
            return ["path": name, "function": name == "durations.aimodel" ? "durations" : "decode",
                    "sha256": ["main.mlirb": sha256Hex(stream), "main.hash": sha256Hex(declaration), "metadata.json": sha256Hex(metadata)]]
        }
        var voiceEntries: [[String: Any]] = []
        for (index, id) in voices.enumerated() {
            // Row r of voice i holds r + 1000 * i in every element; row selection is observable.
            var bytes = Data(capacity: 510 * 256 * 4)
            for row in 0..<510 {
                let value = Float(row + 1000 * index)
                for _ in 0..<256 { withUnsafeBytes(of: value.bitPattern.littleEndian) { bytes.append(contentsOf: $0) } }
            }
            try bytes.write(to: root.appendingPathComponent("voices/\(id).f32"))
            let voice = try KokoroVoice(id)
            voiceEntries.append(["id": id, "language": voice.language.bcp47, "gender": voice.gender == .female ? "female" : "male",
                                 "file": "voices/\(id).f32", "sha256": sha256Hex(bytes), "source_sha256": String(repeating: "0", count: 64)])
        }
        var contract: [String: Any] = [
            "schema": 1, "family": "kokoro-82m", "precision": "float32",
            "source": ["repository": "hexgrad/Kokoro-82M", "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
                       "checkpoint_sha256": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4", "license": "Apache-2.0"],
            "sampleRate": 24000, "samplesPerFrame": 600, "harmonics": 9, "maximumPhonemes": 510,
            "maximumFrames": maximumFrames, "styleDimension": 256, "boundaryToken": 0,
            "decoderBuckets": decoderBuckets as Any,
            "vocabulary": try CPUFixture.load().vocabulary,
            "models": ["durations": try model("durations.aimodel", payload: "durations"),
                       "decoder": try model("decoder.aimodel", payload: decoderPayload)],
            "voices": voiceEntries,
        ]
        mutate(&contract)
        try JSONSerialization.data(withJSONObject: contract, options: [.sortedKeys]).write(to: root.appendingPathComponent("kokoro.json"))
        return FakeBundle(root: root, voices: voices)
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}

final class Locked<Value: Sendable>: Sendable {
    private let value: Mutex<Value>
    init(_ value: Value) { self.value = Mutex(value) }
    func withLock<R>(_ body: (inout Value) throws -> R) rethrows -> R { try value.withLock { try body(&$0) } }
}

final class EventLog: Sendable {
    private let values = Mutex<[String]>([])
    func append(_ value: String) { values.withLock { $0.append(value) } }
    var snapshot: [String] { values.withLock { $0 } }
}

actor Gate {
    private var entered = false
    private var released = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var held: CheckedContinuation<Void, Never>?
    func enter() async {
        if released { return }
        await withCheckedContinuation { continuation in
            held = continuation
            entered = true
            waiters.forEach { $0.resume() }
            waiters.removeAll()
        }
    }
    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func release() { released = true; held?.resume(); held = nil }
}

private func tensor(_ type: String, _ shape: [Int]) -> [String: Any] { ["kind": "tensor", "scalarType": type, "shape": shape] }

func describe(_ name: String, inputs: [String: [String: Any]], outputs: [String: [String: Any]]) throws -> ModelFunctionDescription {
    try JSONDecoder().decode(ModelFunctionDescription.self, from: JSONSerialization.data(withJSONObject: [
        "name": name, "inputs": inputs, "outputs": outputs, "states": [String: Any](),
    ]))
}

func durationsSignature(textChannels: Int = 512) throws -> ModelFunctionDescription {
    try describe("durations", inputs: ["input_ids": tensor("int32", [1, -1]), "style": tensor("float32", [1, 256])],
                 outputs: ["duration": tensor("float32", [1, -1]), "prosody": tensor("float32", [1, -1, 640]),
                           "text": tensor("float32", [1, textChannels, -1])])
}

func decoderSignature(bucketed: Bool = false) throws -> ModelFunctionDescription {
    var inputs = ["prosody": tensor("float32", [1, 640, -1]), "text": tensor("float32", [1, 512, -1]),
                  "style": tensor("float32", [1, 256]), "noise": tensor("float32", [1, -1, 600, 9])]
    if bucketed { inputs["frames"] = tensor("int32", [1]) }
    return try describe("decode", inputs: inputs, outputs: ["audio": tensor("float32", [1, -1])])
}

struct DecoderCall: Sendable {
    let shape: Int
    let validFrames: Int?
    let noiseCount: Int
    let paddingIsZero: Bool
}

struct FakeModel: KokoroExecutable {
    let functions: [ModelFunctionDescription]
    let run: @Sendable ([String: TensorValue]) async throws -> [String: TensorValue]
    let close: @Sendable () -> Void
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] { try await run(inputs) }
    func unload() async { close() }
}

/// Durations: every token gets `framesPerToken` frames; prosody[t,c] = t, text[c,t] = -t, so the
/// decoder can check alignment. Decoder: audio sample k of frame f is f + noise[f,0,0] * 0 (shape only),
/// and records the first noise value and the style's first element.
final class FakeFactory: KokoroModelFactory, Sendable {
    let log: EventLog
    let durationsGate: Gate?
    let decoderGate: Gate?
    let framesPerToken: Float
    let badDurationsSignature: Bool
    let bucketedDecoder: Bool
    let observed = Locked<[(noise: Float, style: Float, prosody: [Float])]>([])
    let decoderCalls = Locked<[DecoderCall]>([])

    init(log: EventLog, durationsGate: Gate? = nil, decoderGate: Gate? = nil, framesPerToken: Float = 2,
         badDurationsSignature: Bool = false, bucketedDecoder: Bool = false) {
        self.bucketedDecoder = bucketedDecoder
        self.log = log
        self.durationsGate = durationsGate
        self.decoderGate = decoderGate
        self.framesPerToken = framesPerToken
        self.badDurationsSignature = badDurationsSignature
    }

    func open(_ asset: KokoroModelAsset) async throws -> any KokoroExecutable {
        let name = asset.function
        log.append("open:\(name)")
        let log = log
        if name == "durations" {
            let gate = durationsGate, perToken = framesPerToken
            return FakeModel(functions: [try durationsSignature(textChannels: badDurationsSignature ? 256 : 512)], run: { inputs in
                log.append("run:durations")
                if let gate { await gate.enter() }
                guard case .int32(let ids) = inputs["input_ids"], case .float32(let style) = inputs["style"], style.shape == [1, 256] else {
                    throw AetherError.invalidTensor("unexpected durations inputs")
                }
                let t = ids.shape[1]
                return [
                    "duration": .float32(try FloatTensor(shape: [1, t], values: Array(repeating: perToken, count: t))),
                    "prosody": .float32(try FloatTensor(shape: [1, t, 640], values: (0..<t).flatMap { Array(repeating: Float($0), count: 640) })),
                    "text": .float32(try FloatTensor(shape: [1, 512, t], values: (0..<512).flatMap { _ in (0..<t).map { -Float($0) } })),
                ]
            }, close: { log.append("close:durations") })
        }
        let gate = decoderGate, observed = observed, calls = decoderCalls
        return FakeModel(functions: [try decoderSignature(bucketed: bucketedDecoder)], run: { inputs in
            log.append("run:decode")
            if let gate { await gate.enter() }
            guard case .float32(let prosody) = inputs["prosody"], case .float32(let text) = inputs["text"],
                  case .float32(let noise) = inputs["noise"], case .float32(let style) = inputs["style"] else {
                throw AetherError.invalidTensor("unexpected decoder inputs")
            }
            let frames = prosody.shape[2]
            guard text.shape == [1, 512, frames], noise.shape == [1, frames, 600, 9] else {
                throw AetherError.invalidTensor("misaligned decoder inputs")
            }
            observed.withLock { $0.append((noise.values[0], style.values[0], Array(prosody.values[0..<frames]))) }
            var valid: Int?
            if case .int32(let count) = inputs["frames"] { valid = Int(count.values[0]) }
            let used = valid ?? frames
            let zeroTail = (0..<640).allSatisfy { c in (used..<frames).allSatisfy { prosody.values[c * frames + $0] == 0 } }
            calls.withLock { $0.append(DecoderCall(shape: frames, validFrames: valid, noiseCount: noise.values.count, paddingIsZero: zeroTail)) }
            return ["audio": .float32(try FloatTensor(shape: [1, frames * 600], values: (0..<(frames * 600)).map { Float($0 / 600) / 1000 }))]
        }, close: { log.append("close:decode") })
    }
}
