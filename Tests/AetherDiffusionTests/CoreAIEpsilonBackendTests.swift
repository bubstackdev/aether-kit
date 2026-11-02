import AetherCoreAI
import AetherKit
@testable import AetherDiffusionCoreAI
import Foundation
import Testing

private actor StageGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var observers: [CheckedContinuation<Void, Never>] = []
    private var entered = false
    func block() async {
        await withCheckedContinuation { continuation in
            self.continuation = continuation; entered = true
            for observer in observers { observer.resume() }; observers.removeAll()
        }
    }
    func waitForEntry() async {
        if !entered { await withCheckedContinuation { observers.append($0) } }
    }
    func release() { continuation?.resume(); continuation = nil }
}

private actor NativeStage: StagedModel {
    nonisolated let functions: [ModelFunctionDescription]
    let gate: StageGate?
    var releases = 0
    var running = false
    let entry: String
    init(entry: String, gate: StageGate? = nil) throws {
        self.entry = entry; self.gate = gate
        func value(_ shape: [Int], _ type: String = "float32") -> [String: Any] {
            ["kind": "tensor", "scalarType": type, "shape": shape]
        }
        let inputs: [String: Any], outputs: [String: Any]
        switch entry {
        case "encode": inputs = ["input_ids": value([1, 2], "int32")]; outputs = ["text_embeds": value([1, 2, 3])]
        case "denoise": inputs = ["sample": value([1, 4, 1, 1]), "timestep": value([1]), "text_embeds": value([1, 2, 3])]; outputs = ["epsilon": value([1, 4, 1, 1])]
        default: inputs = ["latents": value([1, 4, 1, 1])]; outputs = ["image": value([3, 8, 8])]
        }
        let data = try JSONSerialization.data(withJSONObject: [["name": entry, "inputs": inputs, "outputs": outputs, "states": [:]]])
        functions = try JSONDecoder().decode([ModelFunctionDescription].self, from: data)
    }
    func runTensors(function: String, inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        running = true; defer { running = false }
        await gate?.block()
        let output = functions[0].outputs.first!
        let shape = output.value.shape!
        return [output.key: .float32(try FloatTensor(shape: shape, values: Array(repeating: 0, count: shape.reduce(1, *))))]
    }
    func unload() {
        #expect(!running, "release must join the active native call")
        releases += 1
    }
}

private actor Completion {
    var finished = false
    func finish() { finished = true }
}

@Test(arguments: ["encode", "denoise", "decode"])
func nativeBackendReleaseJoinsEveryStage(entry: String) async throws {
    let gate = StageGate(), completed = Completion()
    let model = try NativeStage(entry: entry, gate: gate)
    let url = URL(filePath: "/test/model")
    let backend = try CoreAIEpsilonBackend(textEncoder: url, denoiser: url, decoder: url, width: 8, height: 8,
        latentChannels: 4, latentScale: 8, textLength: 2, textWidth: 3, vocabularySize: 10, openModel: { _ in model })
    let ids = try Int32Tensor(shape: [1, 2], values: [1, 2])
    let latent = try FloatTensor(shape: [1, 4, 1, 1], values: [0, 0, 0, 0])
    let text = try FloatTensor(shape: [1, 2, 3], values: Array(repeating: 0, count: 6))
    let operation = Task {
        switch entry {
        case "encode": _ = try await backend.encode(ids)
        case "denoise": _ = try await backend.denoise(sample: latent, timestep: 1, conditioning: text)
        default: _ = try await backend.decode(latent)
        }
    }
    await gate.waitForEntry()
    let release = Task { await backend.unload(); await completed.finish() }
    // A stage call submitted behind release either sees its active predecessor or the release barrier.
    await #expect(throws: AetherError.sessionBusy) { try await backend.encode(ids) }
    #expect(await completed.finished == false)
    #expect(await model.releases == 0)
    await gate.release()
    try await operation.value
    await release.value
    #expect(await completed.finished)
    #expect(await model.releases == 1)
    await backend.unload()
    #expect(await model.releases == 1)
}
