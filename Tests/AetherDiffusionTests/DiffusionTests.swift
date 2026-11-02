import AetherDiffusion
import AetherKit
import AetherSDTurbo
import Foundation
import Testing

@Test func epsilonScheduleUsesEpsilonAndInputScaling() throws {
    let s = try EpsilonSchedule(timesteps: [999, 499], sigmas: [2, 1, 0], initialNoiseSigma: 2)
    let x = try FloatTensor(shape: [1], values: [6]), e = try FloatTensor(shape: [1], values: [2])
    #expect(try s.scaledInput(x, step: 0).values[0] == 6 / sqrt(Float(5)))
    #expect(try s.advance(x, epsilon: e, step: 0).values == [4])
    #expect(try s.advance(x, epsilon: e, step: 1).values == [4])
    #expect(throws: (any Error).self) { try s.advance(x, epsilon: e, step: 2) }
    #expect(throws: (any Error).self) { try EpsilonSchedule(timesteps: [1], sigmas: [0, 0], initialNoiseSigma: 1) }
    #expect(throws: (any Error).self) { try EpsilonSchedule(timesteps: [1], sigmas: [1, 0], initialNoiseSigma: .nan) }
    #expect(throws: (any Error).self) {
        try JSONDecoder().decode(EpsilonSchedule.self, from: Data(#"{"timesteps":[1],"sigmas":[0,0],"initialNoiseSigma":1}"#.utf8))
    }
}

private actor ToyBackend: EpsilonDiffusionBackend {
    var releases = 0
    var calls: [String] = []
    var fail = false
    func setFail(_ value: Bool) { fail = value }
    func encode(_ tokenIDs: Int32Tensor) async throws -> FloatTensor {
        calls.append("encode")
        if fail { throw AetherError.invalidModel("fixture failure") }
        return try FloatTensor(shape: [1], values: [0])
    }
    func denoise(sample: FloatTensor, timestep: Float, conditioning: FloatTensor) async throws -> FloatTensor {
        calls.append("denoise")
        return try FloatTensor(shape: sample.shape, values: Array(repeating: 1, count: sample.values.count))
    }
    func decode(_ latents: FloatTensor) async throws -> GeneratedImage {
        calls.append("decode")
        return try GeneratedImage(pixelsCHW: FloatTensor(shape: [3, 1, 1], values: Array(repeating: latents.values[0], count: 3)))
    }
    func unload() { releases += 1 }
}

@Test func orchestrationReleaseFailureRecoveryAndClose() async throws {
    let backend = ToyBackend(), session = DiffusionSession(backend: backend)
    let ids = try Int32Tensor(shape: [1, 1], values: [0]), noise = try FloatTensor(shape: [1, 1, 1, 1], values: [3])
    let s = try EpsilonSchedule(timesteps: [9], sigmas: [2, 0], initialNoiseSigma: 2)
    let result = try await session.generate(tokenIDs: ids, noise: noise, schedule: s)
    #expect(result.pixelsCHW.values == [4, 4, 4])
    #expect(await backend.calls == ["encode", "denoise", "decode"])
    #expect(await backend.releases == 1)
    await backend.setFail(true)
    await #expect(throws: (any Error).self) { try await session.generate(tokenIDs: ids, noise: noise, schedule: s) }
    #expect(await backend.releases == 2)
    await backend.setFail(false)
    _ = try await session.generate(tokenIDs: ids, noise: noise, schedule: s)
    await session.unload()
    await #expect(throws: AetherError.sessionClosed) { try await session.generate(tokenIDs: ids, noise: noise, schedule: s) }
}

private actor GateBackend: EpsilonDiffusionBackend {
    private var entered = false
    private var waiting: CheckedContinuation<Void, Never>?
    private var observers: [CheckedContinuation<Void, Never>] = []
    var decoded = false
    var releases = 0
    func waitForEntry() async {
        if entered { return }
        await withCheckedContinuation { observers.append($0) }
    }
    func resume() { waiting?.resume(); waiting = nil }
    func encode(_ tokenIDs: Int32Tensor) async throws -> FloatTensor {
        await withCheckedContinuation { continuation in
            waiting = continuation; entered = true
            for observer in observers { observer.resume() }; observers = []
        }
        return try FloatTensor(shape: [1], values: [0])
    }
    func denoise(sample: FloatTensor, timestep: Float, conditioning: FloatTensor) throws -> FloatTensor {
        try FloatTensor(shape: sample.shape, values: Array(repeating: 0, count: sample.values.count))
    }
    func decode(_ latents: FloatTensor) throws -> GeneratedImage {
        decoded = true
        return try GeneratedImage(pixelsCHW: FloatTensor(shape: [3, 1, 1], values: [0, 0, 0]))
    }
    func unload() { releases += 1 }
}

@Test func diffusionCancellationJoinsBackendAndRejectsOverlap() async throws {
    let backend = GateBackend(), session = DiffusionSession(backend: backend)
    let ids = try Int32Tensor(shape: [1, 1], values: [0]), noise = try FloatTensor(shape: [1, 1, 1, 1], values: [0])
    let schedule = try EpsilonSchedule(timesteps: [1], sigmas: [1, 0], initialNoiseSigma: 1)
    let run = Task { try await session.generate(tokenIDs: ids, noise: noise, schedule: schedule) }
    await backend.waitForEntry()
    await #expect(throws: AetherError.sessionBusy) { try await session.generate(tokenIDs: ids, noise: noise, schedule: schedule) }
    run.cancel()
    #expect(await backend.releases == 0)
    await backend.resume()
    await #expect(throws: CancellationError.self) { try await run.value }
    #expect(await backend.releases == 1)
    #expect(await backend.decoded == false)
    await session.unload()
}

@Test func bundleChecksHashesGeometryAndPaths() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = ["textEncoder.aimodel/file", "denoiser.aimodel/file", "decoder.aimodel/file", "tokenizer/file"]
    for path in paths {
        let file = root.appendingPathComponent(path)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data().write(to: file)
    }
    let schedule: [String: Any] = ["timesteps": [999], "sigmas": [1, 0], "initialNoiseSigma": 1]
    var schedules: [String: Any] = [:]
    for n in 1...4 {
        var s = schedule
        s["timesteps"] = (1...n).reversed().map { $0 * 100 }
        s["sigmas"] = (0...n).reversed().map(Float.init)
        schedules[String(n)] = s
    }
    let emptyHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    var manifest: [String: Any] = ["schema": 1, "family": "sd-turbo", "source": ["repository": "stabilityai/sd-turbo", "revision": SDTurboBundle.sourceRevision],
        "width": 512, "height": 512, "latentChannels": 4, "latentScale": 8, "textLength": 77, "textWidth": 1024,
        "scalarType": "float32", "schedules": schedules, "files": Dictionary(uniqueKeysWithValues: paths.map { ($0, emptyHash) })]
    func save() throws { try JSONSerialization.data(withJSONObject: manifest).write(to: root.appendingPathComponent("diffusion.json")) }
    try save()
    let bundle = try SDTurboBundle(at: root)
    #expect(try bundle.schedule(steps: 4).timesteps.count == 4)
    #expect(try bundle.noise(seed: 42) == bundle.noise(seed: 42))
    #expect(throws: (any Error).self) { try bundle.schedule(steps: 5) }
    try Data([1]).write(to: root.appendingPathComponent(paths[0]))
    #expect(throws: (any Error).self) { try SDTurboBundle(at: root) }
    try Data().write(to: root.appendingPathComponent(paths[0]))
    manifest["width"] = 513; try save()
    #expect(throws: (any Error).self) { try SDTurboBundle(at: root) }
    manifest["width"] = 512
    manifest["files"] = ["../escape": emptyHash]; try save()
    #expect(throws: (any Error).self) { try SDTurboBundle(at: root) }
}
