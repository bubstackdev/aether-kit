import AetherH3
import AetherCoreAI
import AetherKit
import Foundation
import Testing
@testable import AetherH3CoreAI

private struct LoopOracle: Decodable, Sendable {
    let video: [Float]; let audio: [Float]; let steps: [Step]
    struct Step: Decodable, Sendable {
        let video_velocity: [Float]; let audio_velocity: [Float]
        let video_sha256: String; let audio_sha256: String
    }
    static func load() throws -> Self {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: #require(Bundle.module.url(forResource: "turbo8-numerics", withExtension: "json", subdirectory: "Fixtures"))))
    }
    func initial() throws -> H3Latents { try H3Latents(video: tensor(video+video+video, width:96), audio: tensor(audio, width:32)) }
}
private func tensor(_ values: [Float], width: Int) throws -> H3TensorData {
    try H3TensorData(spec: H3TensorSpec(scalarType:.float32, shape:[1,values.count/width,width]), bytes:values.withUnsafeBytes { Data($0) })
}
private func conditioning() throws -> H3TensorData {
    try H3TensorData(spec: H3TensorSpec(scalarType:.bfloat16,shape:[1,1,1]),bytes:Data([128,63]))
}
private actor LoopKernels: H3DenoisingKernels {
    let oracle: LoopOracle
    var calls: [String] = []
    init(_ oracle: LoopOracle) { self.oracle = oracle }
    func prepare(step: Int, latents: H3Latents, conditioning: H3TensorData) throws -> H3TensorData {
        calls.append("prepare\(step)")
        let previousVideo = step == 0 ? try oracle.initial().video.bytes.prefix(128) : nil
        let videoHash = h3SHA256(latents.video.bytes.prefix(128)), audioHash = latents.audio.sha256
        let expectedVideo = step == 0 ? h3SHA256(Data(previousVideo!)) : oracle.steps[step-1].video_sha256
        let expectedAudio = step == 0 ? try oracle.initial().audio.sha256 : oracle.steps[step-1].audio_sha256
        guard videoHash == expectedVideo, audioHash == expectedAudio, conditioning.bytes == Data([128,63]) else { throw AetherError.invalidTensor("stale latent/conditioning carry") }
        return try tensor([Float(step+1)],width:1)
    }
    func backbone(step: Int, hidden: H3TensorData, onOutput: @escaping H3DenoisingOutputHandler) async throws -> H3TensorData {
        guard try hidden.floatValues() == [Float(step+1)] else { throw AetherError.invalidTensor("stale prepared hidden") }
        calls.append("body\(step)")
        var value = hidden
        for block in 0..<50 {
            try Task.checkCancellation(); value = try tensor([Float(100+step*50+block)],width:1)
            try await onOutput(H3DenoisingOutput(stage:"body",step:step,block:block,tensors:["hidden":value]))
        }
        return value
    }
    func heads(step: Int, hidden: H3TensorData, latents: H3Latents) throws -> H3Latents {
        guard try hidden.floatValues() == [Float(149+step*50)] else { throw AetherError.invalidTensor("head did not receive block49") }
        calls.append("heads\(step)"); let value = oracle.steps[step]
        return try H3Latents(video:tensor(value.video_velocity+value.video_velocity+value.video_velocity,width:96), audio:tensor(value.audio_velocity,width:32))
    }
}
private actor LoopEvents {
    var blocks: [String] = []
    var latents: [[String:String]] = []
    func record(_ event: H3DenoisingOutput) {
        if let block = event.block { blocks.append("\(event.step).\(block)") }
        if event.stage == "latents" { latents.append(event.tensors.mapValues(\.sha256)) }
    }
}
private actor LoopBackend: H3DenoisingBackend {
    let oracle: LoopOracle
    init(_ oracle: LoopOracle) { self.oracle = oracle }
    func run(initial: H3Latents, conditioning: H3TensorData, onOutput: @escaping H3DenoisingOutputHandler) async throws -> H3Latents {
        try await H3DenoisingLoop.run(initial:initial,conditioning:conditioning,schedule:H3Schedule(),kernels:LoopKernels(oracle),onOutput:onOutput)
    }
}
@Test func h3LinkedLoopCarriesBothModalitiesThroughEightOracleUpdatesAndAllBlockBoundaries() async throws {
    let oracle = try LoopOracle.load(), kernels = LoopKernels(oracle), events = LoopEvents(), initial = try oracle.initial()
    let output = try await H3DenoisingLoop.run(initial:initial,conditioning:conditioning(),schedule:H3Schedule(),kernels:kernels) { await events.record($0) }
    #expect(await kernels.calls == (0..<8).flatMap { ["prepare\($0)","body\($0)","heads\($0)"] })
    #expect(await events.blocks == (0..<8).flatMap { step in (0..<50).map { "\(step).\($0)" } })
    #expect(await events.latents.map { $0["audio"]! } == oracle.steps.map(\.audio_sha256))
    #expect(h3SHA256(output.video.bytes.prefix(128)) == oracle.steps[7].video_sha256)
    #expect(output.audio.sha256 == oracle.steps[7].audio_sha256)
    #expect(initial.audio.sha256 == (try oracle.initial().audio.sha256))
}
@Test func h3LinkedConsumerFailureRestartsOriginalStateOnRetry() async throws {
    let oracle = try LoopOracle.load(), initial = try oracle.initial(), session = H3DenoisingSession(backend:LoopBackend(oracle))
    await #expect(throws:AetherError.invalidTensor("consumer failed")) {
        try await session.run(initial:initial,conditioning:conditioning()) {
            if $0.stage == "heads" && $0.step == 3 { throw AetherError.invalidTensor("consumer failed") }
        }
    }
    #expect(await session.state == .ready)
    let output = try await session.run(initial:initial,conditioning:conditioning())
    #expect(output.audio.sha256 == oracle.steps[7].audio_sha256)
    await session.unload()
    await #expect(throws:AetherError.sessionClosed) { try await session.run(initial:initial,conditioning:conditioning()) }
}
private actor ConsumerGate {
    var entered = false
    var waiter: CheckedContinuation<Void,Never>?
    var started: CheckedContinuation<Void,Never>?
    func hold() async { await withCheckedContinuation { waiter=$0;entered=true;started?.resume();started=nil } }
    func wait() async { if entered{return};await withCheckedContinuation { started=$0 } }
    func release(){waiter?.resume();waiter=nil}
}
@Test(.timeLimit(.minutes(1))) func h3LinkedCancellationAndUnloadWaitForConsumerThenRejectLaterStages() async throws {
    let oracle = try LoopOracle.load(), initial = try oracle.initial(), gate = ConsumerGate()
    let session = H3DenoisingSession(backend:LoopBackend(oracle))
    let task = Task { try await session.run(initial:initial,conditioning:conditioning()) { if $0.stage == "input" { await gate.hold() } } }
    await gate.wait();task.cancel()
    await #expect(throws:AetherError.sessionBusy) { try await session.run(initial:initial,conditioning:conditioning()) }
    let closing = Task { await session.unload() }
    let deadline = ContinuousClock.now.advanced(by:.seconds(5))
    while await session.state != .closing, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await session.state == .closing)
    await gate.release()
    await #expect(throws:CancellationError.self) { try await task.value }
    await closing.value;#expect(await session.state == .closed)
}
@Test func h3RuntimeDescriptorGateAdmitsOnlyActualDynamicDimensionsAndNamedPrecision() throws {
    let tensor:[String:Any] = ["kind":"tensor","scalarType":"float32","shape":[-1,2688]]
    let raw:[String:Any] = ["name":"main","inputs":["temb":tensor],"outputs":["result":tensor],"states":[:]]
    let descriptor = try JSONDecoder().decode(ModelFunctionDescription.self,from:JSONSerialization.data(withJSONObject:raw))
    for n in [1,2] {
        let spec = try H3TensorSpec(scalarType:.float32,shape:[n,2688])
        try H3RuntimeBindings.validate(descriptor,inputs:["temb":spec],outputs:["result":spec])
    }
    let wrong = try H3TensorSpec(scalarType:.bfloat16,shape:[2,2688]), good = try H3TensorSpec(scalarType:.float32,shape:[2,2688])
    #expect(throws:(any Error).self) { try H3RuntimeBindings.validate(descriptor,inputs:["temb":wrong],outputs:["result":good]) }
    #expect(throws:(any Error).self) { try H3RuntimeBindings.validate(descriptor,inputs:["temb":good],outputs:["other":good]) }
    #expect(throws:(any Error).self) { try H3RuntimeBindings.validate(descriptor,inputs:["temb":good],outputs:["result":H3TensorSpec(scalarType:.float32,shape:[2,2689])]) }
}
