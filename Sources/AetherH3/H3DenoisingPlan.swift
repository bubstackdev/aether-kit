import AetherKit
import Foundation

/// Eight-step T2VA probe using provided conditioning and latents. The two body
/// profiles supply verified graph/weight layouts; their fixed-input references
/// and hashes do not qualify the new runtime contexts.
public struct H3DenoisingPlan: Sendable {
    public let phases: H3PhaseProbePlan
    public let backbone: H3ChainProbePlan
    public let modulations: [H3ModelAsset]
    public let layout: H3Layout
    public let schedule: H3Schedule
    public let inputVideo: H3TensorFile
    public let inputAudio: H3TensorFile
    public let conditioning: H3TensorFile
    public let timeModel: H3ModelAsset
    public let prepareModel: H3ModelAsset
    public let headsModel: H3ModelAsset

    public init(bundle: URL, phases: URL, cpuRecord: URL, chain: URL, modulationDirectory: URL) throws {
        self.phases = try H3PhaseProbePlan(bundle: bundle, phases: phases, cpuRecord: cpuRecord)
        backbone = try H3ChainProbePlan(bundle: bundle, phases: phases, chain: chain, requireReferences: true)
        guard backbone.cases.allSatisfy({ $0.streamsWeights && $0.steps.count == 50 }),
              backbone.blockPlan.header.manifestSHA256 == self.phases.header.manifestSHA256,
              backbone.blockPlan.phaseManifestSHA256 == self.phases.phaseManifestSHA256,
              modulationDirectory.isFileURL else { throw AetherError.invalidModel("H3 trajectory requires the same corrected fifty-block/outer profiles") }
        let data = try Data(contentsOf: bundle)
        guard h3SHA256(data) == self.phases.header.manifestSHA256 else { throw AetherError.invalidModel("H3 trajectory bundle changed") }
        let source = try JSONDecoder().decode(Source.self, from: data)
        guard source.modulations.count == 50 else { throw AetherError.invalidModel("H3 trajectory requires fifty modulation graphs") }
        modulations = try source.modulations.enumerated().map { block, item in
            // Exports retain their old absolute location after the prototype was
            // moved. Relocation is explicit; original stream hashes still apply.
            try H3ModelAsset(url: modulationDirectory.appendingPathComponent("modulation-\(block).aimodel"), function: item.function, sha256: item.sha256)
        }
        let prepare = self.phases.cases[0]
        guard let video = prepare.inputs.first(where: { $0.name == "video" }),
              let audio = prepare.inputs.first(where: { $0.name == "audio" }),
              let text = prepare.inputs.first(where: { $0.name == "text" }) else {
            throw AetherError.invalidModel("missing H3 trajectory inputs")
        }
        inputVideo = video; inputAudio = audio; conditioning = text
        layout = try H3Layout(geometry: backbone.blockPlan.geometry, textTags: Array(repeating: 1, count: text.spec.shape[1]))
        schedule = try H3Schedule()
        prepareModel = prepare.model; timeModel = self.phases.cases[1].model; headsModel = self.phases.cases[2].model
        guard schedule.steps.map({ layout.timesteps(for: $0).values.count }) == [1,2,2,2,2,2,2,2] else {
            throw AetherError.invalidModel("unsupported H3 trajectory timestep cardinalities")
        }
    }
    /// CPU-only file verification. Embedded modulation graphs are verified in
    /// addition to the previously selected outer/body assets and tensor files.
    public func verify() throws {
        try phases.verify(); try backbone.verify()
        for model in modulations { try model.verify() }
    }
    private struct Source: Decodable {
        let modulations: [Model]
        struct Model: Decodable { let function: String; let sha256: String }
    }
}

/// CPU-owned runtime state. BF16 hidden states remain raw bytes; scheduler
/// updates operate only on the original float32 video/audio latent values.
public struct H3Latents: Sendable {
    public let video: H3TensorData
    public let audio: H3TensorData
    public init(video: H3TensorData, audio: H3TensorData) throws {
        guard video.spec.scalarType == .float32, audio.spec.scalarType == .float32,
              video.spec.shape.count == 3, audio.spec.shape.count == 3,
              video.spec.shape[0] == 1, audio.spec.shape[0] == 1,
              video.spec.shape[2] == 96, audio.spec.shape[2] == 32 else {
            throw AetherError.invalidTensor("H3 latents must preserve F32 video/audio geometry")
        }
        self.video = video; self.audio = audio
    }
    /// One scheduler step. The first `conditionVideoRows` video rows (the keyframe anchors) are never written, as in the
    /// pipeline, which steps only `latents[num_condition_video_rows:]`.
    public func advanced(schedule: H3Schedule, step: Int, videoVelocity: H3TensorData, audioVelocity: H3TensorData,
                         conditionVideoRows: Int = 0) throws -> H3Latents {
        guard videoVelocity.spec == video.spec, audioVelocity.spec == audio.spec,
              (0..<video.spec.shape[1]).contains(conditionVideoRows) else {
            throw AetherError.invalidTensor("H3 velocity shape or precision differs from latents")
        }
        func advance(_ modality: H3Schedule.Modality, _ tensor: H3TensorData, _ velocity: H3TensorData, frozen: Int) throws -> H3TensorData {
            let sample = try tensor.floatValues(), velocity = try velocity.floatValues()
            let moved = try schedule.advance(modality, step: step, sample: Array(sample[frozen...]), velocity: Array(velocity[frozen...]))
            let values = Array(sample[..<frozen]) + moved
            return try H3TensorData(spec: tensor.spec, bytes: values.withUnsafeBytes { Data($0) })
        }
        return try H3Latents(video: advance(.video, video, videoVelocity, frozen: conditionVideoRows * 96),
                             audio: advance(.audio, audio, audioVelocity, frozen: 0))
    }
}
