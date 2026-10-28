import AetherKit
import AetherH3
import Foundation

@available(macOS 27, iOS 27, *)
actor NativeH3DenoisingBackend: H3DenoisingBackend, H3DenoisingKernels {
    private let plan: H3DenoisingPlan
    private let observer: (@Sendable (H3DenoisingExecutionEvent) -> Void)?
    private var prepare: NativeH3Function?
    private var heads: NativeH3Function?
    private var time: NativeH3Function?
    private var body: NativeH3ChainBackend?
    private var bodyCardinality: Int?
    private var embeddings: [H3TensorData] = []
    private var tables: [[H3TensorData]] = []
    init(plan: H3DenoisingPlan, observer: (@Sendable (H3DenoisingExecutionEvent) -> Void)?) {
        self.plan = plan; self.observer = observer
    }
    private func observe(_ stage: String, step: Int, block: Int? = nil) -> @Sendable (H3BlockEvent) -> Void {
        let observer = self.observer
        return { observer?(H3DenoisingExecutionEvent(stage: stage, step: step, block: block, event: $0)) }
    }
    func run(initial: H3Latents, conditioning: H3TensorData, onOutput: @escaping H3DenoisingOutputHandler) async throws -> H3Latents {
        guard initial.video.spec == plan.inputVideo.spec, initial.audio.spec == plan.inputAudio.spec,
              conditioning.spec == plan.conditioning.spec else { throw AetherError.invalidTensor("H3 trajectory request geometry or conditioning differs") }
        // These functions retain their stream across runs. A cancelled run leaves
        // no committed latent state; the next run restarts its supplied input.
        if time == nil { time = try await NativeH3Function.open(plan.timeModel) }
        var embeddings: [H3TensorData] = []
        let rows = plan.schedule.steps.map { plan.layout.timesteps(for: $0) }
        for (step, row) in rows.enumerated() {
            let timestep = try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [row.values.count]), bytes: row.values.withUnsafeBytes { Data($0) })
            let output = try await time!.run(inputs: ["timestep": timestep],
                outputs: ["temb": H3TensorSpec(scalarType: .float32, shape: [row.values.count,2688])], observer: observe("time", step: step))
            embeddings.append(output["temb"]!)
            try await onOutput(H3DenoisingOutput(stage: "time", step: step, tensors: output))
        }
        // One modulation model at a time; only the small owned BF16 tables are
        // retained across all eight steps, never fifty native model/workspaces.
        var tables = Array(repeating: [H3TensorData](), count: 50)
        for block in 0..<50 {
            let modulation = try await NativeH3Function.open(plan.modulations[block])
            for step in 0..<8 {
                let output = try await modulation.run(inputs: ["temb": embeddings[step]],
                    outputs: ["modulations": H3TensorSpec(scalarType: .bfloat16, shape: [6,rows[step].values.count * 3,5376])],
                    observer: observe("modulation", step: step, block: block))
                tables[block].append(output["modulations"]!)
                try await onOutput(H3DenoisingOutput(stage: "modulation", step: step, block: block, tensors: output))
            }
        }
        self.embeddings = embeddings; self.tables = tables
        return try await H3DenoisingLoop.run(initial: initial, conditioning: conditioning, schedule: plan.schedule,
            kernels: self, onOutput: onOutput)
    }
    func prepare(step: Int, latents: H3Latents, conditioning: H3TensorData) async throws -> H3TensorData {
        if prepare == nil { prepare = try await NativeH3Function.open(plan.prepareModel) }
        let output = try await prepare!.run(inputs: ["video": latents.video,"audio": latents.audio,"text": conditioning],
            outputs: ["hidden": plan.backbone.cases[0].base.output], observer: observe("prepare", step: step))
        return output["hidden"]!
    }
    func backbone(step: Int, hidden: H3TensorData, onOutput: @escaping H3DenoisingOutputHandler) async throws -> H3TensorData {
        let rows = plan.layout.timesteps(for: plan.schedule.steps[step]), cardinality = rows.values.count
        if bodyCardinality != cardinality {
            body = nil; bodyCardinality = nil // Previous work is already joined.
            body = try await NativeH3ChainBackend.open(plan.backbone.cases[cardinality - 1], observer: nil)
            bodyCardinality = cardinality
        }
        let tags = try H3TensorData(spec: H3TensorSpec(scalarType: .int32, shape: [plan.layout.rows]), bytes: rows.adalnIndices.withUnsafeBytes { Data($0) })
        let tail = H3DenoisingTail(), observer = self.observer
        try await body!.run(initial: hidden, tables: tables.map { $0[step] }, tags: tags,
            executionObserver: { observer?(H3DenoisingExecutionEvent(stage: "body", step: step, block: $0.block, event: $0.event)) }) { block, value in
                try await onOutput(H3DenoisingOutput(stage: "body", step: step, block: block, tensors: ["hidden": value]))
                if block == 49 { await tail.set(value) }
            }
        guard let value = await tail.value else { throw AetherError.invalidTensor("H3 trajectory did not reach block 49") }
        return value
    }
    func heads(step: Int, hidden: H3TensorData, latents: H3Latents) async throws -> H3Latents {
        if heads == nil { heads = try await NativeH3Function.open(plan.headsModel) }
        let rows = plan.layout.timesteps(for: plan.schedule.steps[step])
        let indices = try H3TensorData(spec: H3TensorSpec(scalarType: .int32, shape: [plan.layout.rows]), bytes: rows.indices.withUnsafeBytes { Data($0) })
        let output = try await heads!.run(inputs: ["hidden": hidden,"temb": embeddings[step],"indices": indices],
            outputs: ["video": latents.video.spec,"audio": latents.audio.spec], observer: observe("heads", step: step))
        return try H3Latents(video: output["video"]!, audio: output["audio"]!)
    }
}
@available(macOS 27, iOS 27, *)
private actor H3DenoisingTail {
    var value: H3TensorData?
    func set(_ value: H3TensorData) { self.value = value }
}
