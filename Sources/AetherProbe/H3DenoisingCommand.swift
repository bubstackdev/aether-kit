import AetherKit
import AetherH3
import AetherH3CoreAI
import Darwin
import Foundation
import Metal
import Synchronization

struct H3TrajectoryTensor: Codable, Sendable {
    let key: String
    let stage: String
    let step: Int
    let block: Int?
    let name: String
    let dtype: H3ScalarType
    let shape: [Int]
    let sha256: String
    let path: String?
}
private struct TrajectoryReference: Decodable, Sendable {
    let key: String
    let context: [String:String]
    let tensor: Tensor
    struct Tensor: Decodable, Sendable {
        let path: String; let dtype: H3ScalarType; let shape: [Int]; let sha256: String
        let row_indices: [Int]?
        func file(base: URL) throws -> H3TensorFile {
            guard !path.isEmpty, !path.contains("://") else { throw AetherError.invalidModel("H3 trajectory references must be local") }
            return try H3TensorFile(name:"reference",url:URL(fileURLWithPath:path,relativeTo:base.deletingLastPathComponent()),
                spec:H3TensorSpec(scalarType:dtype,shape:shape),sha256:sha256)
        }
    }
}
private struct TrajectoryReferences: Decodable, Sendable {
    let schema: Int
    let bundle_sha256: String
    let phase_sha256: String
    let chain_sha256: String
    let cpu_record_sha256: String
    let modulation_model_sha256: [String]
    let capture_report_path: String
    let capture_report_sha256: String
    let oracle_recipe_sha256: String
    let expected: [String:String]
    let references: [TrajectoryReference]
}
private struct TrajectoryComparison: Encodable, Sendable {
    let key: String
    let full: H3TensorComparison?
    let sampled: H3ReferenceComparison?
    var passed: Bool { full?.passed ?? sampled?.passed ?? false }
}
private struct TrajectoryLifecycleEvidence: Codable {
    var runs: [String] = []
    var traces: [String:[String]] = [:]
    var settledGrowthLimitBytes: UInt64 = 512 << 20
    var completeRuns = 3
    var cancelledRuns = 2
    var totalOutputCount = 3462
    var totalCPUComparisons = 2212
}
private struct TrajectoryReport: Codable {
    var schema = 1
    var scope = "Eight linked Turbo8 steps with provided conditioning, native time/modulation/prepare/50-block/head stages and coupled CPU F32 latent updates; sampled body checkpoints only, not independent CPU trajectory, conditioning encoder, decoders, media acceptance or benchmark."
    let mode: String
    var stage = "preparation"
    var passed = false
    var captureComplete = false
    var nativeExecuted = false
    var selectedFilesVerified = false
    var readyForQualification = false
    var bundleSHA256: String?
    var phaseSHA256: String?
    var cpuRecordSHA256: String?
    var chainSHA256: String?
    var referenceSHA256: String?
    var modulationModelSHA256: [String] = []
    var lifecycle: TrajectoryLifecycleEvidence?
    var nativeSubmissionBudget = 824
    var expectedOutputCount = 864
    var requiredCPUComparisons = 464
    var physicalFootprintLimitBytes: UInt64 = 20 << 30
    var metalAllocationLimitBytes: UInt64 = 18 << 30
    var resources: [String:BenchmarkMemory] = [:]
    var sampledPeak: BenchmarkMemory?
    var memorySamples = 0
    var error: String?
    // Live per-boundary records are persisted separately in progress.json.
    // Final publication retains metadata only, never an array of tensors.
    var outputs: [H3TrajectoryTensor] = []
}
private struct TrajectoryProgress: Encodable, Sendable {
    var outputs: [H3TrajectoryTensor] = []
    var comparisons: [TrajectoryComparison] = []
}
private func trajectoryKey(_ stage: String, _ step: Int, _ block: Int?, _ name: String) -> String {
    "\(stage).\(step).\(block ?? -1).\(name)"
}
private func isDigest(_ value:String) -> Bool { value.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } }

private actor TrajectoryRecorder {
    private let output: URL
    private let references: TrajectoryReferences?
    private let referenceURL: URL?
    private let sources: [String:String]
    private let specs: [String:H3TensorSpec]
    private let events: [String]
    private var eventIndex = 0
    private var progress = TrajectoryProgress()
    private var identities: [String:String]
    private let sampler: BenchmarkMemorySampler
    init(output: URL, references: TrajectoryReferences?, referenceURL: URL?, sources: [String:String],
         specs: [String:H3TensorSpec], events: [String], sampler: BenchmarkMemorySampler) {
        self.output=output;self.references=references;self.referenceURL=referenceURL;self.sources=sources
        self.specs=specs;self.events=events;self.identities=sources;self.sampler=sampler
    }
    func record(_ event: H3DenoisingOutput) throws {
        try Task.checkCancellation()
        let eventKey = "\(event.stage).\(event.step).\(event.block ?? -1)"
        guard eventIndex < events.count, eventKey == events[eventIndex] else { throw AetherError.invalidTensor("H3 trajectory boundary order differs") }
        eventIndex += 1
        let expectedNames = Set(specs.keys.filter { $0.hasPrefix(eventKey + ".") }.map { String($0.split(separator:".").last!) })
        guard Set(event.tensors.keys) == expectedNames else { throw AetherError.invalidTensor("H3 trajectory named outputs differ") }
        for name in event.tensors.keys.sorted() {
            let value=event.tensors[name]!, key=trajectoryKey(event.stage,event.step,event.block,name)
            guard specs[key] == value.spec, identities[key] == nil else { throw AetherError.invalidTensor("H3 trajectory output shape, precision or uniqueness differs") }
            let hash=value.sha256
            var path: String?
            if references == nil {
                let save = event.stage != "body" || [48,49].contains(event.block ?? -1)
                if save {
                    let url=output.appendingPathComponent(key + ".raw")
                    try value.bytes.write(to:url,options:.withoutOverwriting);path=url.path
                }
            }
            identities[key]=hash
            progress.outputs.append(H3TrajectoryTensor(key:key,stage:event.stage,step:event.step,block:event.block,name:name,
                dtype:value.spec.scalarType,shape:value.spec.shape,sha256:hash,path:path))
            try write()
            if let references {
                guard references.expected[key] == hash else { throw AetherError.invalidTensor("H3 trajectory output changed from its captured context: \(key)") }
                if let reference=references.references.first(where:{$0.key == key}), let referenceURL {
                    guard reference.context.allSatisfy({ identities[$0.key] == $0.value }) else { throw AetherError.invalidTensor("H3 CPU reference context changed: \(key)") }
                    let cpu = try reference.tensor.file(base:referenceURL).read()
                    let comparison:TrajectoryComparison
                    if let rows=reference.tensor.row_indices {
                        comparison=TrajectoryComparison(key:key,full:nil,sampled:try H3ReferenceComparison(actual:value,reference:cpu,rows:rows))
                    } else { comparison=TrajectoryComparison(key:key,full:try H3TensorComparison(actual:value,reference:cpu),sampled:nil) }
                    progress.comparisons.append(comparison);try write()
                    guard comparison.passed else { throw AetherError.invalidTensor("H3 trajectory CPU numerical gate failed: \(key)") }
                }
            }
        }
        if let error=sampler.snapshot.error { throw AetherError.unsupported(error) }
        if let peak=sampler.snapshot.peak { try H3LifecycleLimits().check(peak) }
    }
    private func write() throws {
        let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
        try encoder.encode(progress).write(to:output.appendingPathComponent("progress.json"),options:.atomic)
    }
    func cancelledPrefix() throws {
        // Eight time + 400 modulation outputs, two input arms, prepare and
        // blocks 0...23 are delivered. Submitted block 24 is joined/discarded.
        guard eventIndex == 434, progress.outputs.count == 435,
              progress.comparisons.count == 410 else { throw AetherError.invalidTensor("cancelled trajectory output prefix differs") }
    }
    func finish() throws -> [H3TrajectoryTensor] {
        guard eventIndex == events.count, progress.outputs.count == 864,
              references == nil || progress.comparisons.count == 464 else { throw AetherError.invalidTensor("incomplete H3 trajectory") }
        return progress.outputs
    }
}

enum H3DenoisingCommand {
    static func run(_ args: [String]) async throws -> Bool {
        guard args.first == "qualify-h3-denoising" else { return false }
        guard args.count == 9, ["--prepare-only","--capture","--run","--lifecycle"].contains(args[8]), !["--run","--lifecycle"].contains(args[8]) || args[6] != "-" else {
            throw AetherError.invalidRequest("use qualify-h3-denoising BUNDLE PHASES CPU_RECORD CHAIN MODULATION_DIRECTORY REFERENCES|- NEW_OUTPUT_DIRECTORY --prepare-only|--capture|--run|--lifecycle")
        }
        let output=URL(fileURLWithPath:args[7],isDirectory:true).standardizedFileURL
        try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
        guard mkdir(output.path,0o755) == 0 else { throw AetherError.invalidRequest("H3 trajectory output must be new") }
        var report=TrajectoryReport(mode:args[8])
        if args[8] == "--lifecycle" {
            report.scope = "Provided-conditioning eight-step lifecycle: initial full run, submitted-body cancellation, retained full recovery, concurrent-unload cancellation, reopened full recovery. Same fixed numerical references and memory limits; not conditioning/decoders or a benchmark."
            report.nativeSubmissionBudget = 3340
            report.lifecycle = TrajectoryLifecycleEvidence()
        }
        var session:H3DenoisingSession?
        var sampler:BenchmarkMemorySampler?
        var sampling:Task<Void,Never>?
        func write() throws {
            if let snapshot=sampler?.snapshot { report.sampledPeak=snapshot.peak;report.memorySamples=snapshot.samples }
            let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
            try encoder.encode(report).write(to:output.appendingPathComponent("report.json"),options:.atomic)
        }
        func manifests(_ plan:H3DenoisingPlan) throws {
            for (i,hash) in [(1,plan.phases.header.manifestSHA256),(2,plan.phases.phaseManifestSHA256),(3,plan.phases.cpuRecordSHA256),(4,plan.backbone.manifestSHA256)] {
                guard try h3SHA256(Data(contentsOf:URL(fileURLWithPath:args[i]))) == hash else { throw AetherError.invalidModel("H3 trajectory manifest changed") }
            }
            if let hash=report.referenceSHA256 { guard try h3SHA256(Data(contentsOf:URL(fileURLWithPath:args[6]))) == hash else { throw AetherError.invalidModel("H3 trajectory reference manifest changed") } }
        }
        do {
            try write()
            let plan=try H3DenoisingPlan(bundle:URL(fileURLWithPath:args[1]),phases:URL(fileURLWithPath:args[2]),cpuRecord:URL(fileURLWithPath:args[3]),
                chain:URL(fileURLWithPath:args[4]),modulationDirectory:URL(fileURLWithPath:args[5]))
            report.bundleSHA256=plan.phases.header.manifestSHA256;report.phaseSHA256=plan.phases.phaseManifestSHA256
            report.cpuRecordSHA256=plan.phases.cpuRecordSHA256;report.chainSHA256=plan.backbone.manifestSHA256
            report.modulationModelSHA256=plan.modulations.map(\.sha256)
            let (specs,events,sources)=try boundaryPlan(plan)
            var references:TrajectoryReferences?
            let referenceURL = args[6] == "-" ? nil : URL(fileURLWithPath:args[6])
            if let referenceURL {
                let data=try Data(contentsOf:referenceURL);report.referenceSHA256=h3SHA256(data)
                let refs=try JSONDecoder().decode(TrajectoryReferences.self,from:data)
                try verify(refs,base:referenceURL,plan:plan,specs:specs,sources:sources)
                references=refs;report.readyForQualification=true
            }
            guard args[8] != "--capture" || references == nil else { throw AetherError.invalidRequest("H3 capture requires REFERENCES '-' to keep capture explicitly unqualified") }
            try write();try plan.verify()
            let verifiedMemory = try BenchmarkMemory.read(device:nil)
            report.resources["afterCPUVerification"]=verifiedMemory;try write()
            guard verifiedMemory.physicalFootprintBytes <= report.physicalFootprintLimitBytes,
                  verifiedMemory.processLifetimePeakFootprintBytes <= report.physicalFootprintLimitBytes else {
                throw AetherError.unsupported("H3 CPU verification exceeded the unchanged footprint bound before native acquisition")
            }
            report.selectedFilesVerified=true;try manifests(plan)
            if args[8] == "--prepare-only" {
                report.stage="ready";report.passed=true;try write();try AetherProbe.printJSON(report);return true
            }
            let memory=BenchmarkMemorySampler { try BenchmarkMemory.read(device:MTLCreateSystemDefaultDevice()) };sampler=memory
            sampling=Task { while !Task.isCancelled { memory.sample();do { try await Task.sleep(for:.milliseconds(20)) } catch { break } } }
            if args[8] == "--lifecycle" {
                let trace=H3DenoisingLifecycleTrace()
                let initial=try H3Latents(video:plan.inputVideo.read(),audio:plan.inputAudio.read())
                let conditioning=try plan.conditioning.read()
                func recorder(_ role:String) throws -> TrajectoryRecorder {
                    let directory=output.appendingPathComponent(role,isDirectory:true)
                    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
                    return TrajectoryRecorder(output:directory,references:references,referenceURL:referenceURL,sources:sources,specs:specs,events:events,sampler:memory)
                }
                func full(_ active:H3DenoisingSession,_ role:String) async throws {
                    report.stage=role;trace.reset();try write()
                    let record=try recorder(role)
                    _ = try await active.run(initial:initial,conditioning:conditioning) { try await record.record($0) }
                    _ = try await record.finish()
                    report.lifecycle?.traces[role]=trace.events;try write()
                    try H3DenoisingLifecycleTrace.validate(trace.events,cancelled:false,closing:false)
                    guard await active.state == .ready else { throw AetherError.invalidModel("trajectory recovery session not ready") }
                    report.lifecycle?.runs.append(role);try write()
                }
                func cancel(_ active:H3DenoisingSession,_ role:String,closing:Bool) async throws {
                    report.stage=role;try write()
                    let record=try recorder(role)
                    let start=AsyncStream<Void>.makeStream()
                    let unload=Mutex<Task<Void,Never>?>(nil)
                    let task=Task {
                        for await _ in start.stream { break }
                        do {
                            _ = try await active.run(initial:initial,conditioning:conditioning) { try await record.record($0) }
                            trace.append("runSucceeded")
                        } catch is CancellationError { trace.append("runCancelled") }
                        catch { trace.append("runFailed");throw error }
                    }
                    trace.reset {
                        task.cancel()
                        if closing { unload.withLock { $0=Task { await active.unload() } } }
                    }
                    start.continuation.yield();start.continuation.finish()
                    let result=await task.result
                    await unload.withLock { $0 }?.value
                    report.lifecycle?.traces[role]=trace.events;try write()
                    try result.get()
                    try H3DenoisingLifecycleTrace.validate(trace.events,cancelled:true,closing:closing)
                    try await record.cancelledPrefix()
                    guard await active.state == (closing ? .closed : .ready) else { throw AetherError.invalidModel("trajectory cancellation state differs") }
                    if closing {
                        do { _ = try await active.run(initial:initial,conditioning:conditioning);throw AetherError.invalidModel("closed trajectory accepted work") }
                        catch AetherError.sessionClosed { }
                    }
                    report.lifecycle?.runs.append(role);try write()
                }
                func settle(_ role:String) async throws -> BenchmarkMemory {
                    var final:BenchmarkMemory?
                    for seconds in [0,1,5] {
                        if seconds != 0 { try await Task.sleep(for:.seconds(seconds == 1 ? 1 : 4)) }
                        let value=try BenchmarkMemory.read(device:MTLCreateSystemDefaultDevice())
                        report.resources[role+".afterUnload\(seconds)s"]=value;try write()
                        try H3LifecycleLimits().check(value);final=value
                    }
                    return final!
                }
                report.nativeExecuted=true;report.stage="lifecycle.open";try write()
                // Limit lexical ownership to release both the closed session and
                // backend before measuring settlement or opening its replacement.
                do {
                    let active=try await H3DenoisingSession.open(plan:plan,executionObserver:{ trace.record($0) });session=active
                    try await full(active,"initial")
                    try await cancel(active,"retainedCancellation",closing:false)
                    try await full(active,"retainedRecovery")
                    try await cancel(active,"closingCancellation",closing:true)
                    session=nil
                }
                let baseline=try await settle("firstSession")
                do {
                    let active=try await H3DenoisingSession.open(plan:plan,executionObserver:{ trace.record($0) });session=active
                    try await full(active,"reopenedRecovery")
                    await active.unload();session=nil
                }
                let settled=try await settle("reopenedSession")
                try H3LifecycleLimits().checkGrowth(baseline:baseline,current:settled)
            } else {
                let recorder=TrajectoryRecorder(output:output,references:references,referenceURL:referenceURL,sources:sources,specs:specs,events:events,sampler:memory)
                report.nativeExecuted=true;report.stage="open";try write()
                let active=try await H3DenoisingSession.open(plan:plan);session=active
                let initial=try H3Latents(video:plan.inputVideo.read(),audio:plan.inputAudio.read())
                report.stage="trajectory";try write()
                _ = try await active.run(initial:initial,conditioning:plan.conditioning.read()) { try await recorder.record($0) }
                report.outputs=try await recorder.finish()
                await active.unload();session=nil
                report.stage="settle";try write()
                for seconds in [0,1,5] {
                    if seconds != 0 { try await Task.sleep(for:.seconds(seconds == 1 ? 1 : 4)) }
                    let value=try BenchmarkMemory.read(device:MTLCreateSystemDefaultDevice())
                    report.resources["afterUnload\(seconds)s"]=value;try write();try H3LifecycleLimits().check(value)
                }
            }
            try manifests(plan);sampling?.cancel();await sampling?.value
            if let error=memory.snapshot.error { throw AetherError.unsupported(error) }
            if let peak=memory.snapshot.peak { try H3LifecycleLimits().check(peak) }
            report.captureComplete=args[8] == "--capture";report.passed=["--run","--lifecycle"].contains(args[8])
            report.stage=report.captureComplete ? "capturedAwaitingCPUReferences" : "complete"
            try write();try AetherProbe.printJSON(report)
        } catch {
            if let session { await session.unload() }
            sampling?.cancel();await sampling?.value
            report.stage="failed";report.passed=false;report.error=String(describing:error);try write();throw error
        }
        return true
    }
    private static func boundaryPlan(_ plan:H3DenoisingPlan) throws -> ([String:H3TensorSpec],[String],[String:String]) {
        var specs:[String:H3TensorSpec]=[:],events:[String]=[]
        var sources=["conditioning":plan.conditioning.sha256,"initial.video":plan.inputVideo.sha256,"initial.audio":plan.inputAudio.sha256]
        for file in plan.backbone.cases[0].base.inputs where ["cos","sin"].contains(file.name) { sources["rotary."+file.name]=file.sha256 }
        func add(_ stage:String,_ step:Int,_ block:Int? = nil,_ tensors:[String:H3TensorSpec]) {
            events.append("\(stage).\(step).\(block ?? -1)")
            for (name,spec) in tensors { specs[trajectoryKey(stage,step,block,name)]=spec }
        }
        for step in 0..<8 {
            let rows=plan.layout.timesteps(for:plan.schedule.steps[step])
            sources["timestep.\(step)"]=rows.values.withUnsafeBytes { h3SHA256(Data($0)) }
            sources["indices.\(step)"]=rows.indices.withUnsafeBytes { h3SHA256(Data($0)) }
            sources["tags.\(step)"]=rows.adalnIndices.withUnsafeBytes { h3SHA256(Data($0)) }
            add("time",step,nil,["temb":try H3TensorSpec(scalarType:.float32,shape:[rows.values.count,2688])])
        }
        for block in 0..<50 { for step in 0..<8 {
            let n=step == 0 ? 1 : 2
            add("modulation",step,block,["modulations":try H3TensorSpec(scalarType:.bfloat16,shape:[6,n*3,5376])])
        } }
        let hidden=plan.backbone.cases[0].base.output, latents=["video":plan.inputVideo.spec,"audio":plan.inputAudio.spec]
        for step in 0..<8 {
            add("input",step,nil,latents);add("prepare",step,nil,["hidden":hidden])
            for block in 0..<50 { add("body",step,block,["hidden":hidden]) }
            add("heads",step,nil,latents);add("latents",step,nil,latents)
        }
        return (specs,events,sources)
    }
    private static func context(_ key:String, expected:[String:String], sources:[String:String]) throws -> [String:String] {
        let parts=key.split(separator:".");guard parts.count == 4,let step=Int(parts[1]),let block=Int(parts[2]) else { throw AetherError.invalidModel("invalid H3 trajectory key") }
        let keys:[String]
        switch parts[0] {
        case "time":keys=["timestep.\(step)"]
        case "modulation":keys=[trajectoryKey("time",step,nil,"temb")]
        case "prepare":keys=[trajectoryKey("input",step,nil,"video"),trajectoryKey("input",step,nil,"audio"),"conditioning"]
        case "body":keys=[trajectoryKey(block == 0 ? "prepare" : "body",step,block == 0 ? nil : block-1,"hidden"),
            trajectoryKey("time",step,nil,"temb"),trajectoryKey("modulation",step,block,"modulations"),"tags.\(step)","rotary.cos","rotary.sin"]
        case "heads":keys=[trajectoryKey("body",step,49,"hidden"),trajectoryKey("time",step,nil,"temb"),"indices.\(step)"]
        case "latents":keys=[trajectoryKey("input",step,nil,"video"),trajectoryKey("input",step,nil,"audio"),trajectoryKey("heads",step,nil,"video"),trajectoryKey("heads",step,nil,"audio")]
        default:throw AetherError.invalidModel("unsupported H3 CPU trajectory reference")
        }
        var result:[String:String]=[:]
        for key in keys { guard let hash=expected[key] ?? sources[key] else { throw AetherError.invalidModel("missing H3 CPU context identity") };result[key]=hash }
        return result
    }
    private static func verify(_ refs:TrajectoryReferences,base:URL,plan:H3DenoisingPlan,specs:[String:H3TensorSpec],sources:[String:String]) throws {
        let required=Set(specs.keys.filter { key in
            !key.hasPrefix("input.") && (!key.hasPrefix("body.") || key.split(separator:".")[2] == "0" || key.split(separator:".")[2] == "49")
        })
        guard refs.schema == 1,refs.bundle_sha256 == plan.phases.header.manifestSHA256,
              refs.phase_sha256 == plan.phases.phaseManifestSHA256,refs.chain_sha256 == plan.backbone.manifestSHA256,
              refs.cpu_record_sha256 == plan.phases.cpuRecordSHA256,refs.modulation_model_sha256 == plan.modulations.map(\.sha256),
              isDigest(refs.capture_report_sha256),isDigest(refs.oracle_recipe_sha256),
              Set(refs.expected.keys) == Set(specs.keys),refs.expected.values.allSatisfy(isDigest),
              refs.references.count == required.count,Set(refs.references.map(\.key)) == required else {
            throw AetherError.invalidModel("H3 trajectory references lack the exact corrected scope or complete identities")
        }
        guard !refs.capture_report_path.contains("://") else { throw AetherError.invalidModel("capture report must be local") }
        let captureData=try Data(contentsOf:URL(fileURLWithPath:refs.capture_report_path,relativeTo:base.deletingLastPathComponent()))
        guard h3SHA256(captureData) == refs.capture_report_sha256 else { throw AetherError.invalidModel("H3 capture report changed") }
        let capture=try JSONDecoder().decode(TrajectoryReport.self,from:captureData)
        guard capture.captureComplete,!capture.passed,capture.mode == "--capture",capture.outputs.count == 864,
              Set(capture.outputs.map(\.key)).count == capture.outputs.count,
              capture.bundleSHA256 == refs.bundle_sha256,capture.phaseSHA256 == refs.phase_sha256,
              capture.chainSHA256 == refs.chain_sha256,capture.cpuRecordSHA256 == refs.cpu_record_sha256,
              capture.modulationModelSHA256 == refs.modulation_model_sha256,
              Dictionary(uniqueKeysWithValues:capture.outputs.map { ($0.key,$0.sha256) }) == refs.expected else {
            throw AetherError.invalidModel("H3 references differ from their unqualified capture")
        }
        for reference in refs.references {
            guard let spec=specs[reference.key],reference.context == (try context(reference.key,expected:refs.expected,sources:sources)) else {
                throw AetherError.invalidModel("H3 numerical reference contexts differ")
            }
            let file=try reference.tensor.file(base:base)
            if reference.key.hasPrefix("body.") {
                let rows=plan.backbone.cases[0].base.referenceRows
                guard reference.tensor.row_indices == rows,file.spec.scalarType == spec.scalarType,
                      file.spec.shape == [1,rows.count,spec.shape[2]] else { throw AetherError.invalidModel("H3 trajectory body reference rows differ") }
            } else { guard reference.tensor.row_indices == nil,file.spec == spec else { throw AetherError.invalidModel("H3 trajectory requires complete outer/modulation/update references") } }
            _ = try file.read()
        }
    }
}
