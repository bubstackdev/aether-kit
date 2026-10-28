import AetherKit
import AetherH3
import AetherH3CoreAI
import AetherH3Tokenizers
import Darwin
import Foundation
import ImageIO
import Metal

/// Raw tensor directory: `manifest.json` names every little-endian tensor file with its dtype, shape and SHA-256, so
/// the CPU oracle (`Scripts/prepare-h3-profile-oracle.py`) and the comparison script read exactly these bytes.
@available(macOS 27, iOS 27, *)
struct H3RawEntry: Codable { let file: String; let dtype: String; let shape: [Int]; let sha256: String }
@available(macOS 27, iOS 27, *)
struct H3RawManifest<E: Encodable>: Encodable { let tensors: [String: H3RawEntry]; let fields: [String: String]; let report: E }
@available(macOS 27, iOS 27, *)
final class H3RawDirectory {
    let url: URL
    var tensors: [String: H3RawEntry] = [:]
    var fields: [String: String] = [:]
    init(new url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(url.path, 0o755) == 0 else { throw AetherError.invalidRequest("H3 probe output must be a new directory: \(url.path)") }
        self.url = url
    }
    func put(_ name: String, _ tensor: H3TensorData) throws {
        try tensor.bytes.write(to: url.appendingPathComponent(name + ".raw"))
        tensors[name] = H3RawEntry(file: name + ".raw", dtype: tensor.spec.scalarType.rawValue, shape: tensor.spec.shape, sha256: tensor.sha256)
        try save(["partial": "true"])   // a native abort still leaves every finished tensor readable
    }
    func putBytes(_ name: String, _ data: Data, dtype: String, shape: [Int]) throws {
        try data.write(to: url.appendingPathComponent(name + ".raw"))
        tensors[name] = H3RawEntry(file: name + ".raw", dtype: dtype, shape: shape, sha256: h3SHA256(data))
        try save(["partial": "true"])
    }
    func save<E: Encodable>(_ extra: E) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(H3RawManifest(tensors: tensors, fields: fields, report: extra)).write(to: url.appendingPathComponent("manifest.json"), options: .atomic)
    }
    static func read(_ url: URL, _ name: String) throws -> H3TensorData {
        struct Manifest: Decodable { let tensors: [String: H3RawEntry] }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url.appendingPathComponent("manifest.json")))
        guard let entry = manifest.tensors[name], let type = H3ScalarType(rawValue: entry.dtype) else { throw AetherError.invalidRequest("missing tensor \(name) in \(url.path)") }
        let data = try Data(contentsOf: url.appendingPathComponent(entry.file))
        guard h3SHA256(data) == entry.sha256 else { throw AetherError.invalidModel("tensor \(name) changed on disk") }
        return try H3TensorData(spec: H3TensorSpec(scalarType: type, shape: entry.shape), bytes: data)
    }
    static func field(_ url: URL, _ name: String) throws -> String {
        struct Manifest: Decodable { let fields: [String: String] }
        guard let value = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url.appendingPathComponent("manifest.json"))).fields[name] else {
            throw AetherError.invalidRequest("missing field \(name) in \(url.path)")
        }
        return value
    }
}

@available(macOS 27, iOS 27, *)
private func ints(_ values: [Int32], _ shape: [Int]) throws -> H3TensorData {
    try H3TensorData(spec: H3TensorSpec(scalarType: .int32, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
}
@available(macOS 27, iOS 27, *)
private func floats(_ values: [Float], _ shape: [Int]) throws -> H3TensorData {
    try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
}
/// Round-to-nearest-even float32 -> bf16 (finite inputs).
@available(macOS 27, iOS 27, *)
private func bf16(_ values: [Float], _ shape: [Int]) throws -> H3TensorData {
    let bits = values.map { value -> UInt16 in
        let b = value.bitPattern
        return UInt16(truncatingIfNeeded: (b &+ 0x7fff &+ ((b >> 16) & 1)) >> 16)
    }
    return try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: shape), bytes: bits.withUnsafeBytes { Data($0) })
}

@available(macOS 27, iOS 27, *)
private struct H3Timing: Codable { let name: String; let seconds: Double; var memory: BenchmarkMemory? }

/// Records what each stage returns, so a render can be compared stage by stage against the CPU oracle.
@available(macOS 27, iOS 27, *)
private actor H3RenderTape {
    var tap: H3TensorData?, latents: H3Latents?, frames = Data(), pcm: H3AudioPCM?
    func set(tap: H3TensorData) { self.tap = tap }
    func set(latents: H3Latents) { self.latents = latents }
    func append(_ batch: H3DecodedFrames) { for frame in batch.rgb { frames.append(contentsOf: frame) } }
    func set(pcm: H3AudioPCM) { self.pcm = pcm }
}
@available(macOS 27, iOS 27, *)
private struct TapeConditioner: H3PromptConditioner {
    let inner: any H3PromptConditioner, tape: H3RenderTape
    func encode(_ prompt: String, geometry: H3Geometry, onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3TensorData {
        let tap = try await inner.encode(prompt, geometry: geometry, onLayer: onLayer); await tape.set(tap: tap); return tap
    }
    func release() async { await inner.release() }
}
@available(macOS 27, iOS 27, *)
private struct TapeDenoiser: H3Denoiser {
    let inner: any H3Denoiser, tape: H3RenderTape
    var steps: Int { inner.steps }
    func denoise(_ initial: H3Latents, conditioning: H3TensorData, geometry: H3Geometry, onStep: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Latents {
        let latents = try await inner.denoise(initial, conditioning: conditioning, geometry: geometry, onStep: onStep)
        await tape.set(latents: latents); return latents
    }
    func release() async { await inner.release() }
}

@available(macOS 27, iOS 27, *)
enum H3ProfileCommand {
    static func run(_ arguments: [String]) async throws -> Bool {
        switch arguments.first {
        case "h3-profile-dump":
            guard arguments.count == 9 || arguments.count == 10, let width = Int(arguments[3]), let height = Int(arguments[4]),
                  let frames = Int(arguments[5]), let seed = UInt64(arguments[6]) else {
                throw AetherError.invalidRequest("use h3-profile-dump PROFILE.json PROMPT.txt WIDTH HEIGHT FRAMES SEED RECIPE NEW_DIRECTORY [FIRST_FRAME.png]")
            }
            try await dump(profile: URL(fileURLWithPath: arguments[1]), prompt: URL(fileURLWithPath: arguments[2]), width: width, height: height,
                           frames: frames, seed: seed, recipe: arguments[7], output: URL(fileURLWithPath: arguments[8]),
                           firstFrame: arguments.count == 10 ? URL(fileURLWithPath: arguments[9]) : nil)
            return true
        case "h3-profile-run":
            guard arguments.count >= 6, ["--graphs", "--render"].contains(arguments[5]) else {
                throw AetherError.invalidRequest("use h3-profile-run PROFILE.json DUMP REFERENCES|- NEW_DIRECTORY --graphs|--render [--no-audio] [--capture=all|S,S,...|--replay=CAPTURE_RUN]")
            }
            // --render only: --capture saves the trajectory boundaries of the listed steps and hashes every boundary;
            // --replay hashes every boundary and requires each to equal that capture's.
            let capture = arguments.first { $0.hasPrefix("--capture=") }.map { String($0.dropFirst("--capture=".count)) }
            let replay = arguments.first { $0.hasPrefix("--replay=") }.map { URL(fileURLWithPath: String($0.dropFirst("--replay=".count))) }
            guard arguments[5] == "--render" || (capture == nil && replay == nil), capture == nil || replay == nil else {
                throw AetherError.invalidRequest("--capture and --replay apply to --render, one at a time")
            }
            try await execute(profile: URL(fileURLWithPath: arguments[1]), dump: URL(fileURLWithPath: arguments[2]),
                              references: arguments[3] == "-" ? nil : URL(fileURLWithPath: arguments[3]), output: URL(fileURLWithPath: arguments[4]),
                              render: arguments[5] == "--render", audio: !arguments.contains("--no-audio"), capture: capture, replay: replay)
            return true
        case "h3-weights-bench":
            // CPU only: how long one denoising step's per-block weight reads take (what NativeH3Graph does per block call).
            guard arguments.count == 3, let passes = Int(arguments[2]) else { throw AetherError.invalidRequest("use h3-weights-bench PROFILE.json PASSES") }
            let profile = try H3Profile(url: URL(fileURLWithPath: arguments[1]))
            let largest = profile.parameters.flatMap { $0 }.map(\.spec.byteCount).max() ?? 0
            let scratch = UnsafeMutableRawBufferPointer.allocate(byteCount: largest, alignment: 16384)
            defer { scratch.deallocate() }
            let clock = ContinuousClock()
            for pass in 0..<passes {
                var bytes = 0
                let start = clock.now
                for block in profile.parameters { for file in block {
                    try file.read(into: UnsafeMutableRawBufferPointer(rebasing: scratch[..<file.spec.byteCount]), verify: false); bytes += file.spec.byteCount
                } }
                let d = clock.now - start, seconds = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
                print("pass \(pass): \(bytes / 1_000_000) MB in \(String(format: "%.2f", seconds)) s (\(String(format: "%.2f", Double(bytes) / seconds / 1e9)) GB/s)")
            }
            return true
        case "h3-profile-decode":
            guard arguments.count == 5 || arguments.count == 7 else {
                throw AetherError.invalidRequest("use h3-profile-decode PROFILE.json DUMP RENDER NEW_DIRECTORY [DECODER.aimodel SHA256]")
            }
            let decoder = arguments.count == 7 ? try H3ModelAsset(url: URL(fileURLWithPath: arguments[5]), function: "main", sha256: arguments[6]) : nil
            try await decode(profile: URL(fileURLWithPath: arguments[1]), dump: URL(fileURLWithPath: arguments[2]),
                             render: URL(fileURLWithPath: arguments[3]), output: URL(fileURLWithPath: arguments[4]), decoder: decoder)
            return true
        default:
            return false
        }
    }

    /// CPU only: the tokens, layout, step plan and seeded inputs a run and the CPU oracle share.
    /// An image file at exactly `width`×`height` as interleaved RGB8 (drawn into an sRGB RGBA8 context, alpha dropped).
    static func keyframe(_ url: URL, width: Int, height: Int) throws -> H3Keyframe {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              image.width == width, image.height == height,
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw AetherError.invalidRequest("first frame must be an image of exactly \(width)x\(height)")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        let rgba = UnsafeBufferPointer(start: context.data!.assumingMemoryBound(to: UInt8.self), count: width * height * 4)
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for p in 0..<(width * height) { for c in 0..<3 { rgb[p * 3 + c] = rgba[p * 4 + c] } }
        return try H3Keyframe(width: width, height: height, rgb: rgb)
    }

    private static func dump(profile url: URL, prompt: URL, width: Int, height: Int, frames: Int, seed: UInt64, recipe: String, output: URL,
                             firstFrame: URL? = nil) async throws {
        let profile = try H3Profile(url: url)
        let text = try String(contentsOf: prompt, encoding: .utf8)
        let tokenizer = try await H3Tokenizer.open(directory: profile.tokenizer)
        let ids = try tokenizer.tokenIDs(text)
        let geometry = try H3Geometry(width: width, height: height, requestedFrames: frames, diagnostic: true)
        let keyframe = try firstFrame.map { try keyframe($0, width: width, height: height) }
        let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: ids.count), anchors: keyframe == nil ? [] : [.first])
        guard let schedule = profile.recipes[recipe]?.schedule else { throw AetherError.unsupported("profile has no \(recipe)") }
        let out = try H3RawDirectory(new: output)
        out.fields = ["profile_sha256": profile.manifestSHA256, "recipe": recipe, "seed": String(seed), "width": String(width),
                      "height": String(height), "frames": String(geometry.frames), "tokens": String(ids.count), "rows": String(layout.rows),
                      "steps": String(schedule.count), "prompt_sha256": h3SHA256(Data(text.utf8))]
        try out.put("ids", ints(ids, [ids.count]))
        try out.put("positions", floats(layout.positions.values, [layout.rows, 3]))
        try out.put("token_tags", ints(layout.tokenTags, [layout.rows]))
        try out.put("text_indices", ints(layout.textIndices, [layout.textIndices.count]))
        try out.put("video_indices", ints(layout.videoIndices, [layout.videoIndices.count]))
        try out.put("audio_indices", ints(layout.audioIndices, [layout.audioIndices.count]))
        try out.put("gather", ints(try H3Packing.gatherIndices(layout), [layout.rows]))
        let rotary = try layout.rotary()
        try out.put("cos", floats(rotary.cos.values, [layout.rows, 96])); try out.put("sin", floats(rotary.sin.values, [layout.rows, 96]))
        for step in schedule.steps {
            let rows = layout.timesteps(for: step)
            try out.put("step\(step.index)_timesteps", floats(rows.values, [rows.values.count]))
            try out.put("step\(step.index)_indices", ints(rows.indices, [layout.rows]))
            try out.put("step\(step.index)_adaln", ints(rows.adalnIndices, [layout.rows]))
        }
        if let keyframe {
            // The FL2VA stream: the keyframe's condition noise first, then the generated video rows and audio (the session's order).
            out.fields["anchors"] = "first"
            let frame = 24 * geometry.latentHeight * geometry.latentWidth, video = geometry.targetVideoRows * 96
            let noise = try H3Noise.generate(seed: seed, count: frame + video + geometry.audioRows * 32)
            try out.putBytes("keyframe", Data(keyframe.rgb), dtype: "uint8", shape: [height, width, 3])
            try out.put("condition_noise", floats(Array(noise[..<frame]), [1, 24, 1, geometry.latentHeight, geometry.latentWidth]))
            let target = try geometry.packVideo(FloatTensor(shape: geometry.videoLatentShape, values: Array(noise[frame..<(frame + video)])))
            try out.put("latents_video", floats(target.values, [1, geometry.targetVideoRows, 96]))
            try out.put("latents_audio", floats(Array(noise[(frame + video)...]), [1, geometry.audioRows, 32]))
        } else {
            let initial = try geometry.initialLatents(seed: seed)
            try out.put("latents_video", initial.video); try out.put("latents_audio", initial.audio)
        }
        // A seeded unit-variance hidden state for isolated block/heads checks (values round to bf16 once).
        let probe = try H3Noise.generate(seed: seed ^ 0x5eed, count: layout.rows * 5376)
        try out.put("probe_hidden", bf16(probe, [1, layout.rows, 5376]))
        try out.save(["scope": "inputs only; no native execution"])
        print(output.path)
    }

    private static func execute(profile url: URL, dump: URL, references: URL?, output: URL, render: Bool, audio: Bool,
                                capture: String? = nil, replay: URL? = nil) async throws {
        let profile = try H3Profile(url: url)
        guard try H3RawDirectory.field(dump, "profile_sha256") == profile.manifestSHA256 else { throw AetherError.invalidModel("dump was made for another profile") }
        let geometry = try H3Geometry(width: Int(H3RawDirectory.field(dump, "width"))!, height: Int(H3RawDirectory.field(dump, "height"))!,
                                      requestedFrames: Int(H3RawDirectory.field(dump, "frames"))!, diagnostic: true)
        let recipe = try H3RawDirectory.field(dump, "recipe"), seed = UInt64(try H3RawDirectory.field(dump, "seed"))!
        let out = try H3RawDirectory(new: output)
        out.fields = ["profile_sha256": profile.manifestSHA256, "mode": render ? "render" : "graphs", "dump": dump.path]
        let device = MTLCreateSystemDefaultDevice()
        let sampler = BenchmarkMemorySampler { try BenchmarkMemory.read(device: device) }
        let sampling = Task.detached { while !Task.isCancelled { sampler.sample(); try? await Task.sleep(for: .milliseconds(50)) } }
        defer { sampling.cancel() }
        var timings: [H3Timing] = []
        let clock = ContinuousClock()
        func timed<T>(_ name: String, _ body: () async throws -> T) async throws -> T {
            let start = clock.now
            let value = try await body()
            let elapsed = clock.now - start
            timings.append(H3Timing(name: name, seconds: Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18,
                                    memory: try? BenchmarkMemory.read(device: device)))
            FileHandle.standardError.write(Data("h3: \(name) \(timings.last!.seconds) s\n".utf8))
            return value
        }
        struct Report: Encodable { let timings: [H3Timing]; let sampledPeak: BenchmarkMemory?; let samples: Int; let error: String? }
        func save(_ error: String? = nil) throws {
            let state = sampler.snapshot
            try out.save(Report(timings: timings, sampledPeak: state.peak, samples: state.samples, error: error))
        }
        do {
            if render {
                let ids = try H3RawDirectory.read(dump, "ids")
                let tokens = ids.bytes.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
                let tape = H3RenderTape()
                var recorder: H3TrajectoryRecorder? = nil
                var observer: H3DenoiserObserver? = nil
                if capture != nil || replay != nil {
                    let made = try H3TrajectoryRecorder(directory: output.appendingPathComponent("capture"), steps: capture, reference: replay)
                    recorder = made
                    observer = { @Sendable (event: H3DenoiserObservation) async throws -> Void in try await made.record(event) }
                }
                let stages = try H3ProfileEngine.stages(profile: profile, recipe: recipe, tokenize: { _ in tokens }, observer: observer)
                let session = H3VideoSession(stages: H3VideoStages(conditioner: TapeConditioner(inner: stages.conditioner, tape: tape),
                                                                   keyframes: stages.keyframes,
                                                                   denoiser: TapeDenoiser(inner: stages.denoiser, tape: tape),
                                                                   video: stages.video, audio: stages.audio))
                let stageClock = H3StageClock()
                let first = try firstFrame(dump, geometry: geometry)
                let request = try H3VideoRequest(prompt: "ids", geometry: geometry, seed: seed, firstFrame: first, audio: audio)
                let result = try await timed("render") {
                    try await session.generate(request) { event in
                        await stageClock.record(event, memory: try? BenchmarkMemory.read(device: device))
                        switch event {
                        case .frames(let batch): await tape.append(batch)
                        case .audio(let pcm): await tape.set(pcm: pcm)
                        default: break
                        }
                    }
                }
                await session.unload()
                timings += await stageClock.timings
                if let tap = await tape.tap { try out.put("tap", tap) }
                if let latents = await tape.latents { try out.put("final_video", latents.video); try out.put("final_audio", latents.audio) }
                try out.putBytes("frames", await tape.frames, dtype: "uint8", shape: [result.frames, geometry.height, geometry.width, 3])
                if let pcm = await tape.pcm { try out.put("pcm", floats(pcm.channels[0] + pcm.channels[1], [2, pcm.sampleCount])) }
                if let recorder {
                    let summary = try await recorder.finish()
                    out.fields["trajectory_hashes"] = String(summary.hashes)
                    if let mismatches = summary.mismatches {
                        out.fields["replay_mismatches"] = String(mismatches.count)
                        guard mismatches.isEmpty else {
                            throw AetherError.invalidModel("replay differs from the capture at \(mismatches.count) boundaries, first \(mismatches.prefix(5))")
                        }
                    }
                }
            } else {
                let qualifier = try H3ProfileQualifier(profile: profile)
                if let first = try firstFrame(dump, geometry: geometry) {
                    // An FL2VA dump qualifies the keyframe encoder only; the other graphs are the T2VA ones, qualified on T2VA dumps.
                    for pass in ["cold", "warm"] {
                        let condition = try await timed("keyframe.\(pass)") { try await qualifier.keyframe(first, geometry: geometry) }
                        if pass == "warm" { try out.put("condition", condition) }
                    }
                    await qualifier.release()
                    try save(); print(output.path); return
                }
                let ids = try H3RawDirectory.read(dump, "ids")
                let tokens = ids.bytes.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
                let nativeTap = try await timed("conditioner.tap") { try await qualifier.tap(ids: tokens) }
                try out.put("tap", nativeTap)
                // Every later graph runs on the CPU oracle's own inputs, so each comparison isolates one graph.
                let referenceTap = try references.map { try H3RawDirectory.read($0, "tap") } ?? nativeTap
                let refined = try await timed("prepare.text") { try await qualifier.refine(referenceTap) }
                try out.put("refined", refined)
                let referenceRefined = try references.map { try H3RawDirectory.read($0, "refined") } ?? refined
                let packed = try await timed("prepare.pack") {
                    try await qualifier.pack(video: try H3RawDirectory.read(dump, "latents_video"), audio: try H3RawDirectory.read(dump, "latents_audio"),
                                             text: referenceRefined, gather: try H3RawDirectory.read(dump, "gather"))
                }
                try out.put("packed", packed)
                let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: tokens.count))
                let step0 = try profile.steps(recipe: recipe, layout: layout)[0]
                let tags = try H3RawDirectory.read(dump, "step0_adaln")
                let cos = try H3RawDirectory.read(dump, "cos"), sin = try H3RawDirectory.read(dump, "sin")
                // Real step-0 contexts: the packed sequence enters block 0, and the native chain 0...48 supplies block 49's input.
                // Each checked block's input (`contextNN`, the predecessor's native output) and output are saved, so the CPU
                // oracle evaluates that one block on exactly these bytes. Heads then read block 49's output.
                let checked: Set<Int> = [0, 49]
                var hidden = packed
                for block in 0..<50 {
                    if checked.contains(block) && block > 0 { try out.put(String(format: "context%02d", block), hidden) }
                    let table = try step0.tables[block].read(), input = hidden
                    hidden = try await timed("block\(block)") { try await qualifier.block(block, hidden: input, table: table, tags: tags, cos: cos, sin: sin) }
                    if checked.contains(block) { try out.put(String(format: "block%02d", block), hidden) }
                }
                let final = hidden
                let heads = try await timed("heads") {
                    try await qualifier.heads(hidden: final, temb: try step0.embedding.read(), indices: try H3RawDirectory.read(dump, "step0_indices"),
                                              videoIndices: try H3RawDirectory.read(dump, "video_indices"), audioIndices: try H3RawDirectory.read(dump, "audio_indices"),
                                              video: try H3RawDirectory.read(dump, "latents_video").spec, audio: try H3RawDirectory.read(dump, "latents_audio").spec)
                }
                try out.put("heads_video", heads.video); try out.put("heads_audio", heads.audio)
                let packedVideo = try H3RawDirectory.read(dump, "latents_video").floatValues()
                let tiling = try H3SpatialTiling.source(width: geometry.width, height: geometry.height)
                let rows = try tiling.extract(width: geometry.width, height: geometry.height, packed: try geometry.videoWindowRows(packedVideo, index: 0), index: 0)
                for pass in ["cold", "warm"] {
                    let tile = try await timed("tile.\(pass)") { try await qualifier.tile(try floats(rows, [448, 96])) }
                    if pass == "warm" { try out.put("tile", tile); try out.put("tile_rows", try floats(rows, [448, 96])) }
                }
                if audio {
                    let audioRows = try H3RawDirectory.read(dump, "latents_audio")
                    let na = geometry.audioLatentsPerChannel
                    for pass in ["cold", "warm"] {
                        let pcm = try await timed("audio.\(pass)") {
                            try await qualifier.audio(try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [2, na, 32]), bytes: audioRows.bytes))
                        }
                        if pass == "warm" { try out.put("pcm", pcm) }
                    }
                }
                await qualifier.release()
            }
            try save()
            print(output.path)
        } catch {
            try? save(String(describing: error)); throw error
        }
    }
}

@available(macOS 27, iOS 27, *)
extension H3ProfileCommand {
    /// Video decode only, from a render's final latents, with the GPU tile calls and the CPU work timed apart.
    static func decode(profile url: URL, dump: URL, render: URL, output: URL, decoder: H3ModelAsset?) async throws {
        let profile = try H3Profile(url: url)
        let geometry = try H3Geometry(width: Int(H3RawDirectory.field(dump, "width"))!, height: Int(H3RawDirectory.field(dump, "height"))!,
                                      requestedFrames: Int(H3RawDirectory.field(dump, "frames"))!, diagnostic: true)
        let out = try H3RawDirectory(new: output)
        out.fields = ["profile_sha256": profile.manifestSHA256, "mode": "decode", "render": render.path,
                      "decoder": decoder?.url.path ?? profile.videoDecoder.model.url.path]
        let device = MTLCreateSystemDefaultDevice()
        let sampler = BenchmarkMemorySampler { try BenchmarkMemory.read(device: device) }
        let sampling = Task.detached { while !Task.isCancelled { sampler.sample(); try? await Task.sleep(for: .milliseconds(50)) } }
        defer { sampling.cancel() }
        let clock = ContinuousClock()
        func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
        var gpu: [Double] = [], extract = 0.0, stitch = 0.0, rgb = 0.0
        let qualifier = try H3ProfileQualifier(profile: profile, videoDecoder: decoder)
        let packed = try H3RawDirectory.read(render, "final_video").floatValues()
        let tiling = try H3SpatialTiling.source(width: geometry.width, height: geometry.height)
        var video = try H3VideoStitcher(geometry: geometry)
        var frames = Data()
        let started = clock.now
        func deliver(_ normalized: [Float]) throws {
            let t = clock.now
            for frame in try H3Pixels.rgb8(frames: normalized, width: geometry.width, height: geometry.height) { frames.append(contentsOf: frame) }
            rgb += seconds(clock.now - t)
        }
        for window in 0..<geometry.videoWindows {
            let rows = try geometry.videoWindowRows(packed, index: window)
            var spatial = try H3SpatialStitcher(tiling: tiling, width: geometry.width, height: geometry.height)
            for tile in 0..<tiling.tileCount {
                var t = clock.now
                let input = try tiling.extract(width: geometry.width, height: geometry.height, packed: rows, index: tile)
                extract += seconds(clock.now - t); t = clock.now
                let pixels = try await qualifier.tile(try floats(input, [448, 96]))
                gpu.append(seconds(clock.now - t)); t = clock.now
                try spatial.push(try pixels.floatValues())
                stitch += seconds(clock.now - t)
            }
            let t = clock.now
            let window = try video.push(try spatial.finish())
            stitch += seconds(clock.now - t)
            try deliver(window)
        }
        try deliver(video.finish())
        let total = seconds(clock.now - started)
        await qualifier.release()
        try out.putBytes("frames", frames, dtype: "uint8", shape: [geometry.frames, geometry.height, geometry.width, 3])
        struct Report: Encodable {
            let total: Double; let tiles: Int; let tileFirst: Double; let tileWarmMean: Double; let gpu: Double
            let extract: Double; let stitch: Double; let rgb: Double; let sampledPeak: BenchmarkMemory?
        }
        let warm = gpu.dropFirst()
        try out.save(Report(total: total, tiles: gpu.count, tileFirst: gpu.first ?? 0, tileWarmMean: warm.reduce(0, +) / Double(max(warm.count, 1)),
                            gpu: gpu.reduce(0, +), extract: extract, stitch: stitch, rgb: rgb, sampledPeak: sampler.snapshot.peak))
        print(output.path)
    }
}

@available(macOS 27, iOS 27, *)
extension H3ProfileCommand {
    /// The dump's keyframe (FL2VA dumps), if any.
    static func dumpKeyframe(_ dump: URL, geometry: H3Geometry) throws -> H3Keyframe? { try firstFrame(dump, geometry: geometry) }
}

/// The dump's keyframe (FL2VA dumps), if any.
@available(macOS 27, iOS 27, *)
private func firstFrame(_ dump: URL, geometry: H3Geometry) throws -> H3Keyframe? {
    struct Manifest: Decodable { let tensors: [String: H3RawEntry] }
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: dump.appendingPathComponent("manifest.json")))
    guard let entry = manifest.tensors["keyframe"] else { return nil }
    let data = try Data(contentsOf: dump.appendingPathComponent(entry.file))
    guard h3SHA256(data) == entry.sha256 else { throw AetherError.invalidModel("dump keyframe changed on disk") }
    return try H3Keyframe(width: geometry.width, height: geometry.height, rgb: [UInt8](data))
}

/// Wall-clock per session stage, from the session's own events.
@available(macOS 27, iOS 27, *)
private actor H3StageClock {
    private var started: [String: ContinuousClock.Instant] = [:]
    private(set) var timings: [H3Timing] = []
    private var firstStep: ContinuousClock.Instant?
    func record(_ event: H3VideoEvent, memory: BenchmarkMemory?) {
        let now = ContinuousClock.now
        func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
        switch event {
        case .stageStarted(let stage): started[stage.rawValue] = now; if stage == .denoising { firstStep = now }
        case .stageReleased(let stage):
            if let start = started[stage.rawValue] { timings.append(H3Timing(name: "stage." + stage.rawValue, seconds: seconds(now - start), memory: memory)) }
        case .step(let step, let total):
            if let start = firstStep { timings.append(H3Timing(name: "step.\(step)/\(total)", seconds: seconds(now - start), memory: memory)); firstStep = now }
        default: break
        }
    }
}

/// Records the profile denoiser's trajectory for the CPU trajectory oracle (`--capture`), or checks a fresh run
/// against one (`--replay`). Every boundary (refined text, each step's input latents, packed sequence, all 50 block
/// outputs, heads, updated latents) is hashed; `--capture` also saves the tensors the CPU references read for the
/// selected steps: input latents, packed (block 0's input), blocks 0 and 49, block 48's output (`context49`, block
/// 49's input), heads and updated latents.
@available(macOS 27, iOS 27, *)
actor H3TrajectoryRecorder {
    struct Summary { let hashes: Int; let mismatches: [String]? }
    private let out: H3RawDirectory?
    private let steps: Set<Int>?
    private let reference: [String: String]?
    private var hashes: [String: String] = [:]

    init(directory: URL, steps: String?, reference: URL?) throws {
        if let reference {
            struct Manifest: Decodable { struct Report: Decodable { let hashes: [String: String] }; let report: Report }
            let manifest = reference.appendingPathComponent("capture/manifest.json")
            self.reference = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifest)).report.hashes
            self.out = nil; self.steps = nil
        } else {
            self.reference = nil
            self.steps = try steps.flatMap { spec in
                spec == "all" ? nil : Set(try spec.split(separator: ",").map { guard let s = Int($0) else { throw AetherError.invalidRequest("bad --capture step \($0)") }; return s })
            }
            self.out = try H3RawDirectory(new: directory)
        }
    }

    private func note(_ name: String, _ tensor: H3TensorData, step: Int?) throws {
        hashes[name] = tensor.sha256
        if let out, step.map({ steps?.contains($0) ?? true }) ?? true { try out.put(name, tensor) }
    }

    func record(_ event: H3DenoiserObservation) throws {
        switch event {
        case .refined(let refined): try note("refined", refined, step: nil)
        case .stepStarted(let s, let latents):
            try note("step\(s)_latents_in_video", latents.video, step: s); try note("step\(s)_latents_in_audio", latents.audio, step: s)
        case .packed(let s, let hidden): try note("step\(s)_packed", hidden, step: s)
        case .block(let s, let index, let output):
            let name = String(format: "step%d_block%02d", s, index)
            switch index {
            case 0, 49: try note(name, output, step: s)
            case 48:
                hashes[name] = output.sha256
                if let out, steps?.contains(s) ?? true { try out.put("step\(s)_context49", output) }
            default: hashes[name] = output.sha256
            }
        case .heads(let s, let video, let audio): try note("step\(s)_heads_video", video, step: s); try note("step\(s)_heads_audio", audio, step: s)
        case .advanced(let s, let latents):
            try note("step\(s)_latents_out_video", latents.video, step: s); try note("step\(s)_latents_out_audio", latents.audio, step: s)
        }
    }

    func finish() throws -> Summary {
        struct Report: Encodable { let hashes: [String: String]; let steps: [Int]? }
        if let out {
            out.fields = ["mode": "capture"]
            try out.save(Report(hashes: hashes, steps: steps?.sorted()))
            return Summary(hashes: hashes.count, mismatches: nil)
        }
        let reference = reference ?? [:]
        let names = Set(reference.keys).union(hashes.keys).sorted()
        return Summary(hashes: hashes.count, mismatches: names.filter { reference[$0] != hashes[$0] })
    }
}
