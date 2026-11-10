import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing
@_spi(Testing) @testable import AetherVideo

/// The CPU-side clip API, hermetic: the clip limits and the preflight against generation's own refusals (a prompt too
/// long for the clip, a clip no prompt fits, both typed), and the streamed decode through a fake video decoder (frames
/// in order, none kept, a sink that throws stops it), and the seed echoed on the clip.
@Suite("MiniMax H3 preflight and streaming")
struct MiniMaxH3PreflightTests {
    /// A pipeline over `manifest` whose loader counts loads and fails: nothing here may load a graph.
    static func pipeline(
        _ manifest: Manifest, policy: VideoAdmissionPolicy = .measuring, loads: MiniMaxH3AdmissionTests.Loads = .init()
    ) throws -> MiniMaxH3Pipeline {
        try MiniMaxH3AdmissionTests.pipeline(manifest, policy: policy, loads: loads)
    }

    /// The keyframe test manifest (the 960×544 vision tower) with `h3.limits.textTokens` replaced.
    static func manifest(textTokens: [Int]? = nil) throws -> Manifest {
        let manifest = try MiniMaxH3KeyframeTests.manifest()
        guard let textTokens else { return manifest }
        // The keyframe manifest's JSON: the fixture's with the 960×544 tower and the vision block.
        var json = try MiniMaxH3AdmissionTests.json()
        var graphs = json["graphs"] as! [String: Any]
        graphs["conditionerVision960x544"] = [
            "file": "main:models/conditioner-vision-960x544.aimodel", "function": "main",
            "inputs": ["pixels": "pixels", "posIndex": "pos_index", "posWeight": "pos_weight", "patchRC": "patch_rc"],
            "outputs": ["imageEmbeds": "image_embeds", "deepstack": "deepstack"], "residency": "perPrompt", "compute": "gpu"]
        json["graphs"] = graphs
        var h3 = json["h3"] as! [String: Any]
        h3["vision"] = ["patchSize": 16, "mergeSize": 2, "temporalPatchSize": 2, "tableSide": 48, "deepstackLevels": 3,
                        "presentation": ["imagePad": 151655, "label": "<Picture {index}>: ", "visionEnd": 151653,
                                         "visionStart": 151652]]
        var limits = h3["limits"] as! [String: Any]
        limits["textTokens"] = textTokens
        h3["limits"] = limits
        json["h3"] = h3
        return try Manifest.decode(JSONSerialization.data(withJSONObject: json))
    }

    static func keyframes(_ count: Int) throws -> VideoKeyframes {
        let first = try MiniMaxH3KeyframeTests.keyframe(960, 544), last = try MiniMaxH3KeyframeTests.keyframe(960, 544, last: true)
        return count == 1 ? .first(first) : .firstAndLast(first, last)
    }

    // MARK: Limits and preflight

    @Test func clipLimitsCountThePresentationAndTheClipRows() throws {
        let pipeline = try Self.pipeline(try Self.manifest())
        let t2va = try pipeline.clipLimits()
        // 960×544×124: 18,870 video rows and 414 audio rows; the text bound (2…4096) is the tighter one.
        #expect(t2va.profile == "960x544x124" && t2va.keyframeCount == 0 && t2va.keyframePresentationTokens == 0)
        #expect(t2va.clipRows == 18870 + 414 && t2va.presentationTokenRange == 2...4096 && t2va.packedRowRange == 64...40960)
        #expect(t2va.maximumPromptTokens == 4096 && t2va.minimumPromptTokens == 2 && t2va.fitsSomePrompt)
        #expect(t2va.packedRows(promptTokens: 13) == 19297)

        // Each keyframe: its label ("<Picture i>: ", 13 scalars here), vision start and end, 30 × 17 image pads; and
        // one latent frame of condition rows.
        let fl2va = try pipeline.clipLimits(profile: "960x544x124", keyframeCount: 2)
        #expect(fl2va.keyframePresentationTokens == 2 * (13 + 2 + 510) && fl2va.clipRows == 2 * 510 + 414 + 18870)
        #expect(fl2va.maximumPromptTokens == 4096 - 1050)
        #expect(fl2va.presentationTokens(promptTokens: 10) == 1060)

        #expect(throws: AetherError.self) { _ = try pipeline.clipLimits(keyframeCount: 3) }
        #expect(throws: AetherError.self) { _ = try pipeline.clipLimits(profile: "640x480x124") }
        // Without the export's vision tower for the canvas, keyframes are unsupported.
        let error = #expect(throws: AetherError.self) { _ = try pipeline.clipLimits(profile: "704x704x124", keyframeCount: 1) }
        guard case .unsupportedOperation? = error else { Issue.record("\(String(describing: error))"); return }
    }

    @Test func preflightResolvesAsGenerationDoes() throws {
        let pipeline = try Self.pipeline(try Self.manifest())
        let plain = try pipeline.preflight(VideoRequest(prompt: "a fox in snow", seed: 42))
        #expect(plain.profile == "960x544x124" && plain.recipe == "turbo8" && plain.seed == 42 && plain.frameCount == 124)
        #expect(plain.promptTokens == 13 && plain.presentationTokens == 13 && plain.packedRows == 19297)
        let t2va = try pipeline.clipLimits(), fl2va = try pipeline.clipLimits(keyframeCount: 2)
        #expect(plain.admissionKey == VideoAdmissionKey(profile: "960x544x124") && plain.envelope == nil && plain.limits == t2va)

        let anchored = try pipeline.preflight(VideoRequest(prompt: "a fox in snow", keyframes: try Self.keyframes(2)))
        #expect(anchored.admissionKey == VideoAdmissionKey(profile: "960x544x124", keyframeCount: 2) && anchored.presentationTokens == 1050 + 13)
        #expect(anchored.packedRows == 1063 + 2 * 510 + 414 + 18870)
        #expect(anchored.limits == fl2va)
        // The plan's own packed layout has the rows preflight reports.
        let plan = try pipeline.plan(VideoRequest(prompt: "a fox in snow", keyframes: try Self.keyframes(2)))
        #expect(H3Layout(textTags: plan.textTags, geometry: plan.geometry, anchors: plan.anchors).rows == anchored.packedRows)

        // Under measured admission, an unmeasured key is refused by preflight as by generation.
        let measured = try Self.pipeline(try Self.manifest(), policy: .measuredOnly)
        let error = #expect(throws: AetherError.self) { _ = try measured.preflight(VideoRequest(prompt: "a fox in snow")) }
        #expect(MiniMaxH3AdmissionTests.isUnmeasured(error))
    }

    @Test func aPromptTooLongIsRefusedTypedBeforeLoading() async throws {
        let loads = MiniMaxH3AdmissionTests.Loads()
        let pipeline = try Self.pipeline(try Self.manifest(), loads: loads)
        let request = VideoRequest(prompt: String(repeating: "a", count: 4097))
        let preflighted = #expect(throws: AetherError.self) { _ = try pipeline.preflight(request) }
        #expect(preflighted.flatMap(VideoRefusal.init) == .promptTooLong(promptTokens: 4097, maximumPromptTokens: 4096))
        let generated = await MiniMaxH3AdmissionTests.failure(pipeline, request)
        #expect(generated.flatMap(VideoRefusal.init) == .promptTooLong(promptTokens: 4097, maximumPromptTokens: 4096))
        guard case .capacity(let what, let needed, let available)? = generated else { Issue.record("\(String(describing: generated))"); return }
        #expect(what == VideoRefusal.promptTooLongResource && needed == 4097 && available == 4096)
        #expect(loads.count.withLock { $0 } == 0)

        // With keyframes the presentation takes its share: 4,096 − 525 tokens are left for the prompt.
        let anchored = VideoRequest(prompt: String(repeating: "a", count: 3572), keyframes: try Self.keyframes(1))
        let error = #expect(throws: AetherError.self) { _ = try pipeline.preflight(anchored) }
        #expect(error.flatMap(VideoRefusal.init) == .promptTooLong(promptTokens: 3572, maximumPromptTokens: 3571))
        var fits = anchored
        fits.prompt = String(repeating: "a", count: 3571)
        let fitting = try pipeline.preflight(fits)
        #expect(fitting.presentationTokens == 4096)

        // Too short stays an invalid request, not a capacity refusal.
        let short = #expect(throws: AetherError.self) { _ = try pipeline.preflight(VideoRequest(prompt: "x")) }
        guard case .generation(.invalidParameters)? = short else { Issue.record("\(String(describing: short))"); return }
    }

    @Test func aClipNoPromptFitsIsTooLargeAtSize() async throws {
        let loads = MiniMaxH3AdmissionTests.Loads()
        // A 1,000-token presentation bound: one keyframe (525) leaves 475; two (1,050) leave none.
        let pipeline = try Self.pipeline(try Self.manifest(textTokens: [2, 1000]), loads: loads)
        let two = try pipeline.clipLimits(keyframeCount: 2)
        #expect(!two.fitsSomePrompt && two.maximumPromptTokens == -50)
        #expect(two.refusal(promptTokens: 1) == .tooLargeAtSize(minimumPromptTokens: 1, maximumPromptTokens: 0))
        let one = try pipeline.clipLimits(keyframeCount: 1)
        #expect(one.refusal(promptTokens: 476) == .promptTooLong(promptTokens: 476, maximumPromptTokens: 475))

        let request = VideoRequest(prompt: "a fox", keyframes: try Self.keyframes(2))
        let preflighted = #expect(throws: AetherError.self) { _ = try pipeline.preflight(request) }
        #expect(preflighted.flatMap(VideoRefusal.init) == .tooLargeAtSize(minimumPromptTokens: 1, maximumPromptTokens: 0))
        let generated = await MiniMaxH3AdmissionTests.failure(pipeline, request)
        guard case .capacity(let what, 1, 0)? = generated else { Issue.record("\(String(describing: generated))"); return }
        #expect(what == VideoRefusal.tooLargeAtSizeResource && loads.count.withLock { $0 } == 0)

        // The refusal round-trips through its error; other errors aren't refusals.
        for refusal in [VideoRefusal.promptTooLong(promptTokens: 9, maximumPromptTokens: 8),
                        .tooLargeAtSize(minimumPromptTokens: 2, maximumPromptTokens: 0)] {
            #expect(VideoRefusal(refusal.error) == refusal)
        }
        #expect(VideoRefusal(.capacity(what: "memory", needed: 2, available: 1)) == nil)
        #expect(VideoRefusal(.memoryPressure) == nil)
    }

    // MARK: Streaming

    /// A video decoder whose every pixel of call `n` is `n` (before the ImageNet denormalization).
    final class FakeVideoDecoder: GraphBackend {
        let signature = GraphSignature(
            function: "main", inputs: [TensorSpec(name: "rows", scalarType: .float32, shape: [448, 96])], states: [],
            outputs: [TensorSpec(name: "pixels", scalarType: .float32, shape: [1, 3, 22, 256, 256])])
        let calls = Mutex(0)

        func run(_ bindings: inout Bindings) async throws(AetherError) {
            try bindings.validate(against: signature)
            let call = calls.withLock { value in
                defer { value += 1 }
                return value
            }
            try bindings.output("pixels")!.write([Float](repeating: Float(call) * 0.5, count: 3 * 22 * 256 * 256))
        }
    }

    /// Two decoder windows of one 256 px tile: 2 × 17 + 5 = 39 frames.
    static let geometry = H3Geometry(width: 256, height: 256, frames: 39, latentFrames: 12, audioLatents: 3)

    static func decode(
        _ decoder: FakeVideoDecoder, sink: (@Sendable (VideoFrame) async throws -> Void)?
    ) async throws -> (frames: [[UInt8]], count: Int) {
        let g = geometry
        #expect(g.videoWindows == 2)
        return try await MiniMaxH3Pipeline.decodeVideo(
            [Float](repeating: 0, count: g.videoRows * 96), geometry: g, decoder: decoder,
            names: (["rows": "rows"], ["pixels": "pixels"]), host: TensorArena(), sample: {}, onProgress: nil, sink: sink)
    }

    final class Delivered: Sendable {
        let frames = Mutex<[VideoFrame]>([])
    }

    @Test func streamedFramesArriveInOrderAndAreNotKept() async throws {
        let kept = try await Self.decode(FakeVideoDecoder(), sink: nil)
        #expect(kept.count == 39 && kept.frames.count == 39)

        let delivered = Delivered()
        let decoder = FakeVideoDecoder()
        let streamed = try await Self.decode(decoder, sink: { frame in delivered.frames.withLock { $0.append(frame) } })
        let frames = delivered.frames.withLock { $0 }
        #expect(streamed.count == 39 && streamed.frames.isEmpty && decoder.calls.withLock { $0 } == 2)
        #expect(frames.map(\.index) == Array(0..<39))
        #expect(frames.allSatisfy { $0.width == 256 && $0.height == 256 && $0.rgb.count == 256 * 256 * 3 })
        // The same frames, in the same order, as the kept decode: the first window's, the cross-fade, the second's tail.
        #expect(frames.map(\.rgb) == kept.frames)
        #expect(frames[0].rgb != frames[38].rgb)
        #expect(frames[24].time == .seconds(1))
    }

    @Test func aSinkThatThrowsStopsTheDecode() async throws {
        struct Stop: Error {}
        let decoder = FakeVideoDecoder()
        let delivered = Delivered()
        await #expect(throws: Stop.self) {
            _ = try await Self.decode(decoder, sink: { frame in
                delivered.frames.withLock { $0.append(frame) }
                if frame.index == 3 { throw Stop() }
            })
        }
        // It stopped at the fourth frame, inside the first window: the second window was never decoded.
        #expect(delivered.frames.withLock { $0.count } == 4 && decoder.calls.withLock { $0 } == 1)
    }

    // MARK: Seed

    @Test func theClipCarriesItsSeedAndFrameCount() {
        let kept = GeneratedVideo(width: 2, height: 1, frames: [[0, 0, 0, 0, 0, 0]], audio: nil, seed: 7)
        #expect(kept.seed == 7 && kept.frameCount == 1)
        let streamed = GeneratedVideo(width: 2, height: 1, frames: [], frameCount: 48, audio: nil, seed: .max)
        #expect(streamed.seed == .max && streamed.frameCount == 48 && streamed.duration == .seconds(2))
    }
}
