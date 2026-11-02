import AetherKit
import AetherH3
import AetherH3CoreAI
import AetherH3Tokenizers
import Foundation
import ImageIO
import Metal

/// Native qualification of FL2VA's conditioner presentation (`H3Presentation`): the keyframe vision tower alone, or the
/// whole presentation through the conditioner. Writes the CPU presentation oracle's tensor names
/// (`prepare-h3-profile-oracle.py --part presentation`), so `--part compare` sets the two side by side, plus a 20 ms
/// Metal/footprint timeline per phase.
@available(macOS 27, iOS 27, *)
enum H3PresentationCommand {
    static let usage = "use h3-profile-presentation PROFILE.json PROMPT.txt KEYFRAME.png NEW_DIRECTORY --vision|--conditioner"

    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "h3-profile-presentation" else { return false }
        guard arguments.count == 6, ["--vision", "--conditioner"].contains(arguments[5]) else { throw AetherError.invalidRequest(usage) }
        try await execute(profile: URL(fileURLWithPath: arguments[1]), prompt: URL(fileURLWithPath: arguments[2]),
                          keyframe: URL(fileURLWithPath: arguments[3]), output: URL(fileURLWithPath: arguments[4]),
                          conditioner: arguments[5] == "--conditioner")
        return true
    }

    private struct Report: Encodable {
        let mode: String
        let phases: [H3MemoryTimeline.Phase]
        let metalPeakBytes: UInt64
        let footprintPeakBytes: UInt64
        let visionSpecialization: String
    }

    private static func execute(profile url: URL, prompt: URL, keyframe file: URL, output: URL, conditioner: Bool) async throws {
        let profile = try H3Profile(url: url)
        guard let vision = profile.conditionerVision else { throw AetherError.unsupported("profile has no keyframe vision encoder") }
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int, let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw AetherError.invalidRequest("keyframe is not a readable image")
        }
        let keyframe = try H3ProfileCommand.keyframe(file, width: width, height: height)
        let text = try String(contentsOf: prompt, encoding: .utf8)
        let tokenizer = try await H3Tokenizer.open(directory: profile.tokenizer)
        let ids = try tokenizer.tokenIDs(text)
        let presentation = try H3Presentation(prompt: ids, keyframes: [(width, height)], vision: vision.vision, tokenize: tokenizer.tokenIDs)
        let out = try H3RawDirectory(new: output)
        out.fields = ["profile_sha256": profile.manifestSHA256, "canvas": "\(width)x\(height)", "prompt_sha256": h3SHA256(Data(text.utf8)),
                      "keyframe_sha256": h3SHA256(Data(keyframe.rgb)), "mode": conditioner ? "conditioner" : "vision"]
        try out.putBytes("keyframe0", Data(keyframe.rgb), dtype: "uint8", shape: [height, width, 3])
        try out.put("token_ids", try ints(presentation.ids, [presentation.count]))
        try out.put("text_token_tags", try ints(presentation.tags, [presentation.count]))
        try out.put("position_ids", try ints(presentation.positions.flatMap { $0 }, [3, presentation.count]))
        let inputs = try H3VisionInputs(rgb: keyframe.rgb, width: width, height: height, vision: vision.vision)
        try out.put("pixel_values", inputs.pixels)

        let timeline = H3MemoryTimeline(device: MTLCreateSystemDefaultDevice())
        let sampling = Task.detached { while !Task.isCancelled { timeline.sample(); try? await Task.sleep(for: .milliseconds(20)) } }
        defer { sampling.cancel() }
        timeline.mark("baseline"); try? await Task.sleep(for: .seconds(1))
        let qualifier = try H3ProfileQualifier(profile: profile)
        timeline.mark("vision")
        let images = try await qualifier.vision(keyframe)
        timeline.mark("vision.released"); try? await Task.sleep(for: .seconds(1))
        try out.put("image_embeds", images.embeddings)
        try out.put("deepstack", images.deepstack)
        if conditioner {
            timeline.mark("conditioner")
            // Layers 0-3 and 49, the decoder-layer outputs the presentation oracle records.
            let (result, layers) = try await qualifier.conditioning(ids: ids, keyframes: [keyframe], tokenize: tokenizer.tokenIDs,
                                                                    layers: [0, 1, 2, 3, 49])
            timeline.mark("conditioner.released")
            await qualifier.release(); try? await Task.sleep(for: .seconds(1))
            guard result.tags == presentation.tags else { throw AetherError.invalidModel("conditioner tags differ from the presentation") }
            try out.put("tap", result.tap)
            for (index, hidden) in layers.sorted(by: { $0.key < $1.key }) { try out.put(String(format: "layer%02d", index), hidden) }
        }
        timeline.mark("end"); timeline.sample()
        let samples = timeline.samples
        try out.save(Report(mode: conditioner ? "conditioner" : "vision", phases: timeline.phases(),
                            metalPeakBytes: samples.map(\.metal).max() ?? 0, footprintPeakBytes: samples.map(\.footprint).max() ?? 0,
                            visionSpecialization: "gpu, expectFrequentReshapes false"))
        print(output.path)
    }

    private static func ints(_ values: [Int32], _ shape: [Int]) throws -> H3TensorData {
        try H3TensorData(spec: H3TensorSpec(scalarType: .int32, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
    }
}
