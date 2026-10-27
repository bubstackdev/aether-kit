import AetherFlux2
import AetherFlux2Tokenizers
import AetherKit
import AetherLanguageTokenizers
import CryptoKit
import Foundation
import Testing

private let environment = ProcessInfo.processInfo.environment

/// Weight-free checks against the independent float32 Diffusers fixtures (opt-in by path):
/// schedule timesteps and reference positions exact, and CPU Euler replay of the reference's own
/// velocities reproducing every following latent and the final latent bit for bit.
@Test(.enabled(if: environment["AETHER_FLUX2_FIXTURES"] != nil || environment["AETHER_FLUX2_REFERENCE_FIXTURES"] != nil))
func samplerArithmeticMatchesTheIndependentReference() throws {
    for key in ["AETHER_FLUX2_FIXTURES", "AETHER_FLUX2_REFERENCE_FIXTURES"] {
        guard let path = environment[key] else { continue }
        let root = URL(fileURLWithPath: path)
        let document = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("fixture.json"))) as! [String: Any]
        let hashes = document["sha256"] as! [String: String]
        func data(_ name: String) throws -> Data {
            let bytes = try Data(contentsOf: root.appendingPathComponent(name))
            #expect(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() == hashes[name], "\(name)")
            return bytes
        }
        func floats(_ name: String) throws -> [Float] {
            try data(name).withUnsafeBytes { raw in (0..<(raw.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) } }
        }
        for line in try String(decoding: data("cases.jsonl"), as: UTF8.self).split(separator: "\n") {
            let c = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
            let id = c["id"] as! String, width = c["width"] as! Int, height = c["height"] as! Int, steps = c["steps"] as! Int
            let files = c["files"] as! [String: Any]
            let tokens = (width / 16) * (height / 16)
            let schedule = try Flux2Schedule(steps: steps, imageTokens: tokens)
            #expect(schedule.timesteps == (c["timesteps"] as! [Double]).map(Float.init), "\(id) timesteps")
            let latents = try (files["stepLatents"] as! [String]).map(floats)
            let velocities = try (files["stepVelocities"] as! [String]).map(floats)
            #expect(latents[0] == (try floats(files["noise"] as! String)), "\(id) initial noise")
            var x = latents[0]
            for step in 0..<steps {
                #expect(x == latents[step], "\(id) step \(step) input")
                x = try schedule.advance(step: step, latents: latents[step], velocity: velocities[step])
            }
            #expect(x == (try floats(files["final"] as! String)), "\(id) final latent")
            if let positions = files["positions"] as? String, let refs = c["references"] as? [[String: Int]] {
                var ids = Flux2Rotary.gridIDs(rows: height / 16, columns: width / 16)
                for (k, r) in refs.enumerated() { ids += Flux2Rotary.gridIDs(rows: r["height"]! / 16, columns: r["width"]! / 16, t: Int32(10 * (k + 1))) }
                let recorded = try data(positions).withUnsafeBytes { raw in (0..<(raw.count / 4)).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) } }
                #expect(recorded == ids, "\(id) positions")
            }
        }
    }
}

/// The bundle tokenizer reproduces the reference token ids and mask (local files only).
@Test(.enabled(if: environment["AETHER_FLUX2_BUNDLE"] != nil && environment["AETHER_FLUX2_FIXTURES"] != nil))
func bundleTokenizerMatchesReferenceTokenIDs() async throws {
    let assets = try Flux2Assets.open(at: URL(fileURLWithPath: environment["AETHER_FLUX2_BUNDLE"]!))
    let tokenizer = try await Flux2Tokenizer.open(assets: assets)
    let root = URL(fileURLWithPath: environment["AETHER_FLUX2_FIXTURES"]!)
    for line in try String(contentsOf: root.appendingPathComponent("cases.jsonl"), encoding: .utf8).split(separator: "\n") {
        let c = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
        let files = c["files"] as! [String: Any]
        func ints(_ name: String) throws -> [Int32] {
            try Data(contentsOf: root.appendingPathComponent(name)).withUnsafeBytes { raw in
                (0..<(raw.count / 4)).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) }
            }
        }
        let prompt = try tokenizer.prepare(c["prompt"] as! String)
        #expect(prompt.tokenIDs == (try ints(files["ids"] as! String)), "\(c["id"]!) ids")
        #expect(prompt.attentionMask == (try ints(files["mask"] as! String)), "\(c["id"]!) mask")
    }
}

/// `Flux2Tokenizer` renders through the bundle's own Qwen3 template (`AetherLanguageTokenizers`, thinking off), and its
/// padded prompt equals the fixed render it replaced, encoded by the same tokenizer: the same 512 IDs, mask and
/// truncation flag, for every fixture prompt and for edge cases including a prompt longer than 512 tokens (local files
/// only; CPU).
@Test(.enabled(if: environment["AETHER_FLUX2_BUNDLE"] != nil))
func templateRenderReproducesTheFixedQwen3Render() async throws {
    let assets = try Flux2Assets.open(at: URL(fileURLWithPath: environment["AETHER_FLUX2_BUNDLE"]!))
    let flux = try await Flux2Tokenizer.open(assets: assets)
    let language = try await LanguageTokenizer.open(directory: assets.tokenizerDirectory)
    // The 0.0.2 `Flux2Prompt.render`: Qwen3's chat template with `enable_thinking=False`, written out.
    func fixed(_ text: String) -> String {
        "<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }
    let long = String(repeating: "a lighthouse at dusk, ", count: 120)
    var prompts = ["a red fox in the snow", "Café — 🌙\nsecond line", "  padded  \n\n", "", "<think>x</think>", long]
    for key in ["AETHER_FLUX2_FIXTURES", "AETHER_FLUX2_REFERENCE_FIXTURES"] {
        guard let fixtures = environment[key] else { continue }
        let cases = try String(contentsOf: URL(fileURLWithPath: fixtures).appendingPathComponent("cases.jsonl"), encoding: .utf8)
        for line in cases.split(separator: "\n") {
            let c = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
            prompts.append(c["prompt"] as! String)
        }
    }
    for text in prompts {
        let previous = language.encode(fixed(text))
        let prompt = try flux.prepare(text)
        #expect(prompt == (try Flux2Prompt(renderedTokenIDs: previous)), "\(text.prefix(40))")
        #expect(prompt.tokenIDs.count == Flux2Prompt.length && prompt.attentionMask.count == Flux2Prompt.length)
        // Truncation follows each prompt's rendered length; the synthetic long prompt must exercise it.
        #expect(prompt.wasTruncated == (previous.count > Flux2Prompt.length), "\(text.prefix(40))")
        if text == long { #expect(prompt.wasTruncated) }
    }
}
