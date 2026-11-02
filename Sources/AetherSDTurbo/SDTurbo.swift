import AetherDiffusion
import AetherDiffusionCoreAI
import AetherKit
import Foundation
import CryptoKit

/// Spike bundle contract. Native geometry and schedules are explicit, never inferred from filenames.
public struct SDTurboBundle: Sendable {
    public let root: URL
    public let width: Int
    public let height: Int
    public let schedules: [String: EpsilonSchedule]
    public static let sourceRevision = "b261bac6fd2cf515557d5d0707481eafa0485ec2"

    public init(at root: URL) throws {
        struct Source: Decodable { let repository: String; let revision: String }
        struct Manifest: Decodable {
            let schema: Int; let family: String; let source: Source
            let width: Int; let height: Int; let latentChannels: Int; let latentScale: Int
            let textLength: Int; let textWidth: Int; let scalarType: String
            let schedules: [String: EpsilonSchedule]; let files: [String: String]
        }
        guard root.isFileURL else { throw AetherError.invalidModel("bundle must be local") }
        let m = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: root.appendingPathComponent("diffusion.json")))
        guard m.schema == 1, m.family == "sd-turbo", m.source.repository == "stabilityai/sd-turbo",
              m.source.revision == Self.sourceRevision, m.latentChannels == 4, m.latentScale == 8,
              m.textLength == 77, m.textWidth == 1024, m.scalarType == "float32",
              (64...512).contains(m.width), (64...512).contains(m.height),
              m.width.isMultiple(of: 64), m.height.isMultiple(of: 64), Set(m.schedules.keys) == ["1", "2", "3", "4"],
              m.schedules.allSatisfy({ Int($0.key) == $0.value.timesteps.count }),
              ["textEncoder.aimodel", "denoiser.aimodel", "decoder.aimodel", "tokenizer"].allSatisfy({ dir in
                  m.files.keys.contains { $0.hasPrefix(dir + "/") }
              }) else { throw AetherError.invalidModel("unsupported SD-Turbo bundle contract") }
        for (path, expected) in m.files {
            guard !path.hasPrefix("/"), !path.contains("\\"), path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
                throw AetherError.invalidModel("unsafe bundle path")
            }
            let file = root.appendingPathComponent(path)
            guard file.resolvingSymlinksInPath().path.hasPrefix(root.resolvingSymlinksInPath().path + "/") else {
                throw AetherError.invalidModel("bundle file escapes root")
            }
            let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
            var hash = SHA256()
            while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty { hash.update(data: chunk) }
            guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == expected else {
                throw AetherError.invalidModel("bundle checksum mismatch: \(path)")
            }
        }
        self.root = root; width = m.width; height = m.height; schedules = m.schedules
    }

    public func schedule(steps: Int) throws -> EpsilonSchedule {
        guard let schedule = schedules[String(steps)] else { throw AetherError.invalidRequest("SD-Turbo steps must be 1...4") }
        return schedule
    }
    public func noise(seed: UInt64) throws -> FloatTensor {
        try FloatTensor(shape: [1, 4, height / 8, width / 8], values: SeededGaussianNoise.generate(seed: seed, count: 4 * height / 8 * width / 8))
    }
}

@available(macOS 27, iOS 27, *)
extension SDTurboBundle {
    /// Open an isolated reference session. With no admission object, the host owns GPU coordination.
    public func makeSession(gpu: GPUAdmissionOwner? = nil) throws -> DiffusionSession {
        let backend = try CoreAIEpsilonBackend(textEncoder: root.appendingPathComponent("textEncoder.aimodel"),
            denoiser: root.appendingPathComponent("denoiser.aimodel"), decoder: root.appendingPathComponent("decoder.aimodel"),
            width: width, height: height, latentChannels: 4, latentScale: 8,
            textLength: 77, textWidth: 1024, vocabularySize: 49408)
        return DiffusionSession(backend: backend, gpu: gpu)
    }
}
