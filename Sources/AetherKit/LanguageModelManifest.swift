import CryptoKit
import Foundation

/// Versioned registration for a local dense language model. Registration is an
/// explicit caller-owned contract; it does not establish numerical qualification.
public struct LanguageModelManifest: Codable, Sendable {
    @nonexhaustive public enum Platform: String, Codable, Sendable { case macos, ios }
    public let schemaVersion: Int
    public let modelDirectory: String
    public let tokenizerDirectory: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let modelSHA256: String
    public let tokenizerSHA256: String
    public let maximumContextLength: Int
    public let stopTokenIDs: [Int]
    public let platform: Platform
    public let quantization: String
    public let supportsReasoning: Bool?
    /// The shape bounds the asset was exported with. Required to open a bucketed-contract asset,
    /// whose submissions must stay inside them; optional for running-contract assets.
    public let exportBounds: LanguageExportBounds?
    /// SHA-256 of every file in the tokenizer directory by relative path, as the exporters record them
    /// (`tree_hashes`). When present, `verifyTokenizer` requires the directory to hold exactly these files with
    /// these hashes, so the chat template and special-token maps are covered, not only `tokenizer.json`.
    public let tokenizerFiles: [String: String]?
    /// Provenance only: the upstream model a fine-tune derives from (for example `Qwen/Qwen3-4B-Instruct-2507`).
    public let baseModel: String?
    /// Image input (Qwen3-VL): the vision encoder asset and its preprocessing. Verified with the
    /// rest of the registration; see ``verifyVision(in:)``.
    public let vision: LanguageVisionRegistration?

    public static func load(from directory: URL) throws -> Self {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: directory.appendingPathComponent("language-model.json")))
    }
    public func verify(in directory: URL, platform expectedPlatform: Platform) throws -> URL {
        let (model, tokenizer) = try locate(in: directory, platform: expectedPlatform)
        guard try Self.fileHash(model.appendingPathComponent("main.mlirb")) == modelSHA256,
              try Self.fileHash(tokenizer.appendingPathComponent("tokenizer.json")) == tokenizerSHA256 else {
            throw AetherError.invalidModel("language model or tokenizer checksum mismatch")
        }
        return model
    }
    /// The registration's structural checks without hashing anything: schema, platform, bounds, and model and
    /// tokenizer paths inside the bundle that exist. Cheap enough for a readiness check; `verify` adds the checksums.
    public func locate(in directory: URL, platform expectedPlatform: Platform) throws -> (model: URL, tokenizer: URL) {
        guard schemaVersion == 1, platform == expectedPlatform,
              maximumContextLength > 0, maximumContextLength <= 16384, !stopTokenIDs.isEmpty,
              stopTokenIDs.allSatisfy({ $0 >= 0 }), Self.hex(sourceRevision, count: 40),
              Self.hex(modelSHA256, count: 64), Self.hex(tokenizerSHA256, count: 64),
              !sourceRepository.isEmpty, exportBounds?.isValid ?? true,
              maximumContextLength <= exportBounds?.maximumCacheCapacity ?? .max
        else { throw AetherError.invalidModel("invalid language model registration or platform") }
        if let vision {
            try vision.validate()
            guard exportBounds?.maximumImageChunk.map({ vision.maximumImageTokens <= $0 }) ?? false else {
                throw AetherError.invalidModel("an image registration needs export bounds covering its largest image")
            }
        }
        let model = try Self.child(modelDirectory, of: directory)
        let tokenizer = try Self.child(tokenizerDirectory, of: directory)
        guard FileManager.default.fileExists(atPath: model.appendingPathComponent("main.mlirb").path),
              FileManager.default.fileExists(atPath: tokenizer.appendingPathComponent("tokenizer.json").path) else {
            throw AetherError.invalidModel("language model or tokenizer file missing")
        }
        return (model, tokenizer)
    }
    /// Verifies the registration and the tokenizer, returning the tokenizer directory. For tokenizer-only consumers
    /// (`AetherLanguageTokenizers`): it neither hashes the model asset nor checks the platform. `tokenizer.json` is
    /// always checked; with `tokenizerFiles`, the directory must hold exactly those files with those hashes (the
    /// chat template included). Without it, other tokenizer files are unverified.
    public func verifyTokenizer(in directory: URL) throws -> URL {
        guard schemaVersion == 1, Self.hex(sourceRevision, count: 40), Self.hex(tokenizerSHA256, count: 64),
              !sourceRepository.isEmpty, tokenizerFiles.map({ !$0.isEmpty && $0.values.allSatisfy { Self.hex($0, count: 64) } }) ?? true
        else { throw AetherError.invalidModel("invalid language model registration") }
        let tokenizer = try Self.child(tokenizerDirectory, of: directory)
        guard try Self.fileHash(tokenizer.appendingPathComponent("tokenizer.json")) == tokenizerSHA256 else {
            throw AetherError.invalidModel("language tokenizer checksum mismatch")
        }
        if let tokenizerFiles {
            guard tokenizerFiles["tokenizer.json"] == tokenizerSHA256 else {
                throw AetherError.invalidModel("tokenizer file list disagrees with tokenizerSHA256")
            }
            // Unlisted dotfiles (a Finder .DS_Store, an editor swap file) are ignored: no tokenizer loader reads them.
            let present = try Self.regularFiles(in: tokenizer).filter { path in
                tokenizerFiles[path] != nil || !path.split(separator: "/").contains { $0.hasPrefix(".") }
            }
            if let extra = present.subtracting(tokenizerFiles.keys).sorted().first {
                throw AetherError.invalidModel("tokenizer file \(extra) is not in the registration")
            }
            if let missing = Set(tokenizerFiles.keys).subtracting(present).sorted().first {
                throw AetherError.invalidModel("registered tokenizer file \(missing) is missing")
            }
            for (path, hash) in tokenizerFiles where path != "tokenizer.json" {
                guard try Self.fileHash(Self.child(path, of: tokenizer)) == hash else {
                    throw AetherError.invalidModel("language tokenizer file \(path) checksum mismatch")
                }
            }
        }
        return tokenizer
    }

    /// Relative paths of every regular file under `directory`; a link or other non-regular entry refuses.
    private static func regularFiles(in directory: URL) throws -> Set<String> {
        let base = directory.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        guard let walk = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey]) else {
            throw AetherError.invalidModel("tokenizer directory unreadable")
        }
        var files = Set<String>()
        for case let url as URL in walk {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
            if values.isDirectory == true { continue }
            guard values.isRegularFile == true else { throw AetherError.invalidModel("tokenizer directory holds a non-regular entry") }
            let path = url.standardizedFileURL.resolvingSymlinksInPath().path
            guard path.hasPrefix(base) else { throw AetherError.invalidModel("tokenizer path escapes its directory") }
            files.insert(String(path.dropFirst(base.count)))
        }
        return files
    }
    /// Verifies the vision encoder asset's checksum and returns its URL (nil without image input).
    public func verifyVision(in directory: URL) throws -> URL? {
        guard let vision else { return nil }
        try vision.validate()
        let model = try Self.child(vision.modelDirectory, of: directory)
        guard try Self.fileHash(model.appendingPathComponent("main.mlirb")) == vision.modelSHA256 else {
            throw AetherError.invalidModel("vision model checksum mismatch")
        }
        return model
    }

    public static func fileHash(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        // Drain each autoreleased chunk; otherwise a multi-GB model stays resident
        // until the synchronous verification returns.
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let data = try file.read(upToCount: 4 * 1024 * 1024), !data.isEmpty else { return false }
            hash.update(data: data)
            return true
        }) {}
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    private static func hex(_ string: String, count: Int) -> Bool {
        string.utf8.count == count && string.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    private static func child(_ path: String, of directory: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
            throw AetherError.invalidModel("language model path must be relative to its bundle")
        }
        let base = directory.standardizedFileURL.resolvingSymlinksInPath()
        let url = directory.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
        guard url.path.hasPrefix(base.path + "/") else { throw AetherError.invalidModel("language model path escapes bundle") }
        return url
    }
}

/// Shape bounds a language asset was exported with (its dynamic-dimension maxima). Kit keeps every
/// submission inside them: prefill chunks no longer than `maximumPrefillChunk` tokens and caches
/// no larger than `maximumCacheCapacity` tokens. Submissions outside the exported bounds would fail
/// inside Core AI rather than at admission.
public struct LanguageExportBounds: Codable, Sendable, Equatable {
    public let maximumPrefillChunk: Int
    public let maximumCacheCapacity: Int
    /// Largest image submission (`withImage`), for assets with image input: an image's tokens
    /// always go in one submission.
    public let maximumImageChunk: Int?

    public init(maximumPrefillChunk: Int, maximumCacheCapacity: Int, maximumImageChunk: Int? = nil) {
        self.maximumPrefillChunk = maximumPrefillChunk
        self.maximumCacheCapacity = maximumCacheCapacity
        self.maximumImageChunk = maximumImageChunk
    }

    public var isValid: Bool { maximumPrefillChunk > 0 && maximumCacheCapacity >= 2 && (maximumImageChunk ?? 1) > 0 }
}

/// A fixed patch grid the vision encoder was exported for (one static-shape function each).
public struct LanguageImageGrid: Codable, Sendable, Hashable {
    /// Patch rows and columns.
    public let height: Int
    public let width: Int

    public init(height: Int, width: Int) {
        self.height = height
        self.width = width
    }

    /// The encoder function for this grid, `encode_<height>x<width>`.
    public var functionName: String { "encode_\(height)x\(width)" }
}

/// Image input for a language registration (Qwen3-VL): the vision encoder asset, the special tokens
/// that mark an image, and the preprocessing the encoder was exported for.
public struct LanguageVisionRegistration: Codable, Sendable, Equatable {
    public let modelDirectory: String
    public let modelSHA256: String
    public let imageTokenID: Int
    public let visionStartTokenID: Int
    public let visionEndTokenID: Int
    public let patchSize: Int
    public let mergeSize: Int
    public let temporalPatchSize: Int
    public let minimumPixels: Int
    public let maximumPixels: Int
    public let imageMean: [Float]
    public let imageStandardDeviation: [Float]
    public let positionTableSide: Int
    /// Largest image the registration admits, in decoder tokens.
    public let maximumImageTokens: Int
    /// The fixed patch grids the encoder asset has static-shape functions for. Every image snaps to
    /// the grid nearest its aspect ratio and is resized to it exactly. Required: a dynamic-shape
    /// encoder intermittently computed wrong outputs on the GPU even without
    /// `expectFrequentReshapes` (docs/qwen3-vl.md), so registrations without grids are refused.
    public let imageGrids: [LanguageImageGrid]?
    /// The encoder asset's `expectFrequentReshapes` specialization request. It must be absent or
    /// false: with it, the dynamic-shape encoder computes wrong outputs on the GPU (engine F-P3-26,
    /// reproduced by Kit), so verification refuses a registration requesting it while
    /// ``reshapesQualified`` is false.
    public let expectFrequentReshapes: Bool?

    /// Whether any qualification has shown reshape-optimized specialization correct for the encoder.
    public static let reshapesQualified = false

    public init(modelDirectory: String, modelSHA256: String, imageTokenID: Int, visionStartTokenID: Int, visionEndTokenID: Int,
                patchSize: Int = 16, mergeSize: Int = 2, temporalPatchSize: Int = 2, minimumPixels: Int = 65_536,
                maximumPixels: Int = 16_777_216, imageMean: [Float] = [0.5, 0.5, 0.5],
                imageStandardDeviation: [Float] = [0.5, 0.5, 0.5], positionTableSide: Int = 48, maximumImageTokens: Int = 1024,
                imageGrids: [LanguageImageGrid]?, expectFrequentReshapes: Bool? = nil) {
        self.modelDirectory = modelDirectory
        self.modelSHA256 = modelSHA256
        self.imageTokenID = imageTokenID
        self.visionStartTokenID = visionStartTokenID
        self.visionEndTokenID = visionEndTokenID
        self.patchSize = patchSize
        self.mergeSize = mergeSize
        self.temporalPatchSize = temporalPatchSize
        self.minimumPixels = minimumPixels
        self.maximumPixels = maximumPixels
        self.imageMean = imageMean
        self.imageStandardDeviation = imageStandardDeviation
        self.positionTableSide = positionTableSide
        self.maximumImageTokens = maximumImageTokens
        self.imageGrids = imageGrids
        self.expectFrequentReshapes = expectFrequentReshapes
    }

    /// The encoder's specialization flag: false, or a refusal for a registration requesting true.
    public func encoderExpectsFrequentReshapes() throws -> Bool {
        guard expectFrequentReshapes == true else { return false }
        guard Self.reshapesQualified else {
            throw AetherError.invalidModel("the vision encoder must not be specialized with expectFrequentReshapes (F-P3-26)")
        }
        return true
    }

    public func validate() throws {
        _ = try encoderExpectsFrequentReshapes()
        let tokens = Set([imageTokenID, visionStartTokenID, visionEndTokenID])
        guard tokens.count == 3, tokens.allSatisfy({ $0 >= 0 }), patchSize > 0, mergeSize > 0, temporalPatchSize > 0,
              minimumPixels > 0, maximumPixels >= minimumPixels, imageMean.count == 3, imageStandardDeviation.count == 3,
              imageStandardDeviation.allSatisfy({ $0 > 0 }), positionTableSide > 1, maximumImageTokens > 0,
              modelSHA256.utf8.count == 64, modelSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
        else { throw AetherError.invalidModel("invalid vision registration") }
        // The token budget caps pixels at tokens · (patch · merge)²; it must cover the minimum, or the
        // smart-resize upscale branch could produce an image past the cap.
        guard let budget = imageTokenPixelBudget else {
            throw AetherError.invalidModel("the vision registration's image-token pixel budget overflows")
        }
        guard budget >= minimumPixels else {
            throw AetherError.invalidModel("the vision registration's image-token budget is below its minimum pixels")
        }
        guard let grids = imageGrids, !grids.isEmpty else {
            throw AetherError.invalidModel("the vision registration must list its fixed image grids (dynamic-shape encoders are refused)")
        }
        guard Set(grids).count == grids.count else { throw AetherError.invalidModel("the vision registration repeats an image grid") }
        for grid in grids {
            guard grid.height > 0, grid.width > 0, grid.height % mergeSize == 0, grid.width % mergeSize == 0,
                  let tokens = imageTokens(for: grid), tokens <= maximumImageTokens
            else { throw AetherError.invalidModel("image grid \(grid.height)x\(grid.width) is invalid or exceeds the image-token budget") }
        }
    }

    /// Decoder tokens of an image on `grid`: (height · width) / merge². Nil when it overflows.
    public func imageTokens(for grid: LanguageImageGrid) -> Int? {
        let (patches, overflow) = grid.height.multipliedReportingOverflow(by: grid.width)
        let (window, windowOverflow) = mergeSize.multipliedReportingOverflow(by: mergeSize)
        guard !overflow, !windowOverflow, window > 0 else { return nil }
        return patches / window
    }

    /// The pixels the image-token budget admits: one token per (patch · merge)² pixels. Nil when
    /// the product overflows (a hostile or corrupt registration), which `validate()` refuses.
    public var imageTokenPixelBudget: Int? {
        let (side, sideOverflow) = patchSize.multipliedReportingOverflow(by: mergeSize)
        let (area, areaOverflow) = side.multipliedReportingOverflow(by: side)
        let (pixels, pixelsOverflow) = maximumImageTokens.multipliedReportingOverflow(by: area)
        return sideOverflow || areaOverflow || pixelsOverflow ? nil : pixels
    }
}
