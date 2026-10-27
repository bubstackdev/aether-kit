import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision
import Darwin
import Foundation
import Metal
import Testing

/// Opt-in attribution of a Qwen3-VL bundle's memory by phase: the encoder alone (open, one grid, all grids), then the
/// decoder (open, text-only requests, the first image request, each new grid), then repeated passes over
/// every grid to show whether allocations still grow once all shapes are warm, then unload. Each phase records the
/// process footprint and Metal's allocated size. Images are synthetic gradients at each grid's exact pixel size. The
/// decoder has no memory budget, so no reload interrupts the phases. It attributes the per-function and per-grid weight
/// copies documented in docs/qwen3-vl.md (Limits). To tell private copies from shared, file-backed pages, each phase
/// also records residency: the process's resident, anonymous (internal) and file-backed (external) sizes and its graphics
/// ledgers (counted in the footprint or not), the host's free, wired, file-backed, anonymous and compressed pages (a
/// shared host: other processes move these too), and `footprint`'s categories of at least 16 MiB.
///
/// AETHER_NATIVE_VL_MEMORY=1
/// AETHER_VL_MEMORY_BUNDLE=<bundle with language-model.json and a vision registration>
/// AETHER_VL_MEMORY_OUTPUT=<report .json>
private struct MemoryPhase: Codable {
    let phase: String, footprintMiB: Int, metalMiB: Int, metalDeltaMiB: Int
    var residency: Residency?
}

/// MiB throughout. `categories`: footprint's dirty, clean and wired bytes per category of at least 16 MiB.
private struct Residency: Codable {
    var residentMiB = 0, anonymousMiB = 0, fileBackedMiB = 0, graphicsFootprintMiB = 0, graphicsNoFootprintMiB = 0
    var hostFreeMiB = 0, hostWiredMiB = 0, hostFileBackedMiB = 0, hostAnonymousMiB = 0, hostCompressedMiB = 0
    var categories: [String: [String: Int]] = [:]
}

private func run(_ tool: String, _ arguments: [String]) -> String? {
    let process = Process(), pipe = Pipe()
    process.executableURL = URL(filePath: tool); process.arguments = arguments
    process.standardOutput = pipe; process.standardError = FileHandle.nullDevice
    guard (try? process.run()) != nil else { return nil }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return String(data: data, encoding: .utf8)
}

private func residencyNow() -> Residency {
    var r = Residency()
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    if withUnsafeMutablePointer(to: &info, { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }) == KERN_SUCCESS {
        r.residentMiB = Int(info.resident_size >> 20); r.anonymousMiB = Int(info.internal >> 20); r.fileBackedMiB = Int(info.external >> 20)
        r.graphicsFootprintMiB = Int(info.ledger_tag_graphics_footprint >> 20)
        r.graphicsNoFootprintMiB = Int(info.ledger_tag_graphics_nofootprint >> 20)
    }
    // The host's pages, from vm_stat itself.
    if let text = run("/usr/bin/vm_stat", []) {
        let page = Int(text.firstMatch(of: /page size of (\d+) bytes/)?.1 ?? "") ?? 16384
        func mib(_ label: String) -> Int {
            for line in text.split(separator: "\n") where line.hasPrefix(label + ":") {
                return (Int(line.split(separator: " ").last?.trimmingCharacters(in: CharacterSet(charactersIn: "."))
                    ?? "") ?? 0) * page / (1 << 20)
            }
            return 0
        }
        r.hostFreeMiB = mib("Pages free"); r.hostWiredMiB = mib("Pages wired down")
        r.hostFileBackedMiB = mib("File-backed pages"); r.hostAnonymousMiB = mib("Anonymous pages")
        r.hostCompressedMiB = mib("Pages occupied by compressor")
    }
    // footprint's categories (IOAccelerator, IOSurface, mapped file, ...), from the tool itself.
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("vl-memory-footprint-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: file) }
    let tool = Process()
    tool.executableURL = URL(filePath: "/usr/bin/footprint")
    tool.arguments = ["-p", String(getpid()), "-f", "bytes", "-w", "-j", file.path]
    tool.standardOutput = FileHandle.nullDevice; tool.standardError = FileHandle.nullDevice
    if (try? tool.run()) != nil {
        tool.waitUntilExit()
        if let data = try? Data(contentsOf: file), let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let process = (json["processes"] as? [[String: Any]])?.first, let categories = process["categories"] as? [String: [String: Any]] {
            for (name, values) in categories {
                let sizes = ["dirty", "clean", "wired"].reduce(into: [String: Int]()) { $0[$1] = ((values[$1] as? Int) ?? 0) >> 20 }
                if sizes.values.contains(where: { $0 >= 16 }) { r.categories[name] = sizes }
            }
        }
    }
    return r
}

private struct MemoryReport: Codable {
    var bundle = ""
    var device = LanguageGenerationSession.deviceArchitectureName
    var stage = "preparation"
    var error: String?
    var phases: [MemoryPhase] = []
    var promptTokens: [String: Int] = [:]
    /// The decoder's image steps go through `main` (the one-function asset) rather than `withImage`.
    var imagesThroughMain: Bool?
}

private let memoryDevice = MTLCreateSystemDefaultDevice()

private func memoryNow() -> (footprint: Int, metal: Int) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return (result == KERN_SUCCESS ? Int(info.phys_footprint >> 20) : 0, (memoryDevice?.currentAllocatedSize ?? 0) >> 20)
}

/// A smooth synthetic RGB gradient, `width` x `height`.
private func gradient(width: Int, height: Int, seed: Int) throws -> VisionRGBImage {
    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let i = 3 * (y * width + x)
            rgb[i] = UInt8((x * 255 / max(1, width - 1) + seed * 40) % 256)
            rgb[i + 1] = UInt8((y * 255 / max(1, height - 1) + seed * 70) % 256)
            rgb[i + 2] = UInt8(((x + y) * 127 / max(1, width + height - 2) + seed * 90) % 256)
        }
    }
    return try VisionRGBImage(rgb: rgb, width: width, height: height)
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_VL_MEMORY"] == "1"), .timeLimit(.minutes(20)))
func visionLanguageMemoryByPhase() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    var report = MemoryReport()
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_VL_MEMORY_OUTPUT"]))
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    var lastMetal = memoryNow().metal
    func sample(_ phase: String) {
        let now = memoryNow()
        report.phases.append(MemoryPhase(phase: phase, footprintMiB: now.footprint, metalMiB: now.metal, metalDeltaMiB: now.metal - lastMetal,
                                         residency: residencyNow()))
        lastMetal = now.metal
    }
    do {
        let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_VL_MEMORY_BUNDLE"]))
        report.bundle = bundle.lastPathComponent
        let manifest = try LanguageModelManifest.load(from: bundle)
        let textModel = try manifest.verify(in: bundle, platform: .macos)
        let registration = try #require(manifest.vision)
        let visionModel = try #require(try manifest.verifyVision(in: bundle))
        let grids = try #require(registration.imageGrids)
        let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
        let tokens = LanguageImageTokens(visionStart: registration.visionStartTokenID, imagePad: registration.imageTokenID,
                                         visionEnd: registration.visionEndTokenID)
        sample("before open")

        // Encoder alone.
        report.stage = "encoder"
        var configuration = Qwen3VLImagePreprocessor.Configuration()
        configuration.imageGrids = grids.map { .init(height: $0.height, width: $0.width) }
        let encoder = try await LanguageVisionEncoder.open(at: visionModel,
            configuration: .init(mergeSize: registration.mergeSize, positionTableSide: registration.positionTableSide,
                                 imageGrids: configuration.imageGrids ?? []))
        sample("encoder open")
        var features: [LanguageImageFeatures] = []
        for (index, grid) in grids.enumerated() {
            let image = try gradient(width: grid.width * registration.patchSize, height: grid.height * registration.patchSize, seed: index)
            let processed = try Qwen3VLImagePreprocessor.process(rgb: image.rgb, width: image.width, height: image.height,
                                                                 configuration: configuration)
            features.append(try await encoder.encode(pixels: processed.pixels.map { Float16($0) }, grid: processed.grid))
            sample("encoder, grid \(grid.height)x\(grid.width)")
        }

        // Decoder: text-only first, then images.
        report.stage = "decoder"
        let session = try await LanguageGenerationSession.open(at: textModel, maximumContextLength: manifest.maximumContextLength,
                                                               stopTokenIDs: Set(manifest.stopTokenIDs), memoryGrowthBudgetBytes: nil,
                                                               exportBounds: manifest.exportBounds)
        report.imagesThroughMain = session.imagesThroughMain
        sample("decoder open")
        let short = try tokenizer.render([.user("Name three colours.")]).tokenIDs
        report.promptTokens["main short"] = short.count
        _ = try await session.generate(prompt: short, maximumTokens: 8)
        sample("main, short prompt")
        let long = try tokenizer.render([.user(String(repeating: "The lighthouse keeper paints the railing a calm teal. ", count: 40))]).tokenIDs
        report.promptTokens["main long"] = long.count
        _ = try await session.generate(prompt: long, maximumTokens: 8)
        sample("main, long prompt")
        let rendered = try tokenizer.render([.user("Describe the colours in one line.", images: 1)]).tokenIDs
        for pass in 1...3 {
            for (grid, feature) in zip(grids, features) {
                _ = try await session.generate(prompt: LanguageVisionPrompt(renderedTokenIDs: rendered, images: [feature]),
                                               imageTokens: tokens, maximumTokens: 8)
                report.promptTokens["withImage \(grid.height)x\(grid.width)"] = rendered.count - 1 + feature.tokenCount
                sample("withImage pass \(pass), grid \(grid.height)x\(grid.width)")
            }
        }
        _ = try await session.generate(prompt: long, maximumTokens: 8)
        sample("main, long prompt after images")

        report.stage = "unload"
        await session.unload()
        sample("decoder unloaded")
        await encoder.unload()
        sample("encoder unloaded")
        report.stage = "complete"
    } catch {
        report.error = String(describing: error)
        throw error
    }
}
