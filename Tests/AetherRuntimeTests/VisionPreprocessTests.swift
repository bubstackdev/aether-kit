import AetherCore
import CoreGraphics
import Foundation
import Testing
@testable import AetherRuntime

/// `preprocess.image` (MANIFEST.md §2.5) against Pillow and torch, from `Fixtures/vision/preprocess.json`
/// (`Forge/vision/fixtures_preprocess.py`): the parity filters bit for bit, the `vImage` fast filter within its pinned
/// distance of Pillow's Lanczos, and whole preprocess blocks (geometry, normalization, channel order, layouts, host
/// dtype) to the bit.
@Suite("Vision preprocessing")
struct VisionPreprocessTests {
    struct Fixture: Decodable {
        struct Resize: Decodable {
            var width: Int, height: Int, targetWidth: Int, targetHeight: Int
            var input: String, pillowBicubic: String, torchUint8Bicubic: String
        }
        struct VImage: Decodable {
            var width: Int, height: Int, targetWidth: Int, targetHeight: Int, input: String, pillowLanczos: String
        }
        struct Pipeline: Decodable {
            var name: String, width: Int, height: Int, input: String
            var preprocess: Manifest.ImagePreprocess
            var canvasWidth: Int, canvasHeight: Int, shape: [Int], hostDType: String, expected: String
        }
        var resizes: [Resize]
        var vImage: [VImage]
        var pipelines: [Pipeline]

        static func load() throws -> Fixture {
            let url = try #require(Bundle.module.url(forResource: "preprocess", withExtension: "json", subdirectory: "Fixtures/vision"))
            return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        }
    }

    static func bytes(_ base64: String) throws -> [UInt8] { [UInt8](try #require(Data(base64Encoded: base64))) }

    static func image(_ base64: String, _ width: Int, _ height: Int) throws -> RGB8Image {
        try RGB8Image(rgb: bytes(base64), width: width, height: height)
    }

    @Test func parityBicubicsMatchPillowAndTorchBitForBit() throws {
        let fixture = try Fixture.load()
        #expect(fixture.resizes.count >= 10)
        for r in fixture.resizes {
            let input = try Self.image(r.input, r.width, r.height)
            let pillow = ImageResampler.bicubic(input, width: r.targetWidth, height: r.targetHeight, reference: .pillow)
            let torch = ImageResampler.bicubic(input, width: r.targetWidth, height: r.targetHeight, reference: .torchUint8)
            let label = "\(r.width)x\(r.height) -> \(r.targetWidth)x\(r.targetHeight)"
            #expect(pillow.rgb == (try Self.bytes(r.pillowBicubic)), "Pillow \(label)")
            #expect(torch.rgb == (try Self.bytes(r.torchUint8Bicubic)), "torch \(label)")
        }
    }

    /// `vImageScale`'s kernel is its own (Lanczos-like), not any of Pillow's, so ±1 holds only for most values: against
    /// Pillow's `LANCZOS`, the nearest, on smooth images (macOS 27.2, M5 Max) it is at most 2 levels away and within
    /// one level for 99.58–99.96 % of values per case. On noise it is much further (tens of levels), which is why it is
    /// never a parity filter. These bars pin the measurement, so a change in Accelerate shows here.
    @Test func vImageStaysNearPillowLanczosOnSmoothImages() throws {
        var within = 0, total = 0
        for r in try Fixture.load().vImage {
            let input = try Self.image(r.input, r.width, r.height)
            let out = try ImageResampler.vImage(input, width: r.targetWidth, height: r.targetHeight)
            let reference = try Self.bytes(r.pillowLanczos)
            let differences = zip(out.rgb, reference).map { abs(Int($0) - Int($1)) }
            let close = differences.filter { $0 <= 1 }.count
            let label = "\(r.width)x\(r.height) -> \(r.targetWidth)x\(r.targetHeight)"
            #expect(differences.max()! <= 2, "\(label): max \(differences.max()!)")
            #expect(Double(close) / Double(differences.count) >= 0.995, "\(label): \(close) of \(differences.count) within 1")
            within += close
            total += differences.count
        }
        #expect(Double(within) / Double(total) >= 0.998, "\(within) of \(total) within 1")
    }

    @Test func preprocessBlocksMatchTorchToTheBit() throws {
        let fixture = try Fixture.load()
        #expect(fixture.pipelines.count >= 9)
        for p in fixture.pipelines {
            let preprocessor = try ImagePreprocessor(p.preprocess)
            let prepared = try preprocessor.prepare(try Self.image(p.input, p.width, p.height))
            #expect(prepared.shape == p.shape, "\(p.name)")
            #expect(prepared.geometry.width == p.canvasWidth && prepared.geometry.height == p.canvasHeight, "\(p.name)")
            #expect(prepared.scalarType.rawValue == p.hostDType, "\(p.name)")
            let expected = try Self.bytes(p.expected)
            let actual: [UInt8] = switch prepared.values {
            case .float32(let v): v.withUnsafeBytes { [UInt8]($0) }
            case .float16(let v): v.withUnsafeBytes { [UInt8]($0) }
            }
            #expect(actual.count == expected.count, "\(p.name)")
            let width = p.hostDType == "float16" ? 2 : 4
            let mismatched = stride(from: 0, to: min(actual.count, expected.count), by: width)
                .filter { actual[$0..<($0 + width)] != expected[$0..<($0 + width)] }.count
            #expect(mismatched == 0, "\(p.name): \(mismatched) values differ from torch's")
        }
    }

    // MARK: - Geometry and layout rules

    static func gradient(_ width: Int, _ height: Int) throws -> RGB8Image {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height { for x in 0..<width { rgb[(y * width + x) * 3] = UInt8(y); rgb[(y * width + x) * 3 + 1] = UInt8(x) } }
        return try RGB8Image(rgb: rgb, width: width, height: height)
    }

    @Test func geometryPlacesTheSourceOnTheCanvas() throws {
        func geometry(_ resize: Manifest.ImagePreprocess.Resize, _ w: Int, _ h: Int) throws -> ImageGeometry {
            try ImagePreprocessor(.init(resize: resize)).prepare(try Self.gradient(w, h)).geometry
        }
        let size = Manifest.ImagePreprocess.Size(width: 64, height: 64)
        // YOLOX: top-left, the long side fills; the short side is floored.
        let yolox = try geometry(.init(mode: .letterbox, size: size, anchor: .topLeft, padValue: [114], filter: .pillowBicubic), 100, 50)
        #expect(yolox.content == CGRect(x: 0, y: 0, width: 64, height: 32))
        #expect(yolox.normalizedSourceRect(fromPrepared: CGRect(x: 32, y: 16, width: 32, height: 16))
                == CGRect(x: 0.5, y: 0.5, width: 0.5, height: 0.5))
        let centered = try geometry(.init(mode: .letterbox, size: size, filter: .pillowBicubic), 50, 99)
        #expect(centered.content == CGRect(x: 16, y: 0, width: 32, height: 64))
        // Centre crop: covers, then the excess split (Python's round, ties to even).
        let crop = try geometry(.init(mode: .centerCrop, size: size, filter: .pillowBicubic), 101, 50)
        #expect(crop.content == CGRect(x: -32, y: 0, width: 129, height: 64))
        let crop2 = try geometry(.init(mode: .centerCrop, size: size, filter: .pillowBicubic), 103, 50)
        #expect(crop2.content == CGRect(x: -34, y: 0, width: 131, height: 64))
        // torchvision's Resize(n) + CenterCrop: the shorter side to shortestSide first (the longer floored).
        let shortest = try geometry(.init(mode: .centerCrop, size: .init(width: 32, height: 32), shortestSide: 40, filter: .pillowBicubic), 101, 50)
        #expect(shortest.content == CGRect(x: -24, y: -4, width: 80, height: 40))
        let square = try geometry(.init(mode: .centerCrop, size: size, shortestSide: 64, filter: .pillowBicubic), 101, 50)
        #expect(square == crop)
        // Fit the long side only when it is too long.
        let fit = try geometry(.init(mode: .fitLongSide, maxSide: 40, filter: .vImage), 30, 100)
        #expect(fit.width == 12 && fit.height == 40)
        let small = try geometry(.init(mode: .fitLongSide, maxSide: 400, filter: .vImage), 30, 100)
        #expect(small.width == 30 && small.height == 100 && small.content == CGRect(x: 0, y: 0, width: 30, height: 100))
    }

    @Test func letterboxPadsWithThePadValueAndPatchesNeedWholeBlocks() throws {
        var block = Manifest.ImagePreprocess(resize: .init(mode: .letterbox, size: .init(width: 4, height: 4), anchor: .topLeft,
                                                          padValue: [10, 20, 30], filter: .pillowBicubic))
        let prepared = try ImagePreprocessor(block).prepare(try RGB8Image(rgb: [UInt8](repeating: 200, count: 4 * 2 * 3), width: 4, height: 2))
        let values = prepared.floats()
        // Rows 0-1 are the image, rows 2-3 the pad, per channel.
        #expect(Array(values[0..<8]) == [Float](repeating: 200, count: 8))
        #expect(Array(values[8..<16]) == [Float](repeating: 10, count: 8))
        #expect(Array(values[(16 + 8)..<32]) == [Float](repeating: 20, count: 8))
        #expect(Array(values[(32 + 8)..<48]) == [Float](repeating: 30, count: 8))

        block = .init(layout: .patches, patchify: .init(patch: 16, temporal: 2, merge: 2, rowOrder: .blockMajor, withinPatch: .channelFramePixel))
        #expect(throws: AetherError.self) { try ImagePreprocessor(block).prepare(try Self.gradient(48, 32)) }
        // 0.0.x's merge-window check on a coordinate-coded 96x64 image: red = row, green = column.
        block.scale = 1.0 / 255
        block.mean = [0.5, 0.5, 0.5]
        block.std = [0.5, 0.5, 0.5]
        let patches = try ImagePreprocessor(block).prepare(try Self.gradient(96, 64))
        #expect(patches.shape == [24, 1536])
        // Row 5: window 1 (columns 2-3), patch (0, 1) -> grid (0, 3): pixel rows 0..15, columns 48..63.
        let row = Array(patches.floats()[(5 * 1536)..<(6 * 1536)])
        let level = { (v: Float) -> Int in Int((v * 0.5 + 0.5) * 255 + 0.5) }
        #expect(level(row[0]) == 0 && level(row[16 * 15]) == 15)                 // red, frame 0: rows
        #expect(level(row[2 * 256]) == 48 && level(row[2 * 256 + 15]) == 63)     // green, frame 0: columns
        #expect(Array(row[0..<256]) == Array(row[256..<512]))                      // frames repeat
        #expect(abs(row[4 * 256] + 1) < 1e-6)                                      // blue 0 -> -1
    }

    @Test func invalidBlocksNameTheirField() {
        func field(_ block: Manifest.ImagePreprocess) -> String? {
            do { _ = try ImagePreprocessor(block); return nil } catch { return AetherRuntimeTests.field(of: error) }
        }
        let size = Manifest.ImagePreprocess.Size(width: 8, height: 8)
        #expect(field(.init(resize: .init(mode: .stretch, filter: .vImage))) == "preprocess.image.resize.size")
        #expect(field(.init(resize: .init(mode: .fitLongSide, size: size, filter: .vImage))) == "preprocess.image.resize.maxSide")
        #expect(field(.init(resize: .init(mode: .stretch, size: size, padValue: [1], filter: .vImage))) == "preprocess.image.resize.anchor")
        #expect(field(.init(resize: .init(mode: .letterbox, size: size, padValue: [1, 2], filter: .vImage))) == "preprocess.image.resize.padValue")
        #expect(field(.init(std: [1, 0, 1])) == "preprocess.image.std")
        #expect(field(.init(mean: [0, 0])) == "preprocess.image.mean")
        #expect(field(.init(views: 2)) == "preprocess.image.views")
        #expect(field(.init(layout: .patches)) == "preprocess.image.patchify")
        #expect(field(.init(patchify: .init(patch: 4, rowOrder: .rowMajor, withinPatch: .pixelChannel))) == "preprocess.image.patchify")
        #expect(field(.init(layout: .patches, patchify: .init(patch: 4, temporal: 2, rowOrder: .rowMajor, withinPatch: .pixelChannel)))
                == "preprocess.image.patchify.temporal")
        #expect(field(.init(hostDType: .int32)) == "preprocess.image.hostDType")
    }

    /// The graph input's own type decides what is written: an fp32 host feeds an fp16 graph (DA3), rounded to nearest
    /// even, and a shape other than the graph's is refused.
    @Test func bindingConvertsToTheGraphsTypeAndChecksTheShape() throws {
        final class Fake: GraphBackend {
            let signature: GraphSignature
            init(_ type: ScalarType, _ shape: [Int]) {
                signature = GraphSignature(function: "main", inputs: [TensorSpec(name: "image", scalarType: type, shape: shape)],
                                           states: [], outputs: [])
            }
            func run(_ bindings: inout Bindings) async throws(AetherError) {}
        }
        let block = Manifest.ImagePreprocess(resize: .init(mode: .stretch, size: .init(width: 4, height: 2), filter: .pillowBicubic),
                                             scale: 1.0 / 3)
        let prepared = try ImagePreprocessor(block).prepare(try Self.gradient(4, 2))
        let arena = TensorArena()
        let half = try Fake(.float16, [1, 3, 2, 4]).makeBuffer(for: "image", holding: prepared, in: arena)
        #expect(try half.read(Float16.self) == prepared.floats().map { Float16($0) })
        let single = try Fake(.float32, [1, 3, -1, -1]).makeBuffer(for: "image", holding: prepared, in: arena)
        #expect(single.shape == [1, 3, 2, 4])
        #expect(try single.readFloats() == prepared.floats())
        #expect(throws: AetherError.self) { try Fake(.float32, [1, 3, 4, 4]).makeBuffer(for: "image", holding: prepared, in: arena) }
        #expect(throws: AetherError.self) { try Fake(.int32, [1, 3, 2, 4]).makeBuffer(for: "image", holding: prepared, in: arena) }
    }
}
