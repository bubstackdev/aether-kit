import AetherH3
import AetherAssets
import AetherKit
import Foundation
import Testing

@Test func h3BF16PreservesBitsAndExponentRangeWithoutFP16Narrowing() throws {
    let bits: [UInt16] = [0, 0x8000, 0x3f80, 0x3f81, 0x0001, 0x7f7f, 0xff7f]
    let data = bits.withUnsafeBytes { Data($0) }
    let tensor = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [bits.count]), bytes: data)
    #expect(tensor.bytes == data)
    #expect(try tensor.floatValues().map(\.bitPattern) == bits.map { UInt32($0) << 16 })
    #expect(try tensor.floatValues()[5] > Float(Float16.greatestFiniteMagnitude))
}

@Test func h3TensorRejectsInvalidExtentsOverflowAndNonfiniteBits() throws {
    for shape in [[], [0], [-1], [Int.max], [1 << 30]] {
        #expect(throws: AetherError.self) { try H3TensorSpec(scalarType: .float32, shape: shape) }
    }
    let bf = try H3TensorSpec(scalarType: .bfloat16, shape: [1])
    for bits: UInt16 in [0x7f80, 0xff80, 0x7fc0, 0x7f81] {
        #expect(throws: AetherError.self) { try H3TensorData(spec: bf, bytes: withUnsafeBytes(of: bits) { Data($0) }) }
    }
    #expect(throws: AetherError.self) { try H3TensorData(spec: bf, bytes: Data([0])) }
    let f32 = try H3TensorSpec(scalarType: .float32, shape: [1])
    #expect(throws: AetherError.self) { try H3TensorData(spec: f32, bytes: withUnsafeBytes(of: Float.nan) { Data($0) }) }
    let tags = try H3TensorData(spec: H3TensorSpec(scalarType: .int32, shape: [1]), bytes: Data([255,255,255,255]))
    #expect(throws: AetherError.self) { try tags.floatValues() }
}

@Test func h3FileVerificationRejectsCorruptionAndKeepsOwnedBytes() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("weights.raw"), raw = Data([128,63])
    try raw.write(to: url)
    let file = try H3TensorFile(name: "weight", url: url, spec: H3TensorSpec(scalarType: .bfloat16, shape: [1]), sha256: h3SHA256(raw))
    let retained = try file.read()
    try Data([0,0]).write(to: url)
    #expect(throws: AetherError.self) { try file.read() }
    #expect(retained.bytes == raw)
    try Data([0]).write(to: url)
    #expect(throws: AetherError.self) { try file.read() }
    #expect(throws: AetherError.self) { try H3TensorFile(name: "bad", url: url, spec: file.spec, sha256: "no") }
}

@Test func h3ModelStreamVerificationChecksBothDeclarationAndBytes() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    var raw = Data(repeating:0x5a,count:16 << 20)
    raw.append(contentsOf:0..<17) // Exercise two full chunks and a partial tail.
    let digest = h3SHA256(raw)
    try raw.write(to: root.appendingPathComponent("main.mlirb"))
    let bytes = stride(from: 0, to: 64, by: 2).map { i -> UInt8 in
        let start = digest.index(digest.startIndex, offsetBy: i)
        return UInt8(digest[start..<digest.index(start, offsetBy: 2)], radix: 16)!
    }
    try Data(bytes).write(to: root.appendingPathComponent("main.hash"))
    let model = try H3ModelAsset(url: root, function: "main", sha256: digest)
    try model.verify()
    let writer = try FileHandle(forWritingTo:root.appendingPathComponent("main.mlirb"))
    try writer.seek(toOffset:UInt64((8 << 20)+5));try writer.write(contentsOf:Data([0xff]));try writer.close()
    #expect(throws: AetherError.self) { try model.verify() }
    try raw.write(to: root.appendingPathComponent("main.mlirb"))
    try Data(repeating: 0, count: 32).write(to: root.appendingPathComponent("main.hash"))
    #expect(throws: AetherError.self) { try model.verify() }
}

@Test func h3SegmentedFilesConcatenateRangesInOrderAndFailClosed() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let a = root.appendingPathComponent("a.safetensors"), b = root.appendingPathComponent("b.safetensors")
    try Data([9, 9, 1, 0, 2, 0, 9]).write(to: a); try Data([3, 0, 4, 0]).write(to: b)
    let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [4])
    // Out of order across files, as a head-strided QKV slice or a swapped SwiGLU half would be.
    let expected = Data([3, 0, 1, 0, 2, 0, 4, 0])
    let segments = [try H3TensorFile.Segment(url: b, offset: 0, length: 2), try .init(url: a, offset: 2, length: 4),
                    try .init(url: b, offset: 2, length: 2)]
    let file = try H3TensorFile(name: "w0", segments: segments, spec: spec, sha256: h3SHA256(expected))
    #expect(!file.wholeFile)
    #expect(try file.read().bytes == expected)
    #expect(throws: AetherError.self) { try H3TensorFile(name: "w0", segments: Array(segments.dropLast()), spec: spec, sha256: h3SHA256(expected)) }
    #expect(throws: AetherError.self) { try H3TensorFile.Segment(url: a, offset: -1, length: 2) }
    #expect(throws: AetherError.self) { try H3TensorFile.Segment(url: a, offset: 0, length: 0) }
    let outside = try H3TensorFile(name: "w1", segments: [try .init(url: a, offset: 4, length: 8)], spec: spec, sha256: h3SHA256(expected))
    #expect(throws: AetherError.self) { try outside.read() }
    try Data([9, 9, 1, 0, 2, 1, 9]).write(to: a)
    #expect(throws: AetherError.self) { try file.read() }
}


@Test func h3LeasedSegmentsKeepOrderAndEmbeddingRowsUseOriginalDescriptors() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let shard = root.appendingPathComponent("shard")
    let bytes = Data([128,63, 0,64, 64,64, 128,64]) // BF16 1, 2, 3, 4.
    try bytes.write(to: shard)
    let lease = try ExternalFileLease(files: [shard])
    let expected = Data([64,64, 128,63, 0,64])
    let file = try H3TensorFile(name: "qkv", segments: [.init(url: shard, offset: 4, length: 2), .init(url: shard, offset: 0, length: 4)],
                               spec: H3TensorSpec(scalarType: .bfloat16, shape: [3]), sha256: h3SHA256(expected)).holding(lease)
    let embedding = try H3EmbeddingTable(url: shard, offset: 0, rows: 2, width: 2, byteCount: 8, sha256: h3SHA256(bytes)).holding(lease)
    try FileManager.default.moveItem(at: shard, to: root.appendingPathComponent("old"))
    try Data(repeating: 0, count: 8).write(to: shard)
    #expect(try file.read().floatValues() == [3, 1, 2])
    var destination = Data(count: 6)
    try destination.withUnsafeMutableBytes { try file.read(into: $0, verify: true) }
    #expect(destination == expected)
    try embedding.verify()
    #expect(try embedding.rows([1, 0]).floatValues() == [3, 4, 1, 2])
}

@Test func h3LeasedTensorRejectsCoverageExtentAndOverflowBeforeAllocating() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let shard = root.appendingPathComponent("shard")
    try Data([0,0]).write(to: shard)
    let lease = try ExternalFileLease(files: [shard])
    let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [1])
    let file = try H3TensorFile(name: "w", segments: [.init(url: shard, offset: 1, length: 2)], spec: spec, sha256: h3SHA256(Data([0,0])))
    #expect(throws: AetherError.self) { try file.holding(lease) }
    #expect(throws: AetherError.self) {
        try H3TensorFile(name: "overflow", segments: [.init(url: shard, offset: 0, length: Int.max), .init(url: shard, offset: 0, length: Int.max)],
                         spec: spec, sha256: h3SHA256(Data([0,0])))
    }
    #expect(throws: AetherError.self) { try H3EmbeddingTable(url: shard, offset: 0, rows: 1, width: Int.max, byteCount: 2, sha256: h3SHA256(Data([0,0]))) }
}
