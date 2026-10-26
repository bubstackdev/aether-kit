import AetherH3
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

@Test func h3StreamedReadFillsDestinationAndRejectsCorruptionOrExtent() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    // Two complete 8 MiB chunks and a partial tail of finite BF16 values.
    var raw = Data(repeating: 0x3f, count: 16 << 20)
    raw.append(contentsOf: [0x80, 0x3f, 0x00, 0x40])
    let url = root.appendingPathComponent("weights.raw")
    try raw.write(to: url)
    let file = try H3TensorFile(name: "weight", url: url, spec: H3TensorSpec(scalarType: .bfloat16, shape: [raw.count / 2]), sha256: h3SHA256(raw))
    let destination = UnsafeMutableRawBufferPointer.allocate(byteCount: raw.count, alignment: 16)
    defer { destination.deallocate() }
    try file.read(into: destination)
    #expect(Data(destination) == raw)
    let short = UnsafeMutableRawBufferPointer.allocate(byteCount: raw.count - 2, alignment: 16)
    defer { short.deallocate() }
    #expect(throws: AetherError.self) { try file.read(into: short) }
    var corrupt = raw
    corrupt[(8 << 20) + 5] ^= 1 // Second chunk.
    try corrupt.write(to: url)
    #expect(throws: AetherError.invalidModel("H3 tensor hash mismatch: weight")) { try file.read(into: destination) }
    try raw.prefix(raw.count - 2).write(to: url)
    #expect(throws: AetherError.self) { try file.read(into: destination) }
}
