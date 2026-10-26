import AetherKit
import CryptoKit
import Foundation

/// H3 boundary storage. BF16 is kept as its original 16 bits, never narrowed to fp16.
public enum H3ScalarType: String, Codable, Sendable {
    case bfloat16, float32, int32
    public var byteWidth: Int { self == .bfloat16 ? 2 : 4 }
}

public struct H3TensorSpec: Sendable, Equatable {
    public static let maximumBytes = 1 << 30
    public let scalarType: H3ScalarType
    public let shape: [Int]
    public let byteCount: Int
    public init(scalarType: H3ScalarType, shape: [Int]) throws {
        guard !shape.isEmpty else { throw AetherError.invalidTensor("H3 boundary requires a non-scalar tensor") }
        var bytes = scalarType.byteWidth
        for dimension in shape {
            let next = bytes.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !next.overflow, next.partialValue <= Self.maximumBytes else {
                throw AetherError.invalidTensor("H3 tensor exceeds the 1 GiB boundary limit or has invalid dimensions")
            }
            bytes = next.partialValue
        }
        self.scalarType = scalarType; self.shape = shape; self.byteCount = bytes
    }
}

/// Owned little-endian bytes, with finite floating-point values checked on entry.
/// Widening BF16 for comparisons preserves the source value exactly.
public struct H3TensorData: Sendable {
    public let spec: H3TensorSpec
    public let bytes: Data
    public init(spec: H3TensorSpec, bytes: Data) throws {
        guard bytes.count == spec.byteCount else { throw AetherError.invalidTensor("H3 tensor byte extent differs from its shape") }
        let finite = bytes.withUnsafeBytes { raw in
            switch spec.scalarType {
            case .bfloat16:
                return stride(from: 0, to: bytes.count, by: 2).allSatisfy {
                    raw.loadUnaligned(fromByteOffset: $0, as: UInt16.self).littleEndian & 0x7f80 != 0x7f80
                }
            case .float32:
                return stride(from: 0, to: bytes.count, by: 4).allSatisfy {
                    Float(bitPattern: raw.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian).isFinite
                }
            case .int32: return true
            }
        }
        guard finite else { throw AetherError.invalidTensor("H3 tensor contains nonfinite values") }
        self.spec = spec; self.bytes = bytes
    }
    public func floatValues() throws -> [Float] {
        guard spec.scalarType != .int32 else { throw AetherError.unsupported("H3 integer tags are not floating-point data") }
        return bytes.withUnsafeBytes { raw in
            stride(from: 0, to: bytes.count, by: spec.scalarType.byteWidth).map { offset in
                let bits: UInt32 = spec.scalarType == .bfloat16
                    ? UInt32(raw.loadUnaligned(fromByteOffset: offset, as: UInt16.self).littleEndian) << 16
                    : raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
                return Float(bitPattern: bits)
            }
        }
    }
    public var sha256: String { h3SHA256(bytes) }
}

public struct H3TensorFile: Sendable {
    public let name: String
    public let url: URL
    public let spec: H3TensorSpec
    public let sha256: String
    public init(name: String, url: URL, spec: H3TensorSpec, sha256: String) throws {
        guard !name.isEmpty, url.isFileURL, h3IsSHA256(sha256) else { throw AetherError.invalidModel("invalid H3 tensor identity") }
        self.name = name; self.url = url.standardizedFileURL; self.spec = spec; self.sha256 = sha256
    }
    /// Explicit CPU read/integrity check. Files must remain immutable during use.
    public func read() throws -> H3TensorData {
        try Task.checkCancellation()
        let count = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard count.isRegularFile == true, count.fileSize == spec.byteCount else { throw AetherError.invalidTensor("H3 tensor file extent differs: \(name)") }
        let data = try Data(contentsOf: url)
        guard h3SHA256(data) == sha256 else { throw AetherError.invalidModel("H3 tensor hash mismatch: \(name)") }
        try Task.checkCancellation()
        return try H3TensorData(spec: spec, bytes: data)
    }
    /// Streams the file directly into caller-owned storage, hashing the destination
    /// bytes as they land. No intermediate copy is allocated. For an identity that
    /// `read()` has already admitted, a matching hash implies the same finite values,
    /// so the element scan is not repeated. On failure the destination is partial.
    public func read(into destination: UnsafeMutableRawBufferPointer) throws {
        try Task.checkCancellation()
        let count = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard count.isRegularFile == true, count.fileSize == spec.byteCount, destination.count == spec.byteCount,
              let base = destination.baseAddress else { throw AetherError.invalidTensor("H3 tensor file extent differs: \(name)") }
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw AetherError.invalidModel("H3 tensor file could not be opened: \(name)") }
        defer { close(descriptor) }
        var hasher = SHA256(), offset = 0
        while offset < spec.byteCount {
            try Task.checkCancellation()
            let chunk = Darwin.read(descriptor, base + offset, min(8 << 20, spec.byteCount - offset))
            if chunk < 0 && errno == EINTR { continue }
            guard chunk > 0 else { throw AetherError.invalidModel("H3 tensor file read failed: \(name)") }
            hasher.update(bufferPointer: UnsafeRawBufferPointer(start: base + offset, count: chunk))
            offset += chunk
        }
        guard hasher.finalize().map({ String(format: "%02x", $0) }).joined() == sha256 else {
            throw AetherError.invalidModel("H3 tensor hash mismatch: \(name)")
        }
        try Task.checkCancellation()
    }
}

public func h3SHA256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
func h3IsSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}
