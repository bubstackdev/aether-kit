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
    /// A byte range of an immutable file. A tensor is its segments concatenated in order, so weights can be read in
    /// place from the pinned checkpoint shards (e.g. one head-strided slice of a fused QKV) instead of copied.
    public struct Segment: Sendable, Equatable {
        public let url: URL
        public let offset: Int
        public let length: Int
        public init(url: URL, offset: Int, length: Int) throws {
            guard url.isFileURL, offset >= 0, length > 0, !offset.addingReportingOverflow(length).overflow else {
                throw AetherError.invalidModel("invalid H3 tensor segment")
            }
            self.url = url.standardizedFileURL; self.offset = offset; self.length = length
        }
    }
    public let name: String
    public let segments: [Segment]
    public let spec: H3TensorSpec
    public let sha256: String
    /// True when the tensor is a whole file of exactly its byte extent.
    public let wholeFile: Bool
    /// The first segment's file (the whole file for a single-file tensor).
    public var url: URL { segments[0].url }
    public init(name: String, url: URL, spec: H3TensorSpec, sha256: String) throws {
        guard !name.isEmpty, url.isFileURL, h3IsSHA256(sha256) else { throw AetherError.invalidModel("invalid H3 tensor identity") }
        self.name = name; self.segments = [try Segment(url: url, offset: 0, length: spec.byteCount)]; self.spec = spec
        self.sha256 = sha256; self.wholeFile = true
    }
    public init(name: String, segments: [Segment], spec: H3TensorSpec, sha256: String) throws {
        guard !name.isEmpty, !segments.isEmpty, h3IsSHA256(sha256),
              segments.reduce(0, { $0 + $1.length }) == spec.byteCount else { throw AetherError.invalidModel("invalid H3 tensor identity") }
        self.name = name; self.segments = segments; self.spec = spec; self.sha256 = sha256; self.wholeFile = false
    }
    /// Explicit CPU read/integrity check. Files must remain immutable during use.
    public func read() throws -> H3TensorData {
        try Task.checkCancellation()
        let data: Data
        if wholeFile {
            let count = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            guard count.isRegularFile == true, count.fileSize == spec.byteCount else { throw AetherError.invalidTensor("H3 tensor file extent differs: \(name)") }
            data = try Data(contentsOf: url)
        } else {
            var bytes = Data(capacity: spec.byteCount)
            var handles: [URL: FileHandle] = [:]
            defer { for handle in handles.values { try? handle.close() } }
            for segment in segments {
                let handle: FileHandle
                if let open = handles[segment.url] { handle = open } else {
                    let values = try segment.url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
                    guard values.isRegularFile == true, let size = values.fileSize, segment.offset + segment.length <= size else {
                        throw AetherError.invalidTensor("H3 tensor segment lies outside its file: \(name)")
                    }
                    handle = try FileHandle(forReadingFrom: segment.url); handles[segment.url] = handle
                }
                try handle.seek(toOffset: UInt64(segment.offset))
                guard let chunk = try handle.read(upToCount: segment.length), chunk.count == segment.length else {
                    throw AetherError.invalidTensor("H3 tensor segment is short: \(name)")
                }
                bytes.append(chunk)
            }
            data = bytes
        }
        guard h3SHA256(data) == sha256 else { throw AetherError.invalidModel("H3 tensor hash mismatch: \(name)") }
        try Task.checkCancellation()
        return try H3TensorData(spec: spec, bytes: data)
    }
    /// Reads the bytes straight into `destination` (exactly `spec.byteCount`), hashing them on the way when `verify`
    /// is set. Native weight slots use this to avoid an intermediate copy; finite values are not re-checked here.
    public func read(into destination: UnsafeMutableRawBufferPointer, verify: Bool) throws {
        try Task.checkCancellation()
        guard destination.count == spec.byteCount, let base = destination.baseAddress else {
            throw AetherError.invalidTensor("H3 tensor destination extent differs: \(name)")
        }
        var hasher = SHA256(), written = 0
        var descriptors: [URL: Int32] = [:]
        defer { for fd in descriptors.values { close(fd) } }
        for segment in segments {
            let fd: Int32
            if let open = descriptors[segment.url] { fd = open } else {
                let values = try segment.url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
                guard values.isRegularFile == true, let size = values.fileSize, segment.offset + segment.length <= size,
                      !wholeFile || size == spec.byteCount else {
                    throw AetherError.invalidTensor("H3 tensor segment lies outside its file: \(name)")
                }
                fd = open(segment.url.path, O_RDONLY)
                guard fd >= 0 else { throw AetherError.invalidTensor("H3 tensor file cannot be opened: \(name)") }
                descriptors[segment.url] = fd
            }
            var done = 0
            while done < segment.length {
                let count = pread(fd, base + written + done, min(segment.length - done, 64 << 20), off_t(segment.offset + done))
                guard count > 0 else { throw AetherError.invalidTensor("H3 tensor segment is short: \(name)") }
                done += count
            }
            if verify { hasher.update(bufferPointer: UnsafeRawBufferPointer(start: base + written, count: segment.length)) }
            written += segment.length
            try Task.checkCancellation()
        }
        if verify, hasher.finalize().map({ String(format: "%02x", $0) }).joined() != sha256 {
            throw AetherError.invalidModel("H3 tensor hash mismatch: \(name)")
        }
    }
}

public func h3SHA256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
func h3IsSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}
