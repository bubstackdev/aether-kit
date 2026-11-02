import AetherKit
import AetherAssets
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
    private var lease: ExternalFileLease? = nil
    /// The first segment's file (the whole file for a single-file tensor).
    public var url: URL { segments[0].url }
    public init(name: String, url: URL, spec: H3TensorSpec, sha256: String) throws {
        guard !name.isEmpty, url.isFileURL, h3IsSHA256(sha256) else { throw AetherError.invalidModel("invalid H3 tensor identity") }
        self.name = name; self.segments = [try Segment(url: url, offset: 0, length: spec.byteCount)]; self.spec = spec
        self.sha256 = sha256; self.wholeFile = true
    }
    public init(name: String, segments: [Segment], spec: H3TensorSpec, sha256: String) throws {
        var total = 0
        for segment in segments {
            let sum = total.addingReportingOverflow(segment.length)
            guard !sum.overflow, sum.partialValue <= spec.byteCount else { throw AetherError.invalidModel("invalid H3 tensor extent") }
            total = sum.partialValue
        }
        guard !name.isEmpty, !segments.isEmpty, h3IsSHA256(sha256),
              total == spec.byteCount else { throw AetherError.invalidModel("invalid H3 tensor identity") }
        self.name = name; self.segments = segments; self.spec = spec; self.sha256 = sha256; self.wholeFile = false
    }
    /// Uses pinned descriptors for all future reads. Hash/precision contracts remain unchanged.
    public func holding(_ lease: ExternalFileLease) throws -> Self {
        for segment in segments {
            let size = try lease.byteCount(of: segment.url)
            guard UInt64(segment.offset) <= size, UInt64(segment.length) <= size - UInt64(segment.offset),
                  !wholeFile || size == UInt64(spec.byteCount) else {
                throw AetherError.invalidTensor("H3 leased tensor extent differs: " + name)
            }
        }
        var result = self; result.lease = lease; return result
    }

    /// Explicit CPU read/integrity check. Files must remain immutable during use.
    public func read() throws -> H3TensorData {
        try Task.checkCancellation()
        let data: Data
        if lease != nil {
            var bytes = Data(count: spec.byteCount)
            try bytes.withUnsafeMutableBytes { try read(into: $0, verify: true) }
            return try H3TensorData(spec: spec, bytes: bytes)
        } else if wholeFile {
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
    /// is set, unless the tensor was verified before with its files' current identities (``VerifiedFileHashes``).
    /// Native weight slots use this to avoid an intermediate copy; finite values are not re-checked here.
    ///
    /// The identities come from `fstat` on the descriptors the bytes are read from (the lease's pinned ones when
    /// leased), so a path replaced meanwhile can't lend its identity to other bytes. A cached read checks them again
    /// after copying; if anything changed, it reads again and hashes, and trusts nothing remembered.
    public func read(into destination: UnsafeMutableRawBufferPointer, verify: Bool) throws {
        try Task.checkCancellation()
        guard destination.count == spec.byteCount, destination.baseAddress != nil else {
            throw AetherError.invalidTensor("H3 tensor destination extent differs: \(name)")
        }
        let source = try Source(self)
        defer { source.close() }
        guard verify, let claim = VerifiedFileHashes.claim(claimKey, identities: source.identities()) else {
            return try copy(from: source, into: destination, verify: verify)
        }
        if VerifiedFileHashes.isVerified(claim) {
            try copy(from: source, into: destination, verify: false)
            VerifiedFileHashes.hooks.afterCachedRead?()
            if claim.holds(source.identities()) { return }
            return try copy(from: source, into: destination, verify: true)
        }
        try copy(from: source, into: destination, verify: true)
        VerifiedFileHashes.recordVerified(claim, current: source.identities())
    }

    /// The verified-hash claim key: the digest, then every segment's file, offset and length in order.
    var claimKey: String {
        (["h3-tensor", sha256] + segments.flatMap { [$0.url.path, String($0.offset), String($0.length)] })
            .joined(separator: "\u{0}")
    }

    /// The open descriptors a read uses, one per distinct file in first-use order: the lease's pinned ones, or files
    /// opened for this read (closed by `close()`).
    private struct Source {
        let files: [URL]
        let descriptors: [URL: Int32]
        let lease: ExternalFileLease?

        init(_ tensor: H3TensorFile) throws {
            var files: [URL] = [], descriptors: [URL: Int32] = [:]
            lease = tensor.lease
            do {
                for segment in tensor.segments where descriptors[segment.url] == nil {
                    if let lease = tensor.lease {
                        descriptors[segment.url] = try lease.descriptor(of: segment.url)
                    } else {
                        let fd = open(segment.url.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
                        guard fd >= 0 else { throw AetherError.invalidTensor("H3 tensor file cannot be opened: \(tensor.name)") }
                        descriptors[segment.url] = fd
                    }
                    files.append(segment.url)
                }
                if tensor.lease == nil {
                    for segment in tensor.segments {
                        var info = stat()
                        guard fstat(descriptors[segment.url]!, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
                              segment.offset + segment.length <= Int(info.st_size),
                              !tensor.wholeFile || Int(info.st_size) == tensor.spec.byteCount else {
                            throw AetherError.invalidTensor("H3 tensor segment lies outside its file: \(tensor.name)")
                        }
                    }
                }
            } catch {
                if tensor.lease == nil { for fd in descriptors.values { Darwin.close(fd) } }
                throw error
            }
            self.files = files; self.descriptors = descriptors
        }

        func identities() -> [VerifiedFileHashes.Identity?] {
            files.map { VerifiedFileHashes.Identity(descriptor: descriptors[$0]!) }
        }

        func close() { if lease == nil { for fd in descriptors.values { Darwin.close(fd) } } }
    }

    private func copy(from source: Source, into destination: UnsafeMutableRawBufferPointer, verify: Bool) throws {
        try Task.checkCancellation()
        guard let base = destination.baseAddress else { throw AetherError.invalidTensor("H3 tensor destination extent differs: \(name)") }
        var hasher = SHA256(), written = 0
        for segment in segments {
            let buffer = UnsafeMutableRawBufferPointer(start: base + written, count: segment.length)
            if let lease = source.lease {
                try lease.read(into: buffer, from: segment.url, offset: UInt64(segment.offset))
            } else {
                let fd = source.descriptors[segment.url]!
                var done = 0
                while done < segment.length {
                    let count = pread(fd, base + written + done, min(segment.length - done, 64 << 20), off_t(segment.offset + done))
                    if count < 0, errno == EINTR { continue }
                    guard count > 0 else { throw AetherError.invalidTensor("H3 tensor segment is short: \(name)") }
                    done += count
                }
            }
            if verify { hasher.update(bufferPointer: UnsafeRawBufferPointer(buffer)) }
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
