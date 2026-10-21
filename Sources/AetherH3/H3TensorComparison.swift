import AetherKit
import Foundation

/// Complete-tensor numerical comparison without allocating widened full tensors.
public struct H3TensorComparison: Codable, Sendable {
    public let passed: Bool
    public let relativeL2: Double?
    public let maximumAbsoluteError: Double
    public let actualSHA256: String
    public let referenceSHA256: String
    public let byteIdentical: Bool
    public init(actual: H3TensorData, reference: H3TensorData, limit: Double = 0.02) throws {
        guard actual.spec == reference.spec, actual.spec.scalarType != .int32, limit.isFinite, limit > 0 else {
            throw AetherError.invalidTensor("H3 full numerical comparison requires matching floating tensors and a positive finite limit")
        }
        let width = actual.spec.scalarType.byteWidth
        let sums = actual.bytes.withUnsafeBytes { a in
            reference.bytes.withUnsafeBytes { b in
                var error = 0.0, norm = 0.0, maximum = 0.0
                func value(_ raw: UnsafeRawBufferPointer, _ i: Int) -> Double {
                    let bits = width == 2 ? UInt32(raw.loadUnaligned(fromByteOffset: i, as: UInt16.self).littleEndian) << 16
                        : raw.loadUnaligned(fromByteOffset: i, as: UInt32.self).littleEndian
                    return Double(Float(bitPattern: bits))
                }
                for i in stride(from: 0, to: a.count, by: width) {
                    let x = value(a, i), y = value(b, i), difference = x - y
                    error += difference * difference; norm += y * y; maximum = max(maximum, abs(difference))
                }
                return (error, norm, maximum)
            }
        }
        let relative = sums.1 > 0 ? (sums.0 / sums.1).squareRoot() : (sums.0 == 0 ? 0 : nil)
        relativeL2 = relative; maximumAbsoluteError = sums.2
        passed = relative.map { $0 <= limit } ?? false
        actualSHA256 = actual.sha256; referenceSHA256 = reference.sha256
        byteIdentical = actual.bytes == reference.bytes
    }
}
