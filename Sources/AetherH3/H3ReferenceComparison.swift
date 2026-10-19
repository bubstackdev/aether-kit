import AetherKit
import Foundation

public struct H3ReferenceComparison: Codable, Sendable {
    public let passed: Bool
    public let relativeL2: Double?
    public let maximumAbsoluteError: Double
    public let actualSHA256: String
    public let referenceSHA256: String
    public let byteIdentical: Bool

    /// Compares only independently captured rows. This does not qualify the
    /// unsampled rows or a complete denoising trajectory.
    public init(actual: H3TensorData, reference: H3TensorData, rows: [Int], limit: Double = 0.02) throws {
        guard limit.isFinite, limit > 0, actual.spec.shape.count == 3, actual.spec.shape[0] == 1,
              actual.spec.scalarType != .int32, !rows.isEmpty, Set(rows).count == rows.count,
              rows.allSatisfy({ (0..<actual.spec.shape[1]).contains($0) }),
              reference.spec.scalarType == actual.spec.scalarType,
              reference.spec.shape == [1, rows.count, actual.spec.shape[2]] else {
            throw AetherError.invalidTensor("H3 reference row selection or storage differs")
        }
        let stride = actual.spec.shape[2] * actual.spec.scalarType.byteWidth
        var selected = Data(); selected.reserveCapacity(rows.count * stride)
        for row in rows { selected.append(actual.bytes.subdata(in: (row * stride)..<((row + 1) * stride))) }
        let selectedTensor = try H3TensorData(spec: reference.spec, bytes: selected)
        let a = try selectedTensor.floatValues(), b = try reference.floatValues()
        var squaredError = 0.0, squaredReference = 0.0, maximum = 0.0
        for (a, b) in zip(a, b) {
            let difference = Double(a) - Double(b)
            squaredError += difference * difference; squaredReference += Double(b) * Double(b)
            maximum = max(maximum, abs(difference))
        }
        let relative = squaredReference > 0 ? (squaredError / squaredReference).squareRoot() : (squaredError == 0 ? 0 : nil)
        relativeL2 = relative; maximumAbsoluteError = maximum
        passed = relative.map { $0 <= limit } ?? false
        actualSHA256 = selectedTensor.sha256; referenceSHA256 = reference.sha256
        byteIdentical = selected == reference.bytes
    }
}
