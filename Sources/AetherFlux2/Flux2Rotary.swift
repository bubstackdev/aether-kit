// Derived from the Apache-2.0 Aether prototype (11 §B2–B3, B14). See NOTICE.
import AetherKit
import Foundation

/// Host rotary tables for the klein denoiser: 4-axis position ids `(t, row, col, index)`, 32
/// dimensions per axis, θ 2000. Each table is `[n, 2, 128]` float32 (cos rows then sin rows) in the
/// split-halves layout the export expects (pair i rotates dims i and i + 64); angles, cos and sin are
/// computed in Double and rounded once.
public enum Flux2Rotary {
    public static let axes = [32, 32, 32, 32]
    public static let theta = 2000.0
    public static let headDimension = 128

    /// Text token i: `(0, 0, 0, i)`.
    public static func textIDs(length: Int) -> [Int32] {
        (0..<length).flatMap { [0, 0, 0, Int32($0)] }
    }

    /// A `rows × columns` grid in raster order at temporal offset `t`: `(t, row, col, 0)`.
    /// Generated tokens use t = 0; reference image k (zero-based) uses t = stride·(k + 1).
    public static func gridIDs(rows: Int, columns: Int, t: Int32 = 0) -> [Int32] {
        var ids: [Int32] = []
        ids.reserveCapacity(rows * columns * 4)
        for row in 0..<rows { for column in 0..<columns { ids += [t, Int32(row), Int32(column), 0] } }
        return ids
    }

    public static func table(ids: [Int32]) throws -> FloatTensor {
        guard !ids.isEmpty, ids.count.isMultiple(of: 4) else { throw AetherError.invalidRequest("position ids must be [n,4]") }
        let n = ids.count / 4, half = headDimension / 2
        var frequencies: [(axis: Int, value: Double)] = []
        for (axis, dimension) in axes.enumerated() {
            for j in stride(from: 0, to: dimension, by: 2) {
                frequencies.append((axis, 1 / pow(theta, Double(j) / Double(dimension))))
            }
        }
        var values = [Float](repeating: 0, count: n * 2 * headDimension)
        for token in 0..<n {
            let base = token * 2 * headDimension
            for (index, frequency) in frequencies.enumerated() {
                let angle = Double(ids[token * 4 + frequency.axis]) * frequency.value
                let c = Float(cos(angle)), s = Float(sin(angle))
                values[base + index] = c
                values[base + index + half] = c
                values[base + headDimension + index] = s
                values[base + headDimension + index + half] = s
            }
        }
        return try FloatTensor(shape: [n, 2, headDimension], values: values)
    }
}
