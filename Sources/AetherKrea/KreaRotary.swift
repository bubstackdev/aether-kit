// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit
import Foundation

public enum KreaRotary {
    /// Float32 `[imageTokens,2,128]` using the lab's split-halves layout.
    public static func image(configuration: KreaConfiguration) throws -> FloatTensor {
        let rows = configuration.height / 16
        let columns = configuration.width / 16
        var frequencies: [(axis: Int, frequency: Double)] = []
        for (axis, dimension) in [32, 48, 48].enumerated() {
            for index in stride(from: 0, to: dimension, by: 2) {
                frequencies.append((axis, 1 / pow(1000, Double(index) / Double(dimension))))
            }
        }
        var values = [Float](repeating: 0, count: configuration.imageTokenCount * 256)
        for row in 0..<rows {
            for column in 0..<columns {
                let base = (row * columns + column) * 256
                for (index, entry) in frequencies.enumerated() {
                    let angle = Double([0, row, column][entry.axis]) * entry.frequency
                    let cosine = Float(cos(angle))
                    let sine = Float(sin(angle))
                    values[base + index] = cosine
                    values[base + index + 64] = cosine
                    values[base + index + 128] = sine
                    values[base + index + 192] = sine
                }
            }
        }
        return try FloatTensor(shape: [configuration.imageTokenCount, 2, 128], values: values)
    }
}
