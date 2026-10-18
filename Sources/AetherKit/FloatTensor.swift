/// An owned, contiguous, row-major float32 tensor used by the reference execution path.
///
/// This deliberately copies at the native boundary. Device-resident tensors will have
/// a separate lifetime contract; this type makes no zero-copy performance promise.
public typealias FloatTensor = Tensor<Float>

/// Scalar storage accepted by owned reference tensors.
public protocol TensorScalar: Sendable, Equatable {
    var isFinite: Bool { get }
}

extension Float: TensorScalar {}
extension Float16: TensorScalar {}
extension Int32: TensorScalar {
    public var isFinite: Bool { true }
}

public typealias Float16Tensor = Tensor<Float16>
public typealias Int32Tensor = Tensor<Int32>

public struct Tensor<Scalar: TensorScalar>: Sendable, Equatable {
    public let shape: [Int]
    public let values: [Scalar]

    public init(shape: [Int], values: [Scalar]) throws {
        let count = try Self.elementCount(for: shape)
        guard count == values.count else {
            throw AetherError.invalidTensor("shape requires \(count) elements, received \(values.count)")
        }
        guard values.allSatisfy(\.isFinite) else {
            throw AetherError.invalidTensor("values must be finite")
        }
        self.shape = shape
        self.values = values
    }

    /// An empty shape represents one scalar. Empty dimensions are unsupported.
    public static func elementCount(for shape: [Int]) throws -> Int {
        var count = 1
        for dimension in shape {
            guard dimension > 0 else {
                throw AetherError.invalidTensor("dimensions must be positive")
            }
            let product = count.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else {
                throw AetherError.invalidTensor("element count overflows Int")
            }
            count = product.partialValue
        }
        guard count <= Int.max / MemoryLayout<Scalar>.stride else {
            throw AetherError.invalidTensor("byte count overflows Int")
        }
        return count
    }
}

/// The scalar types supported at the experimental Core AI execution boundary.
public enum TensorValue: Sendable, Equatable {
    case float32(FloatTensor)
    case float16(Float16Tensor)
    case int32(Int32Tensor)

    public var shape: [Int] {
        switch self {
        case .float32(let tensor): tensor.shape
        case .float16(let tensor): tensor.shape
        case .int32(let tensor): tensor.shape
        }
    }
}
