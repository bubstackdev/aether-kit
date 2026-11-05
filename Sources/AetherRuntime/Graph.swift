public import Foundation
import CoreAI

/// Per-graph load options (DESIGN.md §6.1). They replace every process-global setting coreai-kit used.
///
/// Nothing has a default: specialization decides whether a decoder runs at 4 or 274 tok/s (M0), so every field is
/// stated by the caller, normally from the manifest via ``init(manifest:at:cachePolicy:cache:)``.
public struct GraphOptions: Sendable, Hashable {
    public enum CachePolicy: Sendable, Hashable {
        /// Core AI's default purge conditions.
        case `default`
        /// Kept across launches until the source asset changes.
        case persistent
    }

    public enum Cache: Sendable, Hashable {
        /// The process's default specialization cache.
        case `default`
        /// A cache shared through an app group container.
        case appGroup(String)
    }

    public var computeUnit: Manifest.ComputeUnit
    public var expectFrequentReshapes: Bool
    public var cachePolicy: CachePolicy
    public var cache: Cache

    public init(
        computeUnit: Manifest.ComputeUnit, expectFrequentReshapes: Bool, cachePolicy: CachePolicy, cache: Cache
    ) {
        self.computeUnit = computeUnit
        self.expectFrequentReshapes = expectFrequentReshapes
        self.cachePolicy = cachePolicy
        self.cache = cache
    }

    /// Options from a manifest graph. Fails, naming the field, when the manifest does not state its specialization.
    public init(
        manifest graph: Manifest.Graph, at field: String, cachePolicy: CachePolicy, cache: Cache
    ) throws(AetherError) {
        guard let specialization = graph.specialization else {
            throw .manifestInvalid(
                field: "\(field).specialization", reason: "required to load: Aether never specializes with defaults")
        }
        self.init(
            computeUnit: specialization.computeUnit, expectFrequentReshapes: specialization.expectFrequentReshapes,
            cachePolicy: cachePolicy, cache: cache)
    }

    var specializationOptions: SpecializationOptions {
        let kind: ComputeUnitKind = switch computeUnit {
        case .cpu: .cpu
        case .gpu: .gpu
        case .neuralEngine: .neuralEngine
        }
        var options = SpecializationOptions(preferredComputeUnitKind: kind)
        options.expectFrequentReshapes = expectFrequentReshapes
        return options
    }

    var aiCachePolicy: AIModelCache.Policy {
        switch cachePolicy {
        case .default: .default
        case .persistent: .persistent
        }
    }

    func aiCache() throws(AetherError) -> AIModelCache {
        switch cache {
        case .default:
            return .default
        case .appGroup(let group):
            guard let cache = AIModelCache(appGroup: group) else {
                throw .runtime(underlying: CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: group]))
            }
            return cache
        }
    }
}

/// One loaded graph function (DESIGN.md §6.1): `AIModel.specialize` + `loadFunction`, its ``GraphSignature``, and
/// ``run(_:)``.
///
/// Non-`Sendable`: a graph is owned by one engine or pipeline actor. With `NonisolatedNonsendingByDefault`, `run`
/// executes on that actor's executor.
public final class Graph: GraphBackend {
    public let url: URL
    public let options: GraphOptions
    public let signature: GraphSignature
    private let function: InferenceFunction

    /// Specializes the model at `url` with `options` (through the specialization cache) and loads `function`.
    public init(url: URL, function name: String, options: GraphOptions) async throws(AetherError) {
        self.url = url
        self.options = options
        let model: AIModel
        do {
            model = try await AIModel.specialize(
                contentsOf: url, options: options.specializationOptions, cache: try options.aiCache(),
                cachePolicy: options.aiCachePolicy)
        } catch let error as AetherError {
            throw error
        } catch {
            throw .runtime(underlying: error)
        }
        let loaded: InferenceFunction?
        do { loaded = try model.loadFunction(named: name) } catch { throw .runtime(underlying: error) }
        guard let loaded else {
            throw .manifestMismatch(
                field: "function", expected: "a function named '\(name)'",
                found: "functions \(model.functionNames) in \(url.lastPathComponent)")
        }
        function = loaded
        signature = GraphSignature(loaded.descriptor)
    }

    /// Allocates a buffer for `name` from the graph's own descriptor (keeping Core AI's preferred strides), with
    /// dynamic dimensions resolved in order from `dimensions`.
    public func makeBuffer(
        for name: String, role: TensorRole, dimensions: [Int], in arena: TensorArena
    ) throws(AetherError) -> TensorBuffer {
        let descriptor: InferenceValue.Descriptor? = switch role {
        case .input: function.descriptor.inputDescriptor(of: name)
        case .state: function.descriptor.stateDescriptor(of: name)
        case .output: function.descriptor.outputDescriptor(of: name)
        }
        guard case .ndArray(let d)? = descriptor else {
            throw .tensor(name: name, reason: descriptor == nil ? .unknownName : .unsupported("not an NDArray"))
        }
        let spec = TensorSpec(name: name, scalarType: ScalarType(d.scalarType), shape: d.shape)
        let shape = try spec.resolving(dimensions)
        return try arena.allocate(name: name, descriptor: d.hasDynamicShape ? d.resolvingDynamicDimensions(shape) : d)
    }

    /// Runs the function on `bindings`. Validates the bindings against the signature first; allocates no tensors.
    public func run(_ bindings: inout Bindings) async throws(AetherError) {
        try bindings.validate(against: signature)
        try await bindings.arena.run(function, bindings: bindings)
    }
}
