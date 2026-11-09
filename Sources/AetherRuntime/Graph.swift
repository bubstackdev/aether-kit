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
        /// A cache shared through an app group container. Needs a process entitled to the group: Core AI returns no
        /// cache (and loading throws) for an unsigned or ad-hoc signed tool and under `swift test` (docs/RELEASE.md,
        /// "The compile cache").
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

    /// Options from a manifest graph: its required `compute`, and `expectFrequentReshapes` as written or, when
    /// absent, its default (`gpu` and any dynamic input or state), which needs the graph's `signature`.
    public init(
        manifest graph: Manifest.Graph, at field: String, signature: GraphSignature? = nil,
        cachePolicy: CachePolicy, cache: Cache
    ) throws(AetherError) {
        let reshapes: Bool
        if let stated = graph.expectFrequentReshapes {
            reshapes = stated
        } else if let signature {
            reshapes = graph.expectFrequentReshapes(hasDynamicDimension: signature.hasDynamicInputsOrStates)
        } else if graph.compute != .gpu {
            reshapes = false
        } else {
            throw .manifestInvalid(
                field: "\(field).expectFrequentReshapes",
                reason: "absent, and its default depends on the graph's dynamic dimensions; state it or pass the signature")
        }
        self.init(computeUnit: graph.compute, expectFrequentReshapes: reshapes, cachePolicy: cachePolicy, cache: cache)
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
    private var shapes: Set<[[Int]]> = []

    /// How many distinct input shape combinations this graph has run or been encoded at. Core AI keeps a
    /// specialization's memory for each until the model is released, even with `expectFrequentReshapes`
    /// (lab/h3-spike §3), so a measured footprint holds only for the shapes it was measured at
    /// (``ShapeProfile``).
    public var distinctShapeCount: Int { shapes.count }

    /// Specializes the model at `url` with `options` (through the specialization cache) and loads `function`.
    public convenience init(url: URL, function name: String, options: GraphOptions) async throws(AetherError) {
        let model = try await Self.specialize(url: url, options: options)
        try self.init(model: model, url: url, function: name, options: options)
    }

    private init(model: AIModel, url: URL, function name: String, options: GraphOptions) throws(AetherError) {
        self.url = url
        self.options = options
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

    /// Loads several functions of one model file with one specialization (phases such as `step` and `prefill`
    /// share the weights and the states). Keyed by function name.
    public static func load(url: URL, functions names: [String], options: GraphOptions) async throws(AetherError) -> [String: Graph] {
        let model = try await specialize(url: url, options: options)
        var out: [String: Graph] = [:]
        for name in names where out[name] == nil {
            out[name] = try Graph(model: model, url: url, function: name, options: options)
        }
        return out
    }

    private static func specialize(url: URL, options: GraphOptions) async throws(AetherError) -> AIModel {
        do {
            return try await AIModel.specialize(
                contentsOf: url, options: options.specializationOptions, cache: try options.aiCache(),
                cachePolicy: options.aiCachePolicy)
        } catch let error as AetherError {
            throw error
        } catch {
            throw .runtime(underlying: error)
        }
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
        note(bindings)
        try await bindings.arena.run(function, bindings: bindings)
    }

    /// Encodes the function on `bindings` into `stream` without waiting (the pipelined path, DESIGN.md §6.1). The
    /// buffers must be Metal-backed and the arena lent by the caller until the stream drains.
    func encode(_ bindings: borrowing Bindings, to stream: ComputeStream) throws(AetherError) {
        try bindings.validate(against: signature)
        note(bindings)
        try bindings.arena.encode(function, bindings: bindings, to: stream)
    }

    private func note(_ bindings: borrowing Bindings) {
        shapes.insert(bindings.inputs.sorted { $0.name < $1.name }.map(\.shape))
    }
}

extension Graph {
    /// Opens graph `role`'s `phase` function of an installed bundle, resolving its file for `architecture`.
    public convenience init(
        bundle: InstalledBundle, manifest: Manifest, role: String = "decoder", phase: String = "step",
        architecture: String, options: GraphOptions
    ) async throws(AetherError) {
        guard let graph = manifest.graphs[role] else {
            throw .manifestInvalid(field: "graphs.\(role)", reason: "the manifest has no such graph")
        }
        guard let function = graph.phases[phase] else {
            throw .manifestInvalid(field: "graphs.\(role).functions.\(phase)", reason: "the graph has no such phase")
        }
        guard let path = graph.file.resolve(architecture: architecture), let url = bundle.url(of: path) else {
            throw .manifestInvalid(field: "graphs.\(role).file", reason: "no file for architecture \(architecture)")
        }
        try await self.init(url: url, function: function, options: options)
    }
}
