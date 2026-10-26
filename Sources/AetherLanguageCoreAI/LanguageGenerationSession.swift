@preconcurrency import CoreAI
import Foundation
import AetherCoreAI
import AetherKit
#if os(iOS)
import os
#endif
import Metal

/// Sequential correctness reference. Core AI's unannotated stream stays inside
/// this actor. Admission rejects overlap, and every submission is joined before
/// buffers are read, reset, or released, including after task cancellation.
public actor LanguageGenerationSession {
    public struct Reply: Sendable {
        public let tokens: [Int]
        public let timeToFirstTokenSeconds: Double
        public let totalSeconds: Double
        public let stopReason: String
    }
    public enum NativeEvent: Sendable, Equatable {
        case submitted(position: Int, count: Int)
        case completed(position: Int, count: Int)
    }
    public typealias Observer = @Sendable (NativeEvent) -> Void
    public enum State: String, Sendable { case ready, running, closing, closed }

    private var model: AIModel?
    private var function: InferenceFunction?
    private var stream: ComputeStream?
    private var keys: (any MTLBuffer)?
    private var values: (any MTLBuffer)?
    private let keyTemplate: NDArrayDescriptor
    private let valueTemplate: NDArrayDescriptor
    private var keyDescriptor: NDArrayDescriptor
    private var valueDescriptor: NDArrayDescriptor
    private let vocabularySize: Int
    private let stopTokenIDs: Set<Int>
    private struct ReopenConfiguration: Sendable {
        let url: URL
        let compute: ModelComputePolicy
        let expectFrequentReshapes: Bool
        let maximumSubmissions: Int
    }
    private var reopenConfiguration: ReopenConfiguration?
    private var submissionsSinceReload = 0
    private var modelReloadCount = 0
    private let queue: any MTLCommandQueue
    private let device: any MTLDevice
    private var allocatedCapacity: Int
    private var materializedTokens = 0
    private var growthCount = 0
    private let capacity: Int
    private let lifecycle = LanguageSessionLifecycle()
    public var state: State { lifecycle.state }

    /// Opens a registered local model through Aether Kit. No download or fallback.
    public static func open(bundle directory: URL, platform: LanguageModelManifest.Platform) async throws -> LanguageGenerationSession {
        let manifest = try LanguageModelManifest.load(from: directory)
        let modelURL = try manifest.verify(in: directory, platform: platform)
        return try await open(at: modelURL, maximumContextLength: manifest.maximumContextLength,
            stopTokenIDs: Set(manifest.stopTokenIDs))
    }

    /// Explicit raw-asset entry point for qualification before bundle registration.
    public static func open(at modelURL: URL, maximumContextLength: Int, initialCacheCapacity: Int = 256,
                            stopTokenIDs: Set<Int> = [151645], compute: ModelComputePolicy = .preferGPU,
                            expectFrequentReshapes: Bool = true, maximumSubmissionsPerModel: Int = 128) async throws -> LanguageGenerationSession {
        try Task.checkCancellation()
        guard maximumSubmissionsPerModel > 0 else { throw LanguageGenerationError.invalid("Invalid model residency policy") }
        var options: SpecializationOptions
        switch compute {
        case .automatic: options = .default
        case .cpuOnly: options = .cpuOnly
        case .preferGPU: options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
        options.expectFrequentReshapes = expectFrequentReshapes
        let model = try await AIModel(contentsOf: modelURL, options: options)
        try Task.checkCancellation()
        let session = try LanguageGenerationSession(model: model, capacity: maximumContextLength,
            initialCacheCapacity: initialCacheCapacity, stopTokenIDs: stopTokenIDs)
        await session.configureReopening(ReopenConfiguration(url: modelURL, compute: compute,
            expectFrequentReshapes: expectFrequentReshapes, maximumSubmissions: maximumSubmissionsPerModel))
        return session
    }
    public static var deviceArchitectureName: String { AIModel.deviceArchitectureName }

    public init(model: AIModel, capacity: Int = 512, initialCacheCapacity: Int = 256, stopTokenIDs: Set<Int> = [151645]) throws {
        guard capacity > 0, capacity <= 16384, initialCacheCapacity > 0,
              let d = model.functionDescriptor(for: "main"),
              Set(d.inputNames) == ["input_ids", "position_ids"],
              Set(d.stateNames) == ["keyCache", "valueCache"],
              d.outputNames == ["logits"],
              case .ndArray(let key)? = d.stateDescriptor(of: "keyCache"),
              case .ndArray(let value)? = d.stateDescriptor(of: "valueCache"),
              case .ndArray(let logits)? = d.outputDescriptor(of: "logits"),
              key.shape.count == 5, key.shape[3] == -1, value.shape == key.shape,
              key.shape.enumerated().allSatisfy({ $0.offset == 3 || $0.element > 0 }),
              key.scalarType == .float16, value.scalarType == .float16,
              logits.shape.count == 3, logits.shape[0] == 1, logits.shape[1] == -1, logits.shape[2] > 0,
              logits.scalarType == .float16, !stopTokenIDs.isEmpty,
              stopTokenIDs.allSatisfy({ $0 >= 0 && $0 < logits.shape[2] }),
              let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue()
        else { throw LanguageGenerationError.invalid("Unsupported language probe contract") }
        for name in d.inputNames {
            guard case .ndArray(let input)? = d.inputDescriptor(of: name),
                  input.scalarType == .int32, input.shape == [1, -1]
            else { throw LanguageGenerationError.invalid("Unsupported token/position contract") }
        }
        var shape = key.shape
        let initial = min(initialCacheCapacity, capacity)
        shape[3] = initial
        let kd = key.resolvingDynamicDimensions(shape)
        let vd = value.resolvingDynamicDimensions(shape)
        guard let kb = device.makeBuffer(length: kd.minimumByteCount, options: .storageModeShared),
              let vb = device.makeBuffer(length: vd.minimumByteCount, options: .storageModeShared)
        else { throw LanguageGenerationError.invalid("Could not allocate bounded KV caches") }
        self.model = model
        guard let function = try model.loadFunction(named: "main") else {
            throw LanguageGenerationError.invalid("Missing main inference function")
        }
        self.function = function
        self.stream = ComputeStream(commandQueue: queue)
        self.keys = kb
        self.values = vb
        self.keyTemplate = key
        self.valueTemplate = value
        self.keyDescriptor = kd
        self.valueDescriptor = vd
        self.vocabularySize = logits.shape[2]
        self.stopTokenIDs = stopTokenIDs
        self.queue = queue
        self.device = device
        self.allocatedCapacity = initial
        self.capacity = capacity
    }

    public func generate(prompt: [Int], maximumTokens: Int = 48, prefillChunkSize: Int? = nil,
                  onToken: (@Sendable (Int) async throws -> Void)? = nil,
                  observer: Observer? = nil) async throws -> Reply {
        guard maximumTokens > 0 else { throw LanguageGenerationError.invalid("Output limit must be positive") }
        try begin(prompt: prompt, reserve: maximumTokens, chunkSize: prefillChunkSize)
        defer { lifecycle.finish() }
        let start = ContinuousClock.now
        let next = try Self.greedy(await prefill(prompt, chunkSize: prefillChunkSize, observer: observer))
        let first = elapsed(start)
        // Token delivery overlaps the following native step instead of idling the GPU.
        let decoded = try await LanguageDecodeLoop.run(first: next, prompt: prompt.count, maximumTokens: maximumTokens,
            stopTokenIDs: stopTokenIDs, step: { token, position in
                try Self.greedy(await self.step(tokens: [Int32(token)], position: position, observer: observer))
            }, onToken: onToken)
        return Reply(tokens: decoded.tokens, timeToFirstTokenSeconds: first, totalSeconds: elapsed(start), stopReason: decoded.reason)
    }

    /// Last-row logits after a fresh-cache prefill. Different chunk sizes exercise
    /// cache continuity without conflating near-tied logits with answer quality.
    public func prefillLogits(prompt: [Int], chunkSize: Int? = nil) async throws -> [Float] {
        try begin(prompt: prompt, reserve: 0, chunkSize: chunkSize)
        defer { lifecycle.finish() }
        return try await prefill(prompt, chunkSize: chunkSize, observer: nil)
    }

    public func teacherForcedLogits(prompt: [Int], continuation: [Int]) async throws -> [[Float]] {
        guard !prompt.isEmpty else { throw LanguageGenerationError.invalid("Empty oracle prompt") }
        try begin(prompt: prompt + continuation, reserve: 0, chunkSize: nil)
        defer { lifecycle.finish() }
        var rows = [try await prefill(prompt, chunkSize: nil, observer: nil)]
        for (index, token) in continuation.enumerated() {
            rows.append(try await step(tokens: [Int32(token)], position: prompt.count + index, observer: nil))
        }
        return rows
    }

    private func begin(prompt: [Int], reserve: Int, chunkSize: Int?) throws {
        try Task.checkCancellation()
        try lifecycle.admit()
        guard !prompt.isEmpty, prompt.allSatisfy({ $0 >= 0 && $0 < vocabularySize }),
              chunkSize == nil || chunkSize! > 0 else {
            lifecycle.finish()
            throw LanguageGenerationError.invalid("Invalid prompt or chunk size")
        }
        // Subtraction avoids overflow for hostile Int.max output limits.
        guard prompt.count <= capacity, reserve <= capacity - prompt.count else {
            lifecycle.finish()
            throw LanguageGenerationError.contextLimit(limit: capacity)
        }
        materializedTokens = 0
        for buffer in [keys, values].compactMap({ $0 }) {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
    }

    private func prefill(_ prompt: [Int], chunkSize: Int?, observer: Observer?) async throws -> [Float] {
        let width = min(chunkSize ?? prompt.count, prompt.count)
        var last: [Float] = []
        for offset in stride(from: 0, to: prompt.count, by: width) {
            let end = min(prompt.count, offset + width)
            last = try await step(tokens: prompt[offset..<end].map(Int32.init), position: offset, observer: observer)
        }
        return last
    }

    private func step(tokens: [Int32], position: Int, observer: Observer?) async throws -> [Float] {
        try Task.checkCancellation()
        guard position == materializedTokens else { throw LanguageGenerationError.invalid("Noncontiguous cache position") }
        try await recycleModelIfNeeded()
        try Task.checkCancellation()
        guard let function, let stream else { throw LanguageGenerationError.invalid("Missing native model handle") }
        try growCache(to: position + tokens.count)
        guard let keys, let values else { throw LanguageGenerationError.invalid("Missing KV caches") }
        // This export uses the full running index to address and attend its KV cache.
        let inputs = [
            "input_ids": InferenceFunction.AsyncValue(NDArray(scalars: tokens, shape: [1, tokens.count])),
            "position_ids": InferenceFunction.AsyncValue(NDArray(scalars: (0..<(position + tokens.count)).map(Int32.init), shape: [1, position + tokens.count]))
        ]
        var k = InferenceFunction.AsyncMutableValue(unsafeBuffer: keys, scalarType: .float16,
            shape: keyDescriptor.shape, strides: keyDescriptor.preferredStrides, interleaveLayout: keyDescriptor.interleaveLayout)
        var v = InferenceFunction.AsyncMutableValue(unsafeBuffer: values, scalarType: .float16,
            shape: valueDescriptor.shape, strides: valueDescriptor.preferredStrides, interleaveLayout: valueDescriptor.interleaveLayout)
        let outputs: [String: InferenceFunction.AsyncValue]
        do {
            var states = InferenceFunction.AsyncMutableViews()
            states.insert(&k, for: "keyCache")
            states.insert(&v, for: "valueCache")
            outputs = try function.encode(inputs: inputs, states: states, to: stream)
        } catch {
            await stream.currentWorkCompleted()
            throw error
        }
        observer?(.submitted(position: position, count: tokens.count))
        await stream.currentWorkCompleted()
        observer?(.completed(position: position, count: tokens.count))
        materializedTokens = position + tokens.count
        submissionsSinceReload += 1
        try Task.checkCancellation()
        guard let result = outputs["logits"], let logits = try await result.ndArray,
              logits.scalarType == .float16, logits.shape == [1, tokens.count, vocabularySize]
        else { throw LanguageGenerationError.invalid("Unexpected logits") }
        return logits.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
            let row = (shape[1] - 1) * strides[1]
            return (0..<shape[2]).map { Float(pointer[row + $0 * strides[2]]) }
        }
    }

    public static func greedy(_ logits: [Float]) throws -> Int {
        var best = -Float.infinity
        var token: Int?
        for (i, score) in logits.enumerated() {
            guard !score.isNaN else { throw LanguageGenerationError.invalid("NaN logits") }
            if score.isFinite && score > best { best = score; token = i }
        }
        guard let token else { throw LanguageGenerationError.invalid("No finite logits") }
        return token
    }

    public struct CacheStatistics: Sendable {
        public let maximumContext: Int
        public let allocatedCapacity: Int
        public let materializedTokens: Int
        public let allocatedBytes: Int
        public let growthCount: Int
        public let modelReloadCount: Int
    }
    public func cacheStatistics() -> CacheStatistics {
        CacheStatistics(maximumContext: capacity, allocatedCapacity: allocatedCapacity,
            materializedTokens: materializedTokens, allocatedBytes: (keys?.length ?? 0) + (values?.length ?? 0), growthCount: growthCount, modelReloadCount: modelReloadCount)
    }

    private func configureReopening(_ configuration: ReopenConfiguration) { reopenConfiguration = configuration }

    /// Terminal close. New requests fail with `closed`; an active request finishes and
    /// joins its native work first, so cancel its task to end a long generation sooner.
    /// Then the model, function, stream and KV caches are released. Concurrent calls
    /// return once the release completes. Open a new session to load the model again.
    public func unload() async {
        guard await lifecycle.beginClose() else { return }
        function = nil
        stream = nil
        model = nil
        reopenConfiguration = nil
        keys = nil
        values = nil
        materializedTokens = 0
        lifecycle.didClose()
    }

    /// Previous submissions have all joined. State remains in Kit-owned buffers;
    /// releasing the function, stream, and model bounds native shape-cache residency.
    private func recycleModelIfNeeded() async throws {
        guard let configuration = reopenConfiguration,
              function == nil || submissionsSinceReload >= configuration.maximumSubmissions else { return }
        function = nil
        stream = nil
        model = nil
        try Task.checkCancellation()
        var options: SpecializationOptions
        switch configuration.compute {
        case .automatic: options = .default
        case .cpuOnly: options = .cpuOnly
        case .preferGPU: options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
        options.expectFrequentReshapes = configuration.expectFrequentReshapes
        let loaded = try await AIModel(contentsOf: configuration.url, options: options)
        try Task.checkCancellation()
        guard let loadedFunction = try loaded.loadFunction(named: "main") else {
            throw LanguageGenerationError.invalid("Missing function after model reload")
        }
        model = loaded
        function = loadedFunction
        stream = ComputeStream(commandQueue: queue)
        submissionsSinceReload = 0
        modelReloadCount += 1
    }

    private func growCache(to needed: Int) throws {
        guard needed <= capacity else { throw LanguageGenerationError.contextLimit(limit: capacity) }
        guard needed > allocatedCapacity else { return }
        guard let oldKeys = keys, let oldValues = values else { throw LanguageGenerationError.invalid("Missing KV caches") }
        let next = try LanguageCachePlan.nextCapacity(current: allocatedCapacity, needed: needed, maximum: capacity)
        var shape = keyDescriptor.shape
        shape[3] = next
        let kd = keyTemplate.resolvingDynamicDimensions(shape)
        let vd = valueTemplate.resolvingDynamicDimensions(shape)
        guard keyDescriptor.preferredStrides == LanguageCachePlan.strides(keyDescriptor.shape),
              valueDescriptor.preferredStrides == LanguageCachePlan.strides(valueDescriptor.shape),
              kd.preferredStrides == LanguageCachePlan.strides(shape), vd.preferredStrides == LanguageCachePlan.strides(shape)
        else { throw LanguageGenerationError.invalid("Growing KV caches require contiguous row-major storage") }
        #if os(iOS)
        let available = UInt64(max(0, os_proc_available_memory()))
        let required = UInt64(kd.minimumByteCount) + UInt64(vd.minimumByteCount) + 64 * 1024 * 1024
        guard required <= available else {
            throw LanguageGenerationError.memoryBudget(requiredBytes: required, availableBytes: available)
        }
        #endif
        guard let newKeys = device.makeBuffer(length: kd.minimumByteCount, options: .storageModeShared),
              let newValues = device.makeBuffer(length: vd.minimumByteCount, options: .storageModeShared)
        else { throw LanguageGenerationError.invalid("Could not grow KV caches") }
        newKeys.contents().initializeMemory(as: UInt8.self, repeating: 0, count: newKeys.length)
        newValues.contents().initializeMemory(as: UInt8.self, repeating: 0, count: newValues.length)
        let rows = shape[0] * shape[1] * shape[2]
        let rowBytesPerToken = shape[4] * MemoryLayout<Float16>.size
        LanguageCachePlan.copyPrefix(from: oldKeys.contents(), to: newKeys.contents(), rows: rows,
            oldCapacity: allocatedCapacity, newCapacity: next, validTokens: materializedTokens, tokenBytes: rowBytesPerToken)
        LanguageCachePlan.copyPrefix(from: oldValues.contents(), to: newValues.contents(), rows: rows,
            oldCapacity: allocatedCapacity, newCapacity: next, validTokens: materializedTokens, tokenBytes: rowBytesPerToken)
        keys = newKeys; values = newValues; keyDescriptor = kd; valueDescriptor = vd
        allocatedCapacity = next
        growthCount += 1
    }

    private func elapsed(_ start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}

public enum LanguageGenerationError: Error, Sendable, Equatable {
    case invalid(String), busy, closed, contextLimit(limit: Int)
    case memoryBudget(requiredBytes: UInt64, availableBytes: UInt64)
}
