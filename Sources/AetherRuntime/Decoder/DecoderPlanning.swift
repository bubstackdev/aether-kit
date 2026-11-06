import Foundation

#if os(iOS)
import os
#endif

/// How much memory a model may use (DESIGN.md §6.2, §10): `os_proc_available_memory()` on iOS, a configurable
/// fraction of RAM on macOS, minus a safety reserve. Injectable so capacity sizing is testable.
///
/// What is charged against it (``KVSizing``): the engine's own buffers (KV and recurrent states, logits and token
/// buffers, copied tables) **and** the model's own memory. Neither figure the OS reports is complete:
/// - an iPhone probe (lab/device-probe README, second 2026-11-05 section) saw `phys_footprint` barely move with the
///   weights loaded (qwen3-0.6b 0.14 GB after load, 0.85 GB with 469 MB of KV at capacity 4096);
/// - but the earlier AetherKit 0.0.7 measured 400 MiB of footprint against 8.57 GiB of Metal allocations for a 4B
///   model (_reference/aether-kit-remote docs/language-generation.md:577-578): footprint does not see Core AI's
///   Metal memory, and `os_proc_available_memory()` does not see mapped address space either (:524-528).
/// So weights are never assumed free. The engine measures the model's GPU memory as the
/// `MTLDevice.currentAllocatedSize` delta across its load and charges the larger of that and the graph files'
/// size on disk. Which measure iOS actually enforces against is for DESIGN §6.2 to settle on device.
public struct MemoryBudget: Sendable, Hashable {
    public enum Rule: Sendable, Hashable {
        /// What the process may still allocate (`os_proc_available_memory()`; iOS only, RAM elsewhere).
        case processAvailable
        /// A fraction of installed RAM.
        case fractionOfRAM(Double)
        /// A fixed number of bytes (tests, or an app's own policy).
        case bytes(Int64)
    }

    public var rule: Rule
    /// Kept free of the engine's buffers: headroom for the app, transient allocations and charged weight pages.
    public var reserveBytes: Int64

    public init(_ rule: Rule, reserveBytes: Int64 = 0) {
        self.rule = rule
        self.reserveBytes = reserveBytes
    }

    /// The platform default: process-available memory minus 512 MB on iOS, half of RAM on macOS.
    public static var platformDefault: MemoryBudget {
        #if os(iOS)
        MemoryBudget(.processAvailable, reserveBytes: 512 << 20)
        #else
        MemoryBudget(.fractionOfRAM(0.5))
        #endif
    }

    public static func bytes(_ n: Int64) -> MemoryBudget { MemoryBudget(.bytes(n)) }

    /// The bytes the engine's owned buffers may take on `device`: the rule's figure minus the reserve.
    public func bytes(on device: Device) -> Int64 {
        max(0, raw(on: device) - reserveBytes)
    }

    private func raw(on device: Device) -> Int64 {
        switch rule {
        case .bytes(let n):
            return n
        case .fractionOfRAM(let f):
            return Int64(Double(device.physicalMemory) * max(0, min(f, 1)))
        case .processAvailable:
            #if os(iOS)
            return Int64(os_proc_available_memory())
            #else
            return Int64(device.physicalMemory)
            #endif
        }
    }
}

/// KV capacity sizing (DESIGN.md §6.2): `min(maxContext, kvCapacityCap[platform], what fits the budget)`, where
/// the budget is charged with the model's memory and the engine's owned bytes (see ``MemoryBudget``). Pure, so the
/// rule is tested without a model.
public struct KVSizing: Sendable, Hashable {
    /// `limits.maxContext`.
    public var maxContext: Int
    /// `limits.kvCapacityCap[platform]`.
    public var platformCap: Int?
    /// `limits.maxPosition`: positions stop there, so capacity never exceeds it + 1.
    public var maxPosition: Int?
    /// Bytes one more sequence slot costs, summed over every sequence state.
    public var bytesPerToken: Int64
    /// Owned bytes that do not scale with capacity: recurrent states, logits and token buffers, copied tables.
    public var fixedBytes: Int64
    /// The model's own memory: max(the Metal allocation delta across its load, its files on disk).
    public var modelBytes: Int64
    /// A sequence axis that is static in the graph: capacity is that, and fixed.
    public var staticAxis: Int?

    public init(
        maxContext: Int, platformCap: Int? = nil, maxPosition: Int? = nil, bytesPerToken: Int64, fixedBytes: Int64,
        modelBytes: Int64 = 0, staticAxis: Int? = nil
    ) {
        self.maxContext = maxContext
        self.platformCap = platformCap
        self.maxPosition = maxPosition
        self.bytesPerToken = bytesPerToken
        self.fixedBytes = fixedBytes
        self.modelBytes = modelBytes
        self.staticAxis = staticAxis
    }

    /// Memory-derived capacities above this are rounded down to a multiple of it.
    static let granule = 256

    /// The capacity the engine may grow to, given `budget` (``MemoryBudget/bytes(on:)``). `requested` (an app's
    /// explicit capacity) must not exceed it.
    public func capacity(budget: Int64, requested: Int? = nil) throws(AetherError) -> Int {
        var limit = maxContext
        if let platformCap { limit = min(limit, platformCap) }
        if let maxPosition { limit = min(limit, maxPosition + 1) }
        if let staticAxis { limit = min(limit, staticAxis) }
        let available = budget - modelBytes - fixedBytes
        let fits = bytesPerToken > 0 ? available / bytesPerToken : Int64(limit)
        var derived = Int(min(Int64(limit), max(fits, 0)))
        if derived < limit, derived > Self.granule { derived -= derived % Self.granule }
        guard derived > 0 else {
            throw .capacity(what: "KV memory (bytes)", needed: modelBytes + fixedBytes + bytesPerToken, available: budget)
        }
        if let staticAxis, derived < staticAxis {
            throw .capacity(what: "KV capacity (static sequence axis)", needed: Int64(staticAxis), available: Int64(derived))
        }
        if let requested {
            guard requested > 0, requested <= derived else {
                throw .capacity(what: "KV capacity (tokens)", needed: Int64(requested), available: Int64(derived))
            }
            return requested
        }
        return derived
    }

    /// The next allocation when `needed` slots exceed `current`: doubling, at least `needed`, at most `limit`.
    public static func grow(current: Int, needed: Int, limit: Int) throws(AetherError) -> Int {
        guard needed <= limit else {
            throw .capacity(what: "KV capacity (tokens)", needed: Int64(needed), available: Int64(limit))
        }
        var next = max(current, 1)
        while next < needed { next *= 2 }
        return min(next, limit)
    }
}

/// The prefill schedule (DESIGN.md §6.2), from the manifest and graph signatures only:
/// 1. a `prefill` phase with static S=C: full chunks on `prefill`, then the remainder on `step`; with `padTail`, a
///    padded final chunk replaces the remainder;
/// 2. else, a `step` with static S=1: one token per call;
/// 3. else, dynamic chunks of at most `prefillChunk` (default 512, a correctness limit).
public struct PrefillPlan: Sendable, Hashable {
    public enum Phase: String, Sendable, Hashable { case step, prefill }

    public struct Call: Sendable, Hashable {
        public var phase: Phase
        /// Offset of the first real token within the fed tokens.
        public var start: Int
        /// Real tokens in this call.
        public var count: Int
        /// The token axis the call binds: `count`, or the static chunk when padded.
        public var fed: Int

        public init(phase: Phase, start: Int, count: Int, fed: Int) {
            self.phase = phase
            self.start = start
            self.count = count
            self.fed = fed
        }

        public var padding: Int { fed - count }
    }

    /// The shape facts the schedule needs.
    public struct Shape: Sendable, Hashable {
        /// The prefill phase's static chunk, if the graph has one.
        public var prefillChunk: Int?
        public var padTail: Bool
        /// Whether a padded prefill chunk can return the last real token's row (its logits keep every row).
        public var prefillRowsAll: Bool
        /// The step phase's static token axis (1 for decode-only graphs), or nil when dynamic.
        public var stepStatic: Int?
        /// The cap for dynamic chunks (`limits.prefillChunk`, default 512).
        public var dynamicChunk: Int

        public init(prefillChunk: Int? = nil, padTail: Bool = false, prefillRowsAll: Bool = true, stepStatic: Int? = nil, dynamicChunk: Int = 512) {
            self.prefillChunk = prefillChunk
            self.padTail = padTail
            self.prefillRowsAll = prefillRowsAll
            self.stepStatic = stepStatic
            self.dynamicChunk = dynamicChunk
        }
    }

    public var calls: [Call]

    /// Plans feeding `n` tokens. `padRoom` is how many slots past the real tokens the KV can take, so a padded tail
    /// is used only when it fits.
    public init(tokens n: Int, shape: Shape, padRoom: Int = .max) {
        var calls: [Call] = []
        var start = 0
        if let c = shape.prefillChunk, c > 1 {
            let full = n / c
            for _ in 0..<full {
                calls.append(Call(phase: .prefill, start: start, count: c, fed: c))
                start += c
            }
            let remainder = n - start
            if remainder > 0, shape.padTail, shape.prefillRowsAll, c - remainder <= padRoom {
                calls.append(Call(phase: .prefill, start: start, count: remainder, fed: c))
                start = n
            }
        }
        let rest = n - start
        if rest > 0 {
            let chunk = shape.stepStatic ?? max(1, shape.dynamicChunk)
            while start < n {
                let count = min(chunk, n - start)
                calls.append(Call(phase: .step, start: start, count: count, fed: count))
                start += count
            }
        }
        self.calls = calls
    }
}
