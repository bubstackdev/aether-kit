/// The single error domain of Aether-Kit (DESIGN.md §11).
///
/// Cases are structured so callers can react without parsing strings. Family- or layer-specific detail travels
/// in an associated payload (``DownloadFailure``, ``TensorFailure``, or `underlying`), never in a separate public
/// error enum.
public enum AetherError: Error, Sendable, CustomStringConvertible {
    /// `driver` is a known driver id (MANIFEST.md §1) that no registered driver implements in this build.
    /// Raised before anything downloads.
    case unsupported(driver: DriverID)
    /// `driver` is not a driver id this build knows at all.
    case unknownDriver(DriverID)
    /// The catalog entry has no variant this device can run.
    case unsupportedDevice(catalogID: String, platform: String, architecture: String)
    /// The id is not in the catalog.
    case unknownModel(id: String)
    /// A model reference can't name a place in the store: `field` (`repo`, `revision`, `variant`, an asset's pins or
    /// paths) has a component that is empty, `.`/`..`, absolute, or outside the allowed characters. Refused before any
    /// file-system operation.
    case invalidReference(field: String, reason: String)
    /// A manifest (or a catalog or zoo document) is malformed. `field` is the dotted path of the offending field.
    case manifestInvalid(field: String, reason: String)
    /// The manifest and the graph disagree. `field` is the dotted manifest path; the others describe both sides.
    case manifestMismatch(field: String, expected: String, found: String)
    /// A download failed. `file` is the bundle-relative path.
    case download(file: String, reason: DownloadFailure)
    /// A downloaded file's SHA-256 does not match the catalog.
    case integrity(file: String, expected: String, actual: String)
    /// Not enough room: context, memory or disk. `what` names the resource.
    case capacity(what: String, needed: Int64, available: Int64)
    /// A tensor binding or typed access was invalid (wrong type, shape, unbound name, or a buffer in use).
    case tensor(name: String, reason: TensorFailure)
    /// The underlying runtime (Core AI or the file system) failed.
    case runtime(underlying: any Error)
    /// The request is valid in general but not for this model or in its current state: a partial rewind of
    /// recurrent state, feeding a decoder whose state ran past its position, a renderer this build does not
    /// implement yet. `reason` says which, and what to do instead.
    case unsupportedOperation(String)
    /// `resource` is held by someone else: an installed bundle a reader lease keeps open, or an install another
    /// process is running. Retry once the holder lets go.
    case inUse(resource: String)
    /// Whose turn it is to use a shared resource (the GPU) was not settled in this caller's favour.
    case admission(AdmissionFailure)
    /// The system reports critical memory pressure, so no new model load starts.
    case memoryPressure
    /// A generation request failed for a reason of its own (its sampling, its constraint, its structured output or
    /// its tool calls), not of the model or the runtime.
    case generation(GenerationFailure)

    /// Why a generation request failed (``AetherError/generation(_:)``).
    public enum GenerationFailure: Sendable, Hashable, CustomStringConvertible {
        /// The sampling parameters have no defined meaning (a top-p outside (0, 1], a penalty ≤ 0, …).
        case invalidParameters(String)
        /// A logits row the sampler cannot rank: NaN, or no finite value.
        case invalidLogits(String)
        /// The token constraint left no token to choose (or does not fit the model's vocabulary).
        case constraintExhausted(String)
        /// A structured-output schema uses something the JSON-schema constraint does not support (a `pattern`, an
        /// unknown keyword). Such a request is refused before it generates, never run unconstrained.
        case unsupportedSchema(String)
        /// A token left the constraint's grammar (a processor was fed a token it did not allow).
        case constraintViolated(String)
        /// A structured reply did not decode into the requested type (for example it reached `maxTokens` first).
        case decodingFailed(String)
        /// The model's tool-call output did not parse in its dialect, or the dialect is not implemented.
        case toolCall(String)
        /// The app's content filter refused the request's prompt (before anything ran) or the generated output
        /// (before it was returned). `reason` is the filter's.
        case contentRefused(stage: ContentStage, reason: String)

        /// Where a content filter reviewed a generation.
        public enum ContentStage: String, Sendable, Hashable { case prompt, output }

        public var description: String {
            switch self {
            case .invalidParameters(let why): "invalid sampling parameters: \(why)"
            case .invalidLogits(let why): "invalid logits: \(why)"
            case .constraintExhausted(let why): "the constraint allows no token: \(why)"
            case .unsupportedSchema(let why): "unsupported schema: \(why)"
            case .constraintViolated(let why): "a token left the constraint: \(why)"
            case .decodingFailed(let why): "the structured reply did not decode: \(why)"
            case .toolCall(let why): "tool call: \(why)"
            case .contentRefused(let stage, let reason): "the content filter refused the \(stage.rawValue): \(reason)"
            }
        }
    }

    public enum DownloadFailure: Sendable, Hashable, CustomStringConvertible {
        /// The server answered with an unexpected HTTP status.
        case http(status: Int)
        /// The transfer failed (offline, reset, timed out, cancelled).
        case transport(String)
        /// The server's bytes disagree with the catalog (for example a size mismatch).
        case unexpectedContent(String)

        public var description: String {
            switch self {
            case .http(let status): "HTTP \(status)"
            case .transport(let message): message
            case .unexpectedContent(let message): message
            }
        }
    }

    /// Why an admission (in-process `GPUAdmission`, its `PriorityInferenceQueue`, or the cross-process `SharedGPULease`)
    /// refused. `holder` is the process a lease names, when it names one.
    public enum AdmissionFailure: Sendable, Hashable, CustomStringConvertible {
        /// The cross-process GPU lease is held by another live process (or by this one).
        case leaseHeld(holder: Int32?)
        /// A waiting acquisition of the GPU lease reached its timeout while the lease was still held.
        case leaseTimedOut(holder: Int32?)
        /// The lease directory could not be created for a reason other than an existing lease.
        case leaseUnavailable(path: String, errno: Int32)
        /// Fail-fast admission found every slot in use, or work already queued ahead.
        case busy
        /// A queued admission reached its timeout before a slot was free.
        case timedOut
        /// The queue's limits are not valid (a limit below 1, or more running slots than outstanding ones).
        case invalidLimits
        /// A ticket id is already queued or running.
        case duplicateID
        /// The global outstanding limit (queued plus running) refused the request.
        case globalLimit
        /// The owner's outstanding limit refused the request.
        case ownerLimit
        /// A release named a ticket that is not running.
        case notRunning
        /// A memory reservation or capacity of zero bytes.
        case invalidByteCount
        /// The memory reservation was already released.
        case reservationReleased

        public var description: String {
            func holder(_ pid: Int32?) -> String { pid.map { "process \($0)" } ?? "an unidentified holder (no pid file)" }
            switch self {
            case .leaseHeld(let pid): return "the shared GPU lease is held by \(holder(pid))"
            case .leaseTimedOut(let pid): return "timed out waiting for the shared GPU lease held by \(holder(pid))"
            case .leaseUnavailable(let path, let code): return "cannot create the GPU lease at \(path) (errno \(code))"
            case .busy: return "the GPU is in use by other work in this process"
            case .timedOut: return "timed out waiting for the GPU"
            case .invalidLimits: return "invalid admission limits"
            case .duplicateID: return "duplicate admission ticket"
            case .globalLimit: return "the global admission limit is reached"
            case .ownerLimit: return "the owner's admission limit is reached"
            case .notRunning: return "the admission ticket is not running"
            case .invalidByteCount: return "a memory reservation needs at least one byte"
            case .reservationReleased: return "the memory reservation was already released"
            }
        }
    }

    public enum TensorFailure: Sendable, Hashable, CustomStringConvertible {
        case typeMismatch(expected: String, found: String)
        case shapeMismatch(expected: [Int], found: [Int])
        case outOfBounds(index: Int, count: Int)
        case unbound
        case unknownName
        /// The buffer's storage is lent to a running graph, or the handle was released.
        case unavailable(String)
        case unsupported(String)

        public var description: String {
            switch self {
            case .typeMismatch(let e, let f): "type mismatch: expected \(e), found \(f)"
            case .shapeMismatch(let e, let f): "shape mismatch: expected \(e), found \(f)"
            case .outOfBounds(let i, let c): "index \(i) out of bounds for \(c) elements"
            case .unbound: "not bound"
            case .unknownName: "no such tensor in the graph signature"
            case .unavailable(let why): "unavailable: \(why)"
            case .unsupported(let why): "unsupported: \(why)"
            }
        }
    }

    public var description: String {
        switch self {
        case .unsupported(let driver):
            "Unsupported driver '\(driver)': no registered driver loads it"
        case .unknownDriver(let driver):
            "Unknown driver '\(driver)'"
        case .unsupportedDevice(let id, let platform, let arch):
            "'\(id)' has no variant for \(platform) (\(arch))"
        case .unknownModel(let id):
            "Unknown model '\(id)'"
        case .invalidReference(let field, let reason):
            "Invalid model reference \(field): \(reason)"
        case .manifestInvalid(let field, let reason):
            "Invalid manifest field '\(field)': \(reason)"
        case .manifestMismatch(let field, let expected, let found):
            "Manifest field '\(field)' does not match the graph: expected \(expected), found \(found)"
        case .download(let file, let reason):
            "Download of '\(file)' failed: \(reason)"
        case .integrity(let file, let expected, let actual):
            "Integrity check failed for '\(file)': expected sha256 \(expected), got \(actual)"
        case .capacity(let what, let needed, let available):
            "Insufficient \(what): needed \(needed), available \(available)"
        case .tensor(let name, let reason):
            "Tensor '\(name)': \(reason)"
        case .runtime(let underlying):
            "Runtime failure: \(underlying)"
        case .unsupportedOperation(let reason):
            "Unsupported operation: \(reason)"
        case .inUse(let resource):
            "'\(resource)' is in use"
        case .admission(let reason):
            "Admission refused: \(reason)"
        case .memoryPressure:
            "Memory pressure is critical: no new model load starts"
        case .generation(let failure):
            "Generation failed: \(failure)"
        }
    }
}
