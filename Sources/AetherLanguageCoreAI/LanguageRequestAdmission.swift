import AetherKit

/// The start of every language request once it is validated, shared by the native session and CPU
/// tests: GPU admission (when the session has a ``GPUAdmission`` owner), then `prepare` under the
/// grant (the memory-budget reload, so a multi-GB load never runs while another owner holds the
/// GPU), then a close check. Owned by, and only used on, its session's actor.
///
/// `withdraw()` is the session's close: a request waiting for admission fails with `closed` at once,
/// later admissions are refused, and a request inside `prepare` fails with `closed` when it returns
/// instead of going on to generate. So close never waits behind the GPU queue, and waits at most
/// for an admitted request's `prepare` (or its generation, once that has started).
final class LanguageRequestAdmission {
    private var pending: GPUAdmissionAttempt?
    private(set) var withdrawn = false

    func withdraw() {
        withdrawn = true
        pending?.withdraw()
    }

    func begin(gpu: GPUAdmissionOwner?, prepare: () async throws -> Void,
               isolation: isolated (any Actor)? = #isolation) async throws -> GPUAdmissionGrant? {
        guard !withdrawn else { throw LanguageGenerationError.closed }
        var grant: GPUAdmissionGrant?
        if let gpu {
            let attempt = GPUAdmissionAttempt(gpu)
            pending = attempt
            defer { pending = nil }
            grant = try await attempt.admit(orThrow: { LanguageGenerationError.closed })
        }
        do {
            try await prepare()
            guard !withdrawn else { throw LanguageGenerationError.closed }
            return grant
        } catch {
            grant?.release()
            throw error
        }
    }
}
