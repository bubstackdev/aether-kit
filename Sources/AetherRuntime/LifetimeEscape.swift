import CoreAI

// THE LIFETIME ESCAPE (DESIGN.md §6.1). This file is the only place in Aether-Kit that overrides a lifetime
// dependence. Review it as a unit.
//
// Problem. Core AI's `InferenceFunction.run` takes `Inputs` and `MutableViews`, which are `~Escapable`: the
// compiler ties each view to the scoped access that produced it (`arrays[i].mutableRawView()` borrows `arrays[i]`
// for one statement only). A run needs many views alive at once, across an `await`, so the checker cannot prove
// them safe and rejects the natural code.
//
// Escape. Each view is re-tied to an immortal-for-the-checker value (`_overrideLifetime(…, borrowing: Void())`),
// and the three containers are laundered once more by `immortal(_:)` just before the call. This is the pattern
// the coreai-models fork uses; the M0 spike verified it on Swift 6.4.2 against real bundles.
//
// Why it is sound here. The views point into `NDArray` storage held by `TensorArena.arrays`; the escaped values
// never leave `run(…)` below. For the whole call:
//   1. The arena (and so every array) is retained by `self`, and the views are consumed by `function.run`, which
//      returns before this method does.
//   2. No other code touches the storage: `beginRun` marks the arena lent, and every host access path
//      (`check`, allocation, release) fails with `.unavailable` until `endRun`. The arena is non-Sendable, so the
//      only other code that could reach it is a re-entrant call on the owning actor during the `await`, and that
//      path is the one the flag blocks.
//   3. No array is copied or reallocated while lent: `arrays` holds the only reference to each storage (the arena
//      never hands out `NDArray` copies), so `mutableRawView()` does not trigger copy-on-write, and slots are only
//      replaced by `release`/allocation, both blocked while lent. Swift may move the `arrays` buffer itself, but
//      the views point at each NDArray's heap storage, not at the struct.
//   4. A storage is bound at most once per run (`Bindings.validate`), so Core AI never sees aliasing mutable views.

extension TensorArena {
    /// Runs `function` with `bindings`' buffers. Allocates no tensor storage; the containers Core AI requires are
    /// the only per-call allocations. Bindings must already be validated against the function's signature.
    func run(_ function: InferenceFunction, bindings: borrowing Bindings) async throws(AetherError) {
        try beginRun(function.descriptor.name)
        defer { endRun() }

        var inputs = InferenceFunction.Inputs()
        for bound in bindings.inputs {
            let view = _overrideLifetime(arrays[bound.buffer.slot].mutableRawView(), borrowing: Void())
            if let extent = bound.extent {
                let region = view.slice(at: extent.map { (0..<$0) as any NDArray.RangeExpression })
                inputs.insert(_overrideLifetime(region.rawView, copying: region), for: bound.name)
            } else {
                inputs.insert(_overrideLifetime(view.rawView, copying: view), for: bound.name)
            }
        }
        var states = InferenceFunction.MutableViews()
        for bound in bindings.states {
            let view = _overrideLifetime(arrays[bound.buffer.slot].mutableRawView(), borrowing: Void())
            states.insert(view, for: bound.name)
        }
        var outputs = InferenceFunction.MutableViews()
        for bound in bindings.outputs {
            let view = _overrideLifetime(arrays[bound.buffer.slot].mutableRawView(), borrowing: Void())
            if let extent = bound.extent {
                outputs.insert(view.slice(at: extent.map { (0..<$0) as any NDArray.RangeExpression }), for: bound.name)
            } else {
                outputs.insert(view, for: bound.name)
            }
        }

        let boundInputs = immortal(inputs)
        do {
            // Outputs the bindings leave unbound come back here, allocated by Core AI, and are dropped.
            _ = try await function.run(inputs: boundInputs, states: immortal(consume states), outputViews: immortal(consume outputs))
        } catch {
            throw .runtime(underlying: error)
        }
    }
}

/// Launders a view container's lifetime to immortal. Only `TensorArena.run` above may call it.
@inline(__always)
@_unsafeNonescapableResult
@_lifetime(immortal)
private func immortal<T: ~Copyable & ~Escapable>(_ value: consuming T) -> T {
    value
}
