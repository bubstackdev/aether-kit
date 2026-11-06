import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Testing
@testable import AetherImage

/// Second code review, batch C item 5: the image pipelines queue through `SerialGate`, so a request cancelled while it
/// waits for the pipeline's turn leaves at once (before: it waited for the running generation, then checked).
@Suite("Image pipelines' turns")
struct ImageTurnQueueTests {
    /// Holds every run of the graph it wraps until opened.
    final class Held: GraphBackend, @unchecked Sendable {
        let inner: any GraphBackend
        let latch: Latch
        init(_ inner: any GraphBackend, _ latch: Latch) {
            self.inner = inner
            self.latch = latch
        }
        var signature: GraphSignature { inner.signature }
        func makeBuffer(for name: String, role: TensorRole, dimensions: [Int], in arena: TensorArena) throws(AetherError) -> TensorBuffer {
            try inner.makeBuffer(for: name, role: role, dimensions: dimensions, in: arena)
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {
            await latch.entered()
            await latch.wait()
            try await inner.run(&bindings)
        }
    }

    actor Latch {
        private var open = false, hasEntered = false
        private var waiting: [CheckedContinuation<Void, Never>] = [], enteredWaiters: [CheckedContinuation<Void, Never>] = []
        func entered() {
            hasEntered = true
            enteredWaiters.forEach { $0.resume() }
            enteredWaiters = []
        }
        func untilEntered() async { if !hasEntered { await withCheckedContinuation { enteredWaiters.append($0) } } }
        func wait() async { if !open { await withCheckedContinuation { waiting.append($0) } } }
        func release() {
            open = true
            waiting.forEach { $0.resume() }
            waiting = []
        }
    }

    /// Runs `first` until it holds the turn, queues `second`, cancels it, and checks it leaves before `first` ends.
    static func check(first: @escaping @Sendable () async throws -> Void, second: @escaping @Sendable () async throws -> Void,
                      latch: Latch) async throws {
        let running = Task { try await first() }
        await latch.untilEntered()
        let queued = Task { try await second() }
        try await Task.sleep(for: .milliseconds(50))  // let it queue
        queued.cancel()
        await #expect(throws: CancellationError.self) { try await queued.value }
        await latch.release()  // only now does the running generation end
        try await running.value
    }

    @Test(.timeLimit(.minutes(1))) func latentDiffusion() async throws {
        let latch = Latch()
        let (pipeline, _) = try LatentDiffusionTests.pipeline(calls: .init(), wrap: { role, graph in
            role == "textEncoder" ? Held(graph, latch) : graph
        })
        try await Self.check(first: { _ = try await pipeline.generate(.init(prompt: "a cat", steps: 1)) },
                             second: { _ = try await pipeline.generate(.init(prompt: "a dog", steps: 1)) }, latch: latch)
    }

    @Test(.timeLimit(.minutes(1))) func flux2Klein() async throws {
        let latch = Latch()
        let (pipeline, _) = try Flux2KleinTests.pipeline(calls: .init(), wrap: { role, graph in
            role == "textEncoder" ? Held(graph, latch) : graph
        })
        let request = ImageRequest(prompt: "a cat", steps: 1, width: 32, height: 32)
        try await Self.check(first: { _ = try await pipeline.generate(request) },
                             second: { _ = try await pipeline.generate(request) }, latch: latch)
    }

    @Test(.timeLimit(.minutes(1))) func krea2() async throws {
        let latch = Latch()
        let (pipeline, _) = try Krea2Tests.pipeline(calls: .init(), wrap: { role, graph in
            role == "textEncoder" ? Held(graph, latch) : graph
        })
        let request = ImageRequest(prompt: "a cat", steps: 1, width: 32, height: 32)
        try await Self.check(first: { _ = try await pipeline.generate(request) },
                             second: { _ = try await pipeline.generate(request) }, latch: latch)
    }
}
