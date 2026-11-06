import Foundation
import Metal
import Testing
@testable import AetherRuntime

/// The GPU argmax and its completions (the owner's second review, findings 2 and 3). In the `Gates` suite: they need
/// a Metal device, which the hermetic CI runners don't promise.
extension GateTests {
    /// Runs one argmax over `row` (stored as `type`) and returns its outcome and the slot's token.
    static func gpuArgmax(
        _ kernels: GreedyKernels, _ row: [Float], as type: ScalarType
    ) async throws -> (Result<Void, AetherError>, Int32) {
        let device = kernels.device
        let bytes: [UInt8] = switch type {
        case .float16: row.map { Float16($0) }.withUnsafeBytes { Array($0) }
        case .bfloat16: row.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }.withUnsafeBytes { Array($0) }
        default: row.withUnsafeBytes { Array($0) }
        }
        let logits = try #require(device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared))
        let slot = try #require(device.makeBuffer(length: 64, options: .storageModeShared))
        slot.contents().storeBytes(of: Int32(-7), as: Int32.self)
        let outcome = await withCheckedContinuation { (done: CheckedContinuation<Result<Void, AetherError>, Never>) in
            do throws(AetherError) {
                try kernels.commitArgmax(logits: logits, scalarType: type, vocab: row.count, slot: slot, status: 0) { done.resume(returning: $0) }
            } catch {
                done.resume(returning: .failure(error))
            }
        }
        return (outcome, slot.contents().load(as: Int32.self))
    }

    static func isInvalidLogits(_ outcome: Result<Void, AetherError>) -> Bool {
        if case .failure(.generation(.invalidLogits)) = outcome { return true }
        return false
    }

    /// Finding 2 on the GPU: the argmax kernel keeps the CPU's policy. A NaN anywhere, or nothing above -inf, is
    /// `invalidLogits` (the slot gets token 0, which the next step can still read), never a silent token 0.
    @Test func gpuArgmaxKeepsTheInvalidLogitPolicy() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let kernels = try await GreedyKernels(device: device)
        for type: ScalarType in [.float32, .float16, .bfloat16] {
            for n in [5, 151_936] {
                var peaked = [Float](repeating: -1, count: n)
                peaked[n - 2] = 4
                peaked[n / 3] = 4
                let (ok, token) = try await Self.gpuArgmax(kernels, peaked, as: type)
                #expect((try? ok.get()) != nil && token == Int32(n / 3), "\(type) \(n): ties go to the lowest index")

                var forced = [Float](repeating: -.infinity, count: n)
                forced[n - 1] = .infinity
                let (inf, infToken) = try await Self.gpuArgmax(kernels, forced, as: type)
                #expect((try? inf.get()) != nil && infToken == Int32(n - 1), "\(type) \(n): +inf ranks first")

                for at in Set([0, n / 2, n - 1]) {
                    var row = peaked
                    row[at] = .nan
                    let (outcome, token) = try await Self.gpuArgmax(kernels, row, as: type)
                    #expect(Self.isInvalidLogits(outcome) && token == 0, "\(type) \(n): NaN at \(at) gave \(outcome), \(token)")
                }
                var nanAndInf = forced
                nanAndInf[0] = .nan
                #expect(Self.isInvalidLogits(try await Self.gpuArgmax(kernels, nanAndInf, as: type).0), "\(type) \(n): NaN beside +inf")
                let empty = [Float](repeating: -.infinity, count: n)
                #expect(Self.isInvalidLogits(try await Self.gpuArgmax(kernels, empty, as: type).0), "\(type) \(n): all -inf")
            }
        }
    }

    /// Finding 3: a failed command buffer is a `runtime` failure from the argmax completion and from a drain;
    /// "completed" alone is not "succeeded". (Metal can't fail on purpose, so the failure is injected at the status.)
    @Test func gpuCompletionsReportCommandBufferFailure() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let kernels = try await GreedyKernels(device: device)
        #expect(await kernels.drain() == nil)
        kernels.injectedFailure = true
        let (outcome, _) = try await Self.gpuArgmax(kernels, [1, 2, 3], as: .float32)
        guard case .failure(.runtime(let underlying)) = outcome else {
            Issue.record("a failed argmax command buffer reported \(outcome)")
            return
        }
        #expect((underlying as NSError).domain == MTLCommandBufferErrorDomain)
        let drained = await kernels.drain()
        guard case .runtime? = drained else {
            Issue.record("a failed drain sentinel reported \(String(describing: drained))")
            return
        }
        kernels.injectedFailure = false
        #expect(await kernels.drain() == nil)
        // The policy's mapping itself, for the statuses the injection doesn't produce.
        #expect(Self.isInvalidLogits(GreedyKernels.outcome(.completed, nil, invalidLogits: true)))
        if case .failure(.runtime) = GreedyKernels.outcome(.error, nil, invalidLogits: true) {} else {
            Issue.record("a failed command buffer must win over its (unreliable) status word")
        }
    }

    /// Finding 3 through the decoding session: a failed argmax command buffer mid-decode throws from `nextGreedy`,
    /// `endGreedy` still ends the decode (and reports the failed drain), and after a reset the engine decodes exactly.
    @Test func pipelinedDecodePropagatesGPUFailure() async throws {
        let ref = try Self.labJSON(RunFile.self, "parity/qwen3-0.6b-spike.json")
        let engine = try await Self.engine("qwen3-0.6b", Self.pipelinedOptions(capacity: 2048))
        #expect(await engine.decodeMode == .gpu(depth: 3))
        let logits = try #require(try await engine.prefill(ref.promptTokens))
        let first = try Logits.argmax(logits.last)
        try await engine.beginGreedy(first: first, limit: 32)
        var tokens = [first]
        for _ in 0..<4 { tokens.append(try #require(try await engine.nextGreedy())) }
        await engine.injectGPUFailure(true)
        var failure: AetherError?
        do {
            while let t = try await engine.nextGreedy() { tokens.append(t) }
        } catch {
            failure = error
        }
        guard case .runtime? = failure else {
            Issue.record("a failed command buffer mid-decode gave \(String(describing: failure)) after \(tokens.count) tokens")
            _ = try? await engine.endGreedy()
            return
        }
        // Tokens already in flight when the failure was injected still arrive; all of them are the reference's.
        #expect(Array(ref.generated.prefix(tokens.count)) == tokens)
        await #expect(throws: AetherError.self, "the drain saw the failure") { try await engine.endGreedy() }
        await engine.injectGPUFailure(false)
        let (again, _) = try await Self.greedy(engine, prompt: ref.promptTokens, count: 32)
        #expect(again == Array(ref.generated.prefix(32)))
    }
}
