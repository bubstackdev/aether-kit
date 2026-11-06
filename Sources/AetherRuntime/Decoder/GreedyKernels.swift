import CoreAI
import Metal

/// The GPU side of the pipelined decoder (DESIGN.md §6.1): one `MTLCommandQueue` that backs the engine's
/// `ComputeStream`, and a single-threadgroup argmax kernel committed to that same queue right after each `encode`.
///
/// Non-`Sendable`; owned by one engine core, inside its actor.
final class GreedyKernels {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let stream: ComputeStream
    private let argmax: [ScalarType: any MTLComputePipelineState]
    private let spin: any MTLComputePipelineState
    private let spinScratch: any MTLBuffer
    private let threads: Int

    init(device: any MTLDevice) async throws(AetherError) {
        guard let queue = device.makeCommandQueue() else {
            throw .runtime(underlying: GPUFailure("cannot make a command queue"))
        }
        self.device = device
        self.queue = queue
        stream = ComputeStream(commandQueue: queue)
        do {
            let library = try await device.makeLibrary(source: Self.source, options: nil)
            func pipeline(_ name: String) async throws -> any MTLComputePipelineState {
                guard let function = library.makeFunction(name: name) else { throw GPUFailure("no kernel \(name)") }
                return try await device.makeComputePipelineState(function: function)
            }
            argmax = [
                .float16: try await pipeline("aether_argmax_f16"),
                .bfloat16: try await pipeline("aether_argmax_bf16"),
                .float32: try await pipeline("aether_argmax_f32"),
            ]
            spin = try await pipeline("aether_spin")
        } catch {
            throw .runtime(underlying: error)
        }
        guard let scratch = device.makeBuffer(length: 256, options: .storageModeShared) else {
            throw .runtime(underlying: GPUFailure("cannot allocate the spin scratch buffer"))
        }
        spinScratch = scratch
        var t = 1
        let limit = min(argmax[.float16]!.maxTotalThreadsPerThreadgroup, 1024)
        while t * 2 <= limit { t *= 2 }
        threads = t
    }

    /// Commits `argmax(logits[0..<vocab]) → slot[0]` on the stream's queue. `delay` spins a single threadgroup first,
    /// so the token lands well after the next `encode` returns (the load-time canary). `completion` runs on Metal's
    /// completion thread once the token is in the slot.
    func commitArgmax(
        logits: any MTLBuffer, scalarType: ScalarType, vocab: Int, slot: any MTLBuffer, slotOffset: Int = 0, delay: UInt32 = 0,
        completion: @escaping @Sendable () -> Void
    ) throws(AetherError) {
        guard let pipeline = argmax[scalarType] else {
            throw .tensor(name: "logits", reason: .unsupported("no GPU argmax for \(scalarType) logits"))
        }
        guard let buffer = queue.makeCommandBuffer(), let encoder = buffer.makeComputeCommandEncoder() else {
            throw .runtime(underlying: GPUFailure("cannot make a command buffer"))
        }
        if delay > 0 {
            encoder.setComputePipelineState(spin)
            encoder.setBuffer(spinScratch, offset: 0, index: 0)
            var iterations = delay
            encoder.setBytes(&iterations, length: 4, index: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(slot, offset: slotOffset, index: 1)
        var n = UInt32(vocab)
        encoder.setBytes(&n, length: 4, index: 2)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
        buffer.addCompletedHandler { _ in completion() }
        buffer.commit()
    }

    /// Waits until everything committed to the queue so far (Core AI's encoded work and our kernels) has completed:
    /// the stream's own drain, then an empty sentinel command buffer (the queue completes in order). Not cancellable:
    /// buffers stay lent until the GPU is done with them.
    func drain() async {
        // Core AI forces this: `ComputeStream` is not Sendable and `currentWorkCompleted()` is a concurrent async
        // method, so calling it from the engine's actor would "send" the stream. Why it is sound: the stream is used
        // only by this object, which only the engine's DecoderCore holds, inside one actor. Every DecoderCore path that
        // awaits this drain holds its `operation` flag (a feed, a greedy step, the end of a decode), so no other code
        // on the actor can encode to the stream, or touch the arena it was lent, while the drain is suspended; and
        // the call only waits for the GPU.
        nonisolated(unsafe) let stream = self.stream
        await stream.currentWorkCompleted()
        guard let sentinel = queue.makeCommandBuffer() else { return }
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            sentinel.addCompletedHandler { _ in done.resume() }
            sentinel.commit()
        }
    }

    struct GPUFailure: Error, CustomStringConvertible {
        var description: String
        init(_ description: String) { self.description = description }
    }

    /// Argmax over one row with a single threadgroup: each thread scans a strided subset keeping its first maximum,
    /// then a tree reduction keeps the larger value and, on ties, the lower index (the CPU `Logits.argmax` rule).
    /// `aether_spin` is the canary's delay: a dependent FMA chain on one threadgroup.
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    constant int kNone = 0x7fffffff;

    inline float load(device const half *p, uint i) { return float(p[i]); }
    inline float load(device const float *p, uint i) { return p[i]; }
    inline float load(device const ushort *p, uint i) { return as_type<float>(uint(p[i]) << 16); }

    template <typename T>
    inline void argmax_row(device const T *row, device int *out, uint n, uint tid, uint width,
                           threadgroup float *values, threadgroup int *indices) {
        float best = -INFINITY;
        int at = kNone;
        for (uint i = tid; i < n; i += width) {
            float v = load(row, i);
            if (v > best) { best = v; at = int(i); }
        }
        values[tid] = best;
        indices[tid] = at;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint half_width = width / 2; half_width > 0; half_width >>= 1) {
            if (tid < half_width) {
                float v = values[tid + half_width];
                int j = indices[tid + half_width];
                if (v > values[tid] || (v == values[tid] && j < indices[tid])) { values[tid] = v; indices[tid] = j; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) { out[0] = indices[0] == kNone ? 0 : indices[0]; }
    }

    #define ARGMAX_KERNEL(NAME, T) \\
    kernel void NAME(device const T *row [[buffer(0)]], device int *out [[buffer(1)]], constant uint &n [[buffer(2)]], \\
                     uint tid [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]]) { \\
        threadgroup float values[1024]; \\
        threadgroup int indices[1024]; \\
        argmax_row(row, out, n, tid, width, values, indices); \\
    }
    ARGMAX_KERNEL(aether_argmax_f16, half)
    ARGMAX_KERNEL(aether_argmax_f32, float)
    ARGMAX_KERNEL(aether_argmax_bf16, ushort)

    kernel void aether_spin(device float *scratch [[buffer(0)]], constant uint &iterations [[buffer(1)]],
                            uint tid [[thread_index_in_threadgroup]]) {
        float x = scratch[tid];
        for (uint i = 0; i < iterations; i++) { x = fma(x, 0.99999994f, 1.0e-7f); }
        scratch[tid] = x;
    }
    """
}
