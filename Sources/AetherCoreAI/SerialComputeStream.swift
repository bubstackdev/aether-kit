import CoreAI
import Metal

/// A Core AI `ComputeStream` owned by one actor that encodes to it and then joins it.
///
/// Workaround, not a fix: Core AI 27.0 does not annotate `ComputeStream` for concurrency, and its
/// `currentWorkCompleted()` is `@concurrent`, so an actor awaiting its own stream sends the stream off
/// the actor (`sending 'stream' risks causing data races`). The stream itself is only touched by
/// `encode` and `currentWorkCompleted()` below, and the rule that makes that safe is one Swift cannot
/// check: the owner admits one submission at a time and joins it before admitting the next, including
/// after cancellation and a failed encode (the one-call admission of the H3 sessions and graphs, and
/// the lifecycle admission of the language sessions). So a join never overlaps another encode or join
/// on the same stream, which is what `@unchecked Sendable` asserts here. Remove this type once Core AI
/// annotates the stream.
@available(macOS 27, iOS 27, *)
package final class SerialComputeStream: @unchecked Sendable {
    private let stream: ComputeStream

    package init() { stream = ComputeStream() }
    package init(commandQueue: any MTLCommandQueue) { stream = ComputeStream(commandQueue: commandQueue) }

    /// Encodes one submission; call `currentWorkCompleted()` before the next, also when this throws.
    package func encode(_ function: InferenceFunction, inputs: [String: InferenceFunction.AsyncValue],
                        states: consuming InferenceFunction.AsyncMutableViews = InferenceFunction.AsyncMutableViews(),
                        outputViews: consuming InferenceFunction.AsyncMutableViews = InferenceFunction.AsyncMutableViews())
        throws -> [String: InferenceFunction.AsyncValue] {
        try function.encode(inputs: inputs, states: states, outputViews: outputViews, to: stream)
    }

    /// Joins every submission encoded so far.
    @concurrent package func currentWorkCompleted() async { await stream.currentWorkCompleted() }
}
