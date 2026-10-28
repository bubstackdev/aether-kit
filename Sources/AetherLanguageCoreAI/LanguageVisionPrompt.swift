/// One image's vision-encoder output, ready for the decoder's image submission (`main` of a one-function
/// asset, or a 0.0.2 asset's `withImage`).
@available(macOS 27, iOS 27, *)
public struct LanguageImageFeatures: Sendable {
    /// Patch grid (temporal groups, rows, columns) the encoder saw.
    public let grid: (t: Int, h: Int, w: Int)
    public let mergeSize: Int
    public let hiddenSize: Int
    public let deepstackLevels: Int
    /// Image rows [tokens, hidden], row-major fp16.
    public let embeddings: [Float16]
    /// DeepStack features [levels, tokens, hidden], row-major fp16.
    public let deepstack: [Float16]

    /// Decoder tokens the image occupies: one per merge x merge window.
    public var tokenCount: Int { grid.t * (grid.h / mergeSize) * (grid.w / mergeSize) }

    public init(grid: (t: Int, h: Int, w: Int), mergeSize: Int, hiddenSize: Int, deepstackLevels: Int,
                embeddings: [Float16], deepstack: [Float16]) throws {
        guard grid.t > 0, grid.h > 0, grid.w > 0, mergeSize > 0, grid.h % mergeSize == 0, grid.w % mergeSize == 0,
              hiddenSize > 0, deepstackLevels >= 0 else { throw LanguageGenerationError.invalid("Invalid image grid") }
        let tokens = grid.t * (grid.h / mergeSize) * (grid.w / mergeSize)
        guard embeddings.count == tokens * hiddenSize, deepstack.count == deepstackLevels * tokens * hiddenSize else {
            throw LanguageGenerationError.invalid("Image features do not match their grid")
        }
        self.grid = grid
        self.mergeSize = mergeSize
        self.hiddenSize = hiddenSize
        self.deepstackLevels = deepstackLevels
        self.embeddings = embeddings
        self.deepstack = deepstack
    }
}

/// The special tokens that mark an image in a rendered Qwen3-VL prompt.
@available(macOS 27, iOS 27, *)
public struct LanguageImageTokens: Sendable, Equatable {
    public let visionStart: Int
    public let imagePad: Int
    public let visionEnd: Int

    public init(visionStart: Int, imagePad: Int, visionEnd: Int) {
        self.visionStart = visionStart
        self.imagePad = imagePad
        self.visionEnd = visionEnd
    }
}

/// A rendered prompt with images: the caller renders the chat template with one
/// `<|vision_start|><|image_pad|><|vision_end|>` placeholder per image (tokenization stays the
/// consumer's responsibility) and supplies the images' features in the same order. Kit expands
/// each placeholder's single pad token to the image's token count.
@available(macOS 27, iOS 27, *)
public struct LanguageVisionPrompt: Sendable {
    public let renderedTokenIDs: [Int]
    public let images: [LanguageImageFeatures]

    public init(renderedTokenIDs: [Int], images: [LanguageImageFeatures]) {
        self.renderedTokenIDs = renderedTokenIDs
        self.images = images
    }
}

/// Pure planning for an image prompt, shared by the session and CPU tests.
@available(macOS 27, iOS 27, *)
struct LanguagePromptPlan {
    struct ImageSpan { let start: Int; let count: Int; let image: Int }
    /// Expanded token ids (each image's pads repeated to its token count).
    let ids: [Int]
    /// 3-axis rotary positions for every prompt token: rows (time, height, width).
    let rope: [[Int32]]
    let spans: [ImageSpan]
    /// Rotary position minus cache position for every token after the prompt (text tokens take
    /// the same value on all axes and advance with the cache position).
    let ropeDelta: Int

    static func expand(_ prompt: LanguageVisionPrompt, tokens: LanguageImageTokens) throws -> LanguagePromptPlan {
        var ids: [Int] = []
        var spans: [ImageSpan] = []
        var next = 0
        var index = 0
        let rendered = prompt.renderedTokenIDs
        while index < rendered.count {
            let token = rendered[index]
            if token == tokens.imagePad {
                guard index > 0, rendered[index - 1] == tokens.visionStart, index + 1 < rendered.count,
                      rendered[index + 1] == tokens.visionEnd else {
                    throw LanguageGenerationError.invalid("An image pad must sit between vision start and end tokens")
                }
                guard next < prompt.images.count else {
                    throw LanguageGenerationError.invalid("More image placeholders than images")
                }
                let count = prompt.images[next].tokenCount
                spans.append(ImageSpan(start: ids.count, count: count, image: next))
                ids += Array(repeating: tokens.imagePad, count: count)
                next += 1
            } else {
                ids.append(token)
            }
            index += 1
        }
        guard next == prompt.images.count else { throw LanguageGenerationError.invalid("More images than placeholders") }
        let (rope, delta) = positions(count: ids.count, spans: spans, images: prompt.images)
        return LanguagePromptPlan(ids: ids, rope: rope, spans: spans, ropeDelta: delta)
    }

    /// Qwen3-VL M-RoPE: a text token takes the running rotary position p on every axis and p
    /// advances by one; an image of merged grid (t, h, w) takes (p + ti, p + hi, p + wi) in raster
    /// order and p then advances by max(t, h, w), not by its token count.
    static func positions(count: Int, spans: [ImageSpan], images: [LanguageImageFeatures]) -> ([[Int32]], Int) {
        var rows = [[Int32]](repeating: [], count: 3)
        for r in 0..<3 { rows[r].reserveCapacity(count) }
        var p = 0, index = 0, spanIndex = 0
        while index < count {
            if spanIndex < spans.count, spans[spanIndex].start == index {
                let image = images[spans[spanIndex].image]
                let (t, h, w) = (image.grid.t, image.grid.h / image.mergeSize, image.grid.w / image.mergeSize)
                for ti in 0..<t { for hi in 0..<h { for wi in 0..<w {
                    rows[0].append(Int32(p + ti)); rows[1].append(Int32(p + hi)); rows[2].append(Int32(p + wi))
                } } }
                index += spans[spanIndex].count
                p += max(t, h, w)
                spanIndex += 1
            } else {
                for r in 0..<3 { rows[r].append(Int32(p)) }
                p += 1
                index += 1
            }
        }
        return (rows, p - count)
    }

    /// Refuses a prompt with an image larger than the exported image chunk, before any native work.
    /// Without a declared chunk (`nil`), any image is refused; a prompt without images passes.
    func requireImageSpans(within limit: Int?) throws {
        guard let largest = spans.map(\.count).max() else { return }
        guard let limit else {
            throw LanguageGenerationError.invalid("This asset declares no maximum image chunk, so it accepts text-only prompts")
        }
        if largest > limit {
            throw LanguageGenerationError.invalid("An image has \(largest) tokens, more than the exported image chunk (\(limit))")
        }
    }

    /// Prefill submissions for a decoder that takes image rows in `main` (one-function assets): the whole prompt in
    /// the contract's prefill chunks, as for a text prompt, image tokens included. Each chunk carries the rows of its
    /// own image tokens (``alignedImageRows(offset:count:images:hiddenSize:levels:)``), so an image may straddle
    /// chunks and a stepping-back chunk carries its rewritten image rows again. Image spans then add no input
    /// shapes of their own: Core AI retains memory on every shape change of a function.
    func uniformSubmissions(contract: LanguageStepContract, requestedChunk: Int?, maximumChunk: Int?) -> [LanguagePrefillChunk] {
        contract.prefillChunks(promptCount: ids.count, requestedChunk: requestedChunk, maximumChunk: maximumChunk)
    }

    /// Image rows aligned with the tokens `offset..<offset + count`: row i is token offset + i's image row (image
    /// embeddings [count, hidden], DeepStack [levels, count, hidden], row-major) and zero for text tokens. Nil when
    /// the range holds no image token.
    func alignedImageRows(offset: Int, count: Int, images: [LanguageImageFeatures], hiddenSize: Int,
                          levels: Int) -> (embeddings: [Float16], deepstack: [Float16])? {
        let end = offset + count
        let touching = spans.filter { $0.start < end && $0.start + $0.count > offset }
        guard !touching.isEmpty else { return nil }
        var embeddings = [Float16](repeating: 0, count: count * hiddenSize)
        var deepstack = [Float16](repeating: 0, count: levels * count * hiddenSize)
        for span in touching {
            let image = images[span.image]
            for token in max(offset, span.start)..<min(end, span.start + span.count) {
                let row = token - span.start, slot = token - offset
                embeddings.replaceSubrange(slot * hiddenSize..<(slot + 1) * hiddenSize,
                                           with: image.embeddings[row * hiddenSize..<(row + 1) * hiddenSize])
                for level in 0..<levels {
                    let source = (level * image.tokenCount + row) * hiddenSize
                    let target = (level * count + slot) * hiddenSize
                    deepstack.replaceSubrange(target..<target + hiddenSize, with: image.deepstack[source..<source + hiddenSize])
                }
            }
        }
        return (embeddings, deepstack)
    }

    /// Prefill submissions for a 0.0.2 two-function decoder: text runs in the contract's prefill chunks (bucketed:
    /// one fixed width chosen by the whole prompt's length, the last chunk stepping back within the run), each image span as one `withImage`
    /// submission (an image never straddles submissions).
    func submissions(contract: LanguageStepContract, requestedChunk: Int?, maximumChunk: Int?) -> [(start: Int, count: Int, image: Int?)] {
        var out: [(start: Int, count: Int, image: Int?)] = []
        var cursor = 0
        // One width for every text run, chosen by the whole prompt's length.
        let width = requestedChunk ?? contract.defaultPrefillChunk(promptCount: ids.count)
        func text(until end: Int) {
            guard end > cursor else { return }
            for chunk in contract.prefillChunks(promptCount: end - cursor, requestedChunk: width, maximumChunk: maximumChunk) {
                out.append((cursor + chunk.offset, chunk.count, nil))
            }
            cursor = end
        }
        for span in spans {
            text(until: span.start)
            out.append((span.start, span.count, span.image))
            cursor = span.start + span.count
        }
        text(until: ids.count)
        return out
    }
}
