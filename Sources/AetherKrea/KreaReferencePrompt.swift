// Derived from the Apache-2.0 Aether prototype (AetherGen/VisionReferences, AetherCore/MRoPERule). See NOTICE.
import AetherKit

/// One image span in an expanded reference prompt: its rows and merged grid, in prompt order.
public struct KreaReferenceSpan: Sendable, Equatable {
    public let range: Range<Int>
    public let t: Int, h: Int, w: Int
    public let digest: String
}

/// Exact Qwen3-VL encoder input for Krea's `encodeWithImages`: the chat template with one user message holding each
/// image part (`Picture n: ` then `<|vision_start|><|image_pad|><|vision_end|>`) before the prompt text, each image
/// pad expanded to that image's merged token count, and three-axis M-RoPE positions over the whole sequence.
///
/// The prompt text keeps the text-only budget (546 rendered tokens, last 5 kept); images and labels are never cut and
/// count against `KreaReferenceContract.maximumImageTokens` instead. The 34-row system prefix is unchanged, so the
/// conditioning is every row after it, image rows included.
public struct KreaReferencePrompt: Sendable, Equatable {
    /// The export bounds denoiser text rows at 512 text + 576 image tokens + 4 × (5 label + 2 marker) tokens.
    public static let maximumConditioningRows = 1116

    public let tokenIDs: [Int32]
    /// `[3, L]` axis-major: time, height, width.
    public let ropePositions: [Int32]
    public let images: [KreaReferenceSpan]
    public let wasTruncated: Bool
    public var conditioningRows: Int { tokenIDs.count - KreaPrompt.leadingRows }
    public var imageTokens: Int { images.reduce(0) { $0 + $1.range.count } }

    /// The user message's image parts, in order, before the prompt text.
    public static func render(_ text: String, imageCount: Int) -> String {
        let parts = (0..<imageCount).map {
            KreaReferenceContract.label($0 + 1) + "<|vision_start|><|image_pad|><|vision_end|>"
        }.joined()
        return KreaPrompt.render(parts + text)
    }

    /// - Parameters:
    ///   - renderedTokenIDs: the tokenizer's IDs for `render(text, imageCount: images.count)` (one pad per image).
    ///   - textOnlyTokenCount: the token count of `KreaPrompt.render(text)`, which alone decides truncation.
    ///   - images: the prepared references, in prompt order.
    public init(renderedTokenIDs: [Int], textOnlyTokenCount: Int, images: [KreaReferencePixels]) throws {
        guard (1...KreaReferenceContract.maximumImages).contains(images.count) else {
            throw AetherError.invalidRequest("Krea takes 1...\(KreaReferenceContract.maximumImages) reference images")
        }
        let total = images.reduce(0) { $0 + $1.tokens }
        guard images.allSatisfy({ $0.tokens > 0 }), total <= KreaReferenceContract.maximumImageTokens else {
            throw AetherError.invalidRequest("reference images need \(total) tokens; the limit is \(KreaReferenceContract.maximumImageTokens)")
        }
        guard textOnlyTokenCount >= KreaPrompt.leadingRows + 1,
              renderedTokenIDs.allSatisfy({ $0 >= 0 && $0 <= Int32.max }) else {
            throw AetherError.invalidRequest("invalid Krea rendered token IDs")
        }
        var ids = renderedTokenIDs.map(Int32.init)
        let limit = KreaPrompt.maximumTokens
        wasTruncated = textOnlyTokenCount > limit
        if wasTruncated {
            // Image parts precede the text, so the cut just before the kept suffix removes only text.
            let keep = KreaPrompt.trailingTokens
            let cut = ids.count - keep - (textOnlyTokenCount - limit)
            guard cut > 0 else { throw AetherError.invalidRequest("invalid Krea rendered token IDs") }
            ids = Array(ids.prefix(cut)) + Array(ids.suffix(keep))
        }
        var out: [Int32] = []
        out.reserveCapacity(ids.count + total)
        var spans: [KreaReferenceSpan] = []
        for id in ids {
            guard id == KreaReferenceContract.imageToken else { out.append(id); continue }
            guard spans.count < images.count else {
                throw AetherError.invalidRequest("the rendered prompt has more image pads than its \(images.count) image(s)")
            }
            let image = images[spans.count], grid = image.mergedGrid
            spans.append(KreaReferenceSpan(range: out.count..<(out.count + image.tokens), t: grid.t, h: grid.h, w: grid.w,
                                           digest: image.digest))
            out += [Int32](repeating: KreaReferenceContract.imageToken, count: image.tokens)
        }
        guard spans.count == images.count else {
            throw AetherError.invalidRequest("the rendered prompt has \(spans.count) image pad(s) for \(images.count) image(s)")
        }
        guard let first = spans.first, first.range.lowerBound >= KreaPrompt.leadingRows else {
            throw AetherError.invalidRequest("a reference image renders inside the dropped system rows")
        }
        guard out.count - KreaPrompt.leadingRows <= Self.maximumConditioningRows else {
            throw AetherError.invalidRequest("reference prompt exceeds \(Self.maximumConditioningRows) conditioning rows")
        }
        tokenIDs = out
        self.images = spans
        ropePositions = try Self.positions(count: out.count, spans: spans)
    }

    /// Text tokens get the running position on all three axes. An image span starting at running position s with
    /// merged grid (t, h, w) puts its tokens in raster order at (s + ti, s + hi, s + wi); the running position then
    /// advances to s + max(t, h, w).
    static func positions(count: Int, spans: [KreaReferenceSpan]) throws -> [Int32] {
        var axes = [[Int32]](repeating: [], count: 3)
        for a in 0..<3 { axes[a].reserveCapacity(count) }
        var p: Int32 = 0, i = 0, k = 0
        while i < count {
            if k < spans.count, spans[k].range.lowerBound == i {
                let s = spans[k]
                guard s.t > 0, s.h > 0, s.w > 0, s.range.count == s.t * s.h * s.w, s.range.upperBound <= count else {
                    throw AetherError.invalidRequest("image span \(k) does not fit the prompt")
                }
                for j in 0..<s.range.count {
                    axes[0].append(p + Int32(j / (s.h * s.w)))
                    axes[1].append(p + Int32((j / s.w) % s.h))
                    axes[2].append(p + Int32(j % s.w))
                }
                p += Int32(max(s.t, s.h, s.w))
                i += s.range.count
                k += 1
                continue
            }
            for a in 0..<3 { axes[a].append(p) }
            p += 1
            i += 1
        }
        guard k == spans.count else { throw AetherError.invalidRequest("image spans lie outside the prompt") }
        return axes.flatMap { $0 }
    }

    /// `encodeWithImages` token inputs. `image_embeds` and `deepstack` come from the vision encoder.
    /// `ropeShape` follows the function's declaration: `[3, L]` or `[3, 1, L]`.
    public func encoderInputs(ropeShape: [Int]? = nil) throws -> [String: TensorValue] {
        let n = tokenIDs.count
        let shape = ropeShape ?? [3, n]
        guard shape == [3, n] || shape == [3, 1, n] else {
            throw AetherError.invalidModel("unsupported rope_positions declaration \(shape)")
        }
        return [
            "input_ids": .int32(try Int32Tensor(shape: [1, n], values: tokenIDs)),
            "position_ids": .int32(try Int32Tensor(shape: [1, n], values: (0..<n).map(Int32.init))),
            "attention_mask": .int32(try Int32Tensor(shape: [1, n], values: Array(repeating: 1, count: n))),
            "rope_positions": .int32(try Int32Tensor(shape: shape, values: ropePositions)),
        ]
    }

    /// Text cache key: tokens plus each image's content digest (identical pads around different images differ).
    public var cacheKey: String {
        tokenIDs.map(String.init).joined(separator: ",") + "|" + images.map(\.digest).joined(separator: ",")
    }
}
