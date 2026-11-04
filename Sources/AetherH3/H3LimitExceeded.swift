/// An H3 request past a fixed limit of the profile or the admission, refused before any stage runs. Each case carries
/// the limit and the request's own numbers, split so a caller can tell what the prompt contributes from what the clip
/// (canvas, frames, audio and keyframes) needs alone; callers never parse ``description``.
@nonexhaustive public enum H3LimitExceeded: Error, Sendable, Equatable, CustomStringConvertible {
    /// The prompt has more UTF-8 bytes than ``H3ResourceAdmission/maximumPromptUTF8Bytes``.
    case promptBytes(limit: Int, bytes: Int)
    /// The conditioner's presentation, `prompt + keyframes` tokens, is outside the profile's text-token limits
    /// (`H3Profile.textTokenLimits`). `keyframes` is what the keyframes' labels and vision blocks add: 0 for T2VA. A
    /// shorter prompt fits only when `keyframes` alone is within the limit.
    case presentationTokens(limit: ClosedRange<Int>, prompt: Int, keyframes: Int)
    /// The packed rows, `clip + prompt`, are outside the denoiser's row limits (`H3Profile.rowLimits`). `prompt` is the
    /// prompt's own tokens; `clip` is every other row: target video, audio, the keyframes' condition video and their
    /// presentation tokens. A prompt has at least one token, so no prompt fits when `clip >= limit.upperBound`: the
    /// size and length alone don't fit.
    case packedRows(limit: ClosedRange<Int>, clip: Int, prompt: Int)

    public var description: String {
        switch self {
        case .promptBytes(let limit, let bytes): "H3 prompt exceeds \(limit) UTF-8 bytes (it has \(bytes))"
        case .presentationTokens(let limit, let prompt, let keyframes):
            "H3 presentation has \(prompt + keyframes) tokens (prompt \(prompt)); the profile supports \(limit)"
        case .packedRows(let limit, let clip, let prompt):
            "H3 clip needs \(clip + prompt) packed rows (\(clip) for the clip, \(prompt) for the prompt); the profile supports \(limit)"
        }
    }
}
