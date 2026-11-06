// The `kokoro` block that `kokoro@1` owns (MANIFEST.md §5.9). The driver implements exactly the rules typed here; any
// other value is refused when the manifest decodes, naming its field.

extension Manifest {
    /// `kokoro` (MANIFEST.md §5.9): Kokoro-82M's phoneme alphabet, voice-style rows, duration rounding, decoder frame
    /// buckets, harmonic-source noise and voices.
    public struct Kokoro: Sendable, Hashable, Codable {
        /// The phoneme alphabet: single Unicode scalars to token ids, framed by the boundary token.
        public struct Phonemes: Sendable, Hashable, Codable {
            public enum Normalization: String, Sendable, Hashable, Codable {
                /// Scalars are taken as given: `a` + U+0303 and precomposed `ã` are different symbols.
                case none
            }
            public enum Unknown: String, Sendable, Hashable, Codable {
                /// A scalar outside the vocabulary is an error (upstream silently drops it).
                case refuse
            }

            public var vocabulary: [String: Int]
            public var boundaryToken: Int
            public var maxPhonemes: Int
            public var normalization: Normalization
            public var unknown: Unknown

            public init(vocabulary: [String: Int], boundaryToken: Int = 0, maxPhonemes: Int = 510,
                        normalization: Normalization = .none, unknown: Unknown = .refuse) {
                self.vocabulary = vocabulary
                self.boundaryToken = boundaryToken
                self.maxPhonemes = maxPhonemes
                self.normalization = normalization
                self.unknown = unknown
            }
        }

        /// A voice is a `[maxPhonemes, dimension]` table; the row is chosen by the phoneme count. Both graphs take the
        /// whole row (the decoder reads its first half, the prosody predictor its second).
        public struct Style: Sendable, Hashable, Codable {
            public enum Row: String, Sendable, Hashable, Codable {
                case phonemeCountMinusOne = "phonemeCount - 1"
            }

            public var dimension: Int
            public var row: Row
            public var decoder: String?
            public var prosody: String?

            public init(dimension: Int = 256, row: Row = .phonemeCountMinusOne, decoder: String? = nil, prosody: String? = nil) {
                self.dimension = dimension
                self.row = row
                self.decoder = decoder
                self.prosody = prosody
            }
        }

        /// How predicted per-token durations become integer frames.
        public struct Durations: Sendable, Hashable, Codable {
            public enum Frames: String, Sendable, Hashable, Codable {
                /// torch.round's rule, in float32, at least one frame per token.
                case roundHalfToEven = "max(1, roundHalfToEven(duration / speed))"
            }
            public enum Alignment: String, Sendable, Hashable, Codable {
                /// Token `t`'s column repeated `frames[t]` times (upstream's one-hot alignment matmul).
                case `repeat`
            }

            public var frames: Frames
            public var alignment: Alignment

            public init(frames: Frames = .roundHalfToEven, alignment: Alignment = .repeat) {
                self.frames = frames
                self.alignment = alignment
            }
        }

        /// The decoder's frame buckets: inputs are zero-padded to the smallest bucket that fits, `frames` carries the
        /// valid count, and audio past it is dropped. `cap` is the default longest utterance per platform.
        public struct Frames: Sendable, Hashable, Codable {
            public var buckets: [Int]
            public var maxFrames: Int
            /// Platform (`macos`, `ios`) → the default frame cap.
            public var cap: [String: Int]
            public var padding: String?

            public init(buckets: [Int], maxFrames: Int, cap: [String: Int], padding: String? = nil) {
                self.buckets = buckets
                self.maxFrames = maxFrames
                self.cap = cap
                self.padding = padding
            }

            /// The smallest bucket holding `frames`, or nil past the largest.
            public func bucket(for frames: Int) -> Int? { buckets.first { $0 >= frames } }
        }

        /// The vocoder's harmonic-source noise: the seeded stream, `[1, bucketFrames, samplesPerFrame, harmonics]`
        /// row-major, extended to the bucket so the valid prefix matches an unpadded run.
        public struct Noise: Sendable, Hashable, Codable {
            public enum Order: String, Sendable, Hashable, Codable { case rowMajor }

            public var kind: Latent.Noise.Kind
            public var shape: String?
            public var order: Order
            public var dtype: ScalarType

            public init(kind: Latent.Noise.Kind = .splitmix64BoxMuller, shape: String? = nil, order: Order = .rowMajor,
                        dtype: ScalarType = .float32) {
                self.kind = kind
                self.shape = shape
                self.order = order
                self.dtype = dtype
            }
        }

        public struct Voice: Sendable, Hashable, Codable {
            public enum Gender: String, Sendable, Hashable, Codable { case female, male }

            /// A `tables` key.
            public var table: String
            /// BCP 47 (`en-US`, `ja`, …).
            public var language: String
            public var gender: Gender

            public init(table: String, language: String, gender: Gender) {
                self.table = table
                self.language = language
                self.gender = gender
            }
        }

        public var sampleRate: Int
        public var samplesPerFrame: Int
        public var harmonics: Int
        public var phonemes: Phonemes
        public var style: Style
        public var durations: Durations
        public var frames: Frames
        public var noise: Noise
        /// Voice identifier (`af_heart`) → its table.
        public var voices: [String: Voice]

        public init(sampleRate: Int = 24_000, samplesPerFrame: Int = 600, harmonics: Int = 9, phonemes: Phonemes,
                    style: Style = Style(), durations: Durations = Durations(), frames: Frames, noise: Noise = Noise(),
                    voices: [String: Voice]) {
            self.sampleRate = sampleRate
            self.samplesPerFrame = samplesPerFrame
            self.harmonics = harmonics
            self.phonemes = phonemes
            self.style = style
            self.durations = durations
            self.frames = frames
            self.noise = noise
            self.voices = voices
        }

        func validate(tables: [String: Table]) throws(AetherError) {
            try require((1...768_000).contains(sampleRate), "kokoro.sampleRate", "must be in 1...768000")
            try require(samplesPerFrame >= 1, "kokoro.samplesPerFrame", "must be >= 1")
            try require(harmonics >= 1, "kokoro.harmonics", "must be >= 1")

            var ids = Set<Int>()
            for key in phonemes.vocabulary.keys.sorted() {
                let id = phonemes.vocabulary[key]!
                try require(key.unicodeScalars.count == 1, "kokoro.phonemes.vocabulary.\(key)", "must be one Unicode scalar")
                try require(id != phonemes.boundaryToken && (1...Int(Int32.max)).contains(id) && ids.insert(id).inserted,
                            "kokoro.phonemes.vocabulary.\(key)", "ids must be distinct, positive and not the boundary token")
            }
            try require(!phonemes.vocabulary.isEmpty, "kokoro.phonemes.vocabulary", "must not be empty")
            try require(phonemes.boundaryToken >= 0, "kokoro.phonemes.boundaryToken", "must be a token id")
            try require(phonemes.maxPhonemes >= 1, "kokoro.phonemes.maxPhonemes", "must be >= 1")
            try require(style.dimension >= 1, "kokoro.style.dimension", "must be >= 1")

            try require(!frames.buckets.isEmpty && frames.buckets.allSatisfy { $0 >= 1 }
                        && zip(frames.buckets, frames.buckets.dropFirst()).allSatisfy { $0 < $1 },
                        "kokoro.frames.buckets", "must be increasing positive frame counts")
            try require(frames.maxFrames == frames.buckets.last, "kokoro.frames.maxFrames", "must equal the largest bucket")
            for key in frames.cap.keys.sorted() {
                try require(["macos", "ios"].contains(key), "kokoro.frames.cap.\(key)", "platforms are macos and ios")
                try require((1...frames.maxFrames).contains(frames.cap[key]!), "kokoro.frames.cap.\(key)",
                            "must be in 1...\(frames.maxFrames)")
            }
            try require(noise.dtype == .float32, "kokoro.noise.dtype", "the noise is float32")

            try require(!voices.isEmpty, "kokoro.voices", "must name at least one voice")
            for id in voices.keys.sorted() {
                let voice = voices[id]!
                try require(Self.isVoiceIdentifier(id), "kokoro.voices.\(id)", "must be <language><gender>_<name>")
                try require(!voice.language.isEmpty, "kokoro.voices.\(id).language", "must be a BCP 47 tag")
                guard let table = tables[voice.table] else {
                    throw .manifestInvalid(field: "kokoro.voices.\(id).table", reason: "names table '\(voice.table)', which is not in tables")
                }
                try require(table.dtype == .float32 && table.shape == [phonemes.maxPhonemes, style.dimension],
                            "tables.\(voice.table)", "a voice is float32 [\(phonemes.maxPhonemes), \(style.dimension)]")
            }
        }

        /// `af_heart`: a language letter, a gender letter, `_`, then lowercase letters, digits or `_`.
        public static func isVoiceIdentifier(_ id: String) -> Bool {
            let s = Array(id.unicodeScalars)
            guard (4...64).contains(s.count), s[2] == "_", ("a"..."z").contains(s[0]), s[1] == "f" || s[1] == "m" else { return false }
            return s[3...].allSatisfy { ("a"..."z").contains($0) || ("0"..."9").contains($0) || $0 == "_" }
        }
    }
}
