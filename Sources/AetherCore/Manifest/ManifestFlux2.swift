// The `flux2` block that `flux2-klein@1` owns (MANIFEST.md §5.7). The driver implements exactly the values typed here;
// any other value is refused when the manifest decodes, naming its field.

extension Manifest {
    /// `flux2` (MANIFEST.md §5.7): FLUX.2 [klein]'s prompt rule, text conditioning, rotary positions and segment order.
    public struct Flux2: Sendable, Hashable, Codable {
        /// The text encoder's input: the bundle's chat template over one user message, padded to `maxTokens`.
        public struct Prompt: Sendable, Hashable, Codable {
            public enum Template: String, Sendable, Hashable, Codable { case chat }
            public enum Thinking: String, Sendable, Hashable, Codable { case off }
            public enum Padding: String, Sendable, Hashable, Codable { case right }
            public enum Truncation: String, Sendable, Hashable, Codable {
                /// A longer prompt keeps its first `maxTokens` tokens (the tokenizer's right truncation).
                case keepPrefix
            }
            public enum Positions: String, Sendable, Hashable, Codable {
                /// `position_ids` are `0 ..< maxTokens`, padding included.
                case running
            }

            public var template: Template
            public var thinking: Thinking
            public var addGenerationPrompt: Bool
            public var maxTokens: Int
            public var padding: Padding
            public var padToken: Int
            public var truncation: Truncation
            public var positions: Positions

            public init(
                template: Template = .chat, thinking: Thinking = .off, addGenerationPrompt: Bool = true, maxTokens: Int,
                padding: Padding = .right, padToken: Int, truncation: Truncation = .keepPrefix, positions: Positions = .running
            ) {
                self.template = template
                self.thinking = thinking
                self.addGenerationPrompt = addGenerationPrompt
                self.maxTokens = maxTokens
                self.padding = padding
                self.padToken = padToken
                self.truncation = truncation
                self.positions = positions
            }
        }

        /// The text encoder's output rows: hidden states after layers `taps`, concatenated to `width` per row.
        public struct Conditioning: Sendable, Hashable, Codable {
            public enum PaddingRows: String, Sendable, Hashable, Codable {
                /// Every row conditions the denoiser, padding rows included: they can't be trimmed.
                case conditioning
            }

            public var taps: [Int]
            public var width: Int
            public var rows: Int
            public var paddingRows: PaddingRows

            public init(taps: [Int], width: Int, rows: Int, paddingRows: PaddingRows = .conditioning) {
                self.taps = taps
                self.width = width
                self.rows = rows
                self.paddingRows = paddingRows
            }
        }

        /// The host-built rotary tables `[n, 2, Σ axes]` (cos rows, then sin rows).
        public struct Rotary: Sendable, Hashable, Codable {
            public enum Layout: String, Sendable, Hashable, Codable {
                /// Pair `i` rotates dimensions `i` and `i + Σ axes / 2`.
                case splitHalves
            }

            /// The position-id rules the driver implements.
            public static let textIDs = "(0, 0, 0, i)"
            public static let imageIDs = "(0, row, column, 0)"

            public var axes: [Int]
            public var theta: Double
            public var textIds: String
            public var imageIds: String
            /// Reserved for the reference-image path.
            public var referenceIds: String?
            public var layout: Layout
            public var dtype: ScalarType

            public init(
                axes: [Int], theta: Double, textIds: String = Rotary.textIDs, imageIds: String = Rotary.imageIDs,
                referenceIds: String? = nil, layout: Layout = .splitHalves, dtype: ScalarType = .float32
            ) {
                self.axes = axes
                self.theta = theta
                self.textIds = textIds
                self.imageIds = imageIds
                self.referenceIds = referenceIds
                self.layout = layout
                self.dtype = dtype
            }

            /// Σ axes: the rotary width per head.
            public var dimension: Int { axes.reduce(0, +) }
        }

        public var prompt: Prompt
        public var conditioning: Conditioning
        public var rotary: Rotary
        /// The denoiser graph roles, in call order.
        public var segments: [String]
        /// Rows of the shared modulation tensor the first segment emits.
        public var condRows: Int
        /// Reference images (the VAE encoder path): reserved, not implemented by the driver yet.
        public var references: JSONValue?

        public init(prompt: Prompt, conditioning: Conditioning, rotary: Rotary, segments: [String], condRows: Int,
                    references: JSONValue? = nil) {
            self.prompt = prompt
            self.conditioning = conditioning
            self.rotary = rotary
            self.segments = segments
            self.condRows = condRows
            self.references = references
        }

        func validate(graphs: [String: Graph]) throws(AetherError) {
            try require(prompt.maxTokens >= 1, "flux2.prompt.maxTokens", "must be >= 1")
            try require(prompt.padToken >= 0 && prompt.padToken <= Int(Int32.max), "flux2.prompt.padToken",
                        "must be a token id")
            try require(!conditioning.taps.isEmpty, "flux2.conditioning.taps", "must name at least one layer")
            try require(conditioning.width >= 1, "flux2.conditioning.width", "must be >= 1")
            try require(conditioning.rows == prompt.maxTokens, "flux2.conditioning.rows",
                        "must equal flux2.prompt.maxTokens (every padded row conditions)")
            try require(rotary.axes.count == 4 && rotary.axes.allSatisfy { $0 >= 2 && $0 % 2 == 0 }, "flux2.rotary.axes",
                        "must be four positive even dimensions (the position ids have four axes)")
            try require(rotary.theta > 0, "flux2.rotary.theta", "must be > 0")
            try require(rotary.textIds == Rotary.textIDs, "flux2.rotary.textIds", "must be \(Rotary.textIDs)")
            try require(rotary.imageIds == Rotary.imageIDs, "flux2.rotary.imageIds", "must be \(Rotary.imageIDs)")
            try require(rotary.dtype == .float32, "flux2.rotary.dtype", "the host tables are float32")
            try require(!segments.isEmpty && Set(segments).count == segments.count, "flux2.segments",
                        "must list distinct denoiser roles")
            for (i, role) in segments.enumerated() {
                try require(graphs[role] != nil, "flux2.segments.\(i)", "names graph '\(role)', which is not in graphs")
            }
            try require(condRows >= 1, "flux2.condRows", "must be >= 1")
        }
    }
}
