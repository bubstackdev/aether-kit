// The `krea2` block that `krea2@1` owns (MANIFEST.md §5.6). The driver implements exactly the values typed here; any
// other value is refused when the manifest decodes, naming its field.

extension Manifest {
    /// `krea2` (MANIFEST.md §5.6): Krea 2's prompt rule, text conditioning, image rotary positions, segment order and
    /// the content-filter requirement of its license.
    public struct Krea2: Sendable, Hashable, Codable {
        /// The text encoder's input: `systemPrefix + prompt + suffix`, tokenized without added specials, never padded.
        public struct Prompt: Sendable, Hashable, Codable {
            public enum Padding: String, Sendable, Hashable, Codable {
                /// The rendered ids are the whole input: no padding rows, so every row is a real one.
                case unpadded = "none"
            }
            public enum Positions: String, Sendable, Hashable, Codable {
                /// `position_ids` are `0 ..< n`.
                case running
            }

            public var systemPrefix: String
            public var suffix: String
            /// At most this many rendered ids; a longer prompt keeps its first `maxTokens − keepTrailingTokens` and its
            /// last `keepTrailingTokens` (the suffix's).
            public var maxTokens: Int
            public var keepTrailingTokens: Int
            /// The text encoder drops this many leading rows (the system prefix) from its output.
            public var dropLeadingRows: Int
            public var padding: Padding
            public var positions: Positions

            public init(
                systemPrefix: String, suffix: String, maxTokens: Int, keepTrailingTokens: Int, dropLeadingRows: Int,
                padding: Padding = .unpadded, positions: Positions = .running
            ) {
                self.systemPrefix = systemPrefix
                self.suffix = suffix
                self.maxTokens = maxTokens
                self.keepTrailingTokens = keepTrailingTokens
                self.dropLeadingRows = dropLeadingRows
                self.padding = padding
                self.positions = positions
            }
        }

        /// The text encoder's output: per real row after the dropped prefix, the hidden states after layers `taps`,
        /// tap-major, `taps.count · width` wide.
        public struct Conditioning: Sendable, Hashable, Codable {
            public var taps: [Int]
            public var width: Int
            /// The row-count range, `"lower...upper"`.
            public var rows: String

            public init(taps: [Int], width: Int, rows: String) {
                self.taps = taps
                self.width = width
                self.rows = rows
            }

            /// Columns per conditioning row.
            public var columns: Int { taps.count * width }

            /// ``rows`` as a range, nil unless it is `"a...b"` with `1 ≤ a ≤ b`.
            public var rowRange: ClosedRange<Int>? {
                let parts = rows.split(separator: ".", omittingEmptySubsequences: false)
                guard parts.count == 4, parts[1].isEmpty, parts[2].isEmpty, let a = Int(parts[0]), let b = Int(parts[3]),
                      a >= 1, a <= b else { return nil }
                return a...b
            }
        }

        /// The host-built image rotary table `[n, 2, Σ axes]` (cos rows, then sin rows). Text rows are not rotated.
        public struct Rotary: Sendable, Hashable, Codable {
            public enum Layout: String, Sendable, Hashable, Codable {
                /// Pair `i` rotates dimensions `i` and `i + Σ axes / 2`.
                case splitHalves
            }

            /// The position-id rule the driver implements.
            public static let imagePositions = "(0, row, column)"

            public var axes: [Int]
            public var theta: Double
            public var positions: String
            public var layout: Layout
            public var textRotated: Bool
            public var dtype: ScalarType

            public init(
                axes: [Int], theta: Double, positions: String = Rotary.imagePositions, layout: Layout = .splitHalves,
                textRotated: Bool = false, dtype: ScalarType = .float32
            ) {
                self.axes = axes
                self.theta = theta
                self.positions = positions
                self.layout = layout
                self.textRotated = textRotated
                self.dtype = dtype
            }

            /// Σ axes: the rotary width per head.
            public var dimension: Int { axes.reduce(0, +) }
        }

        /// The Krea 2 Community License (§4.2) requires a deployment to filter output: the driver takes the app's
        /// `ContentFilter` and has no default.
        public struct ContentFilterRequirement: Sendable, Hashable, Codable {
            public var required: Bool
            public var reason: String?

            public init(required: Bool = true, reason: String? = nil) {
                self.required = required
                self.reason = reason
            }
        }

        public var prompt: Prompt
        public var conditioning: Conditioning
        public var rotary: Rotary
        /// The denoiser graph roles, in call order.
        public var segments: [String]
        /// Rows of the modulation tensor the first segment emits.
        public var condRows: Int
        public var contentFilter: ContentFilterRequirement

        public init(
            prompt: Prompt, conditioning: Conditioning, rotary: Rotary, segments: [String], condRows: Int,
            contentFilter: ContentFilterRequirement = ContentFilterRequirement()
        ) {
            self.prompt = prompt
            self.conditioning = conditioning
            self.rotary = rotary
            self.segments = segments
            self.condRows = condRows
            self.contentFilter = contentFilter
        }

        func validate(graphs: [String: Graph]) throws(AetherError) {
            try require(prompt.dropLeadingRows >= 0, "krea2.prompt.dropLeadingRows", "must be >= 0")
            try require(prompt.keepTrailingTokens >= 0, "krea2.prompt.keepTrailingTokens", "must be >= 0")
            try require(prompt.maxTokens > prompt.dropLeadingRows + prompt.keepTrailingTokens, "krea2.prompt.maxTokens",
                        "must exceed dropLeadingRows + keepTrailingTokens")
            try require(!conditioning.taps.isEmpty, "krea2.conditioning.taps", "must name at least one layer")
            try require(conditioning.width >= 1, "krea2.conditioning.width", "must be >= 1")
            guard let range = conditioning.rowRange else {
                throw .manifestInvalid(field: "krea2.conditioning.rows", reason: "must be \"lower...upper\" with 1 <= lower <= upper")
            }
            try require(range.upperBound == prompt.maxTokens - prompt.dropLeadingRows, "krea2.conditioning.rows",
                        "the upper bound must be maxTokens - dropLeadingRows")
            try require(rotary.axes.count == 3 && rotary.axes.allSatisfy { $0 >= 2 && $0 % 2 == 0 }, "krea2.rotary.axes",
                        "must be three positive even dimensions (the positions have three axes)")
            try require(rotary.theta > 0, "krea2.rotary.theta", "must be > 0")
            try require(rotary.positions == Rotary.imagePositions, "krea2.rotary.positions", "must be \(Rotary.imagePositions)")
            try require(!rotary.textRotated, "krea2.rotary.textRotated", "only unrotated text rows are implemented")
            try require(rotary.dtype == .float32, "krea2.rotary.dtype", "the host table is float32")
            try require(!segments.isEmpty && Set(segments).count == segments.count, "krea2.segments",
                        "must list distinct denoiser roles")
            for (i, role) in segments.enumerated() {
                try require(graphs[role] != nil, "krea2.segments.\(i)", "names graph '\(role)', which is not in graphs")
            }
            try require(condRows >= 1, "krea2.condRows", "must be >= 1")
            try require(contentFilter.required, "krea2.contentFilter.required",
                        "must be true: the Krea 2 Community License (§4.2) requires a content filter")
        }
    }
}
