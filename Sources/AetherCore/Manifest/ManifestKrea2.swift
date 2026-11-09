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
            public var rowRange: ClosedRange<Int>? { Self.range(rows) }

            static func range(_ rows: String) -> ClosedRange<Int>? {
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

        /// Reference images (MANIFEST.md §5.6 `krea2.references`): the 0.0.x `visionTokens` contract. Images become extra
        /// text-encoder rows: each reference is embedded by the `vision` graph and the text encoder's `references`
        /// function takes them at the prompt's image-pad positions. Absent: text only.
        public struct References: Sendable, Hashable, Codable {
            public struct Tokens: Sendable, Hashable, Codable {
                public var pad: Int32
                public var start: Int32
                public var end: Int32
            }

            /// How a reference image becomes patch rows.
            public struct Preprocess: Sendable, Hashable, Codable {
                public static let fitRounded = "longSideRounded"
                public static let keysBicubic = "keysBicubic-0.5-antialias"
                public static let mergeWindowOrder = "mergeWindow C,T,py,px"

                /// `longSideRounded`: scale the longer side down to `maxSide`, the other side rounded (never up).
                public var fit: String
                public var maxSide: Int
                /// Smart resize: each side a multiple of `factor` (ties to even), the area within the pixel bounds.
                public var factor: Int
                public var minPixels: Int
                public var maxPixels: Int
                public var maxAspect: Double
                /// `keysBicubic-0.5-antialias`: one separable antialiased Keys cubic resample from the source.
                public var filter: String
                /// `(v − mean) / std` per 8-bit value.
                public var mean: Double
                public var std: Double
                public var patch: Int
                public var temporalPatch: Int
                public var merge: Int
                /// `mergeWindow C,T,py,px`: rows over merge windows in raster order, each `[C][T][py][px]`.
                public var order: String

                /// Values per patch row.
                public var rowLength: Int { 3 * temporalPatch * patch * patch }
            }

            public var maxImages: Int
            public var maxSide: Int
            /// Image tokens (merged patches) over all of a prompt's references.
            public var maxImageTokens: Int
            /// Each image part's label, `{n}` its 1-based number: `"Picture {n}: "`.
            public var label: String
            public var tokens: Tokens
            /// Image-pad rows are fed to the text encoder as `extensionBase + slot` (slot 0… across the images).
            public var extensionBase: Int
            /// Conditioning rows a reference prompt may have, `"lower...upper"`.
            public var rows: String
            /// `mrope`: text at the running position on three axes; an image's tokens at `(p + t, p + h, p + w)`.
            public var positions: String
            public var deepstackLevels: Int
            /// The vision graph role and the text encoder's phase for references.
            public var vision: String
            public var function: String
            public var preprocess: Preprocess

            /// ``rows`` as a range, nil unless it is `"a...b"` with `1 ≤ a ≤ b`.
            public var rowRange: ClosedRange<Int>? { Conditioning.range(rows) }

            /// `label` for image `number` (1-based).
            public func label(_ number: Int) -> String { label.split(separator: "{n}", omittingEmptySubsequences: false).joined(separator: String(number)) }

            func validate(graphs: [String: Graph], conditioning: Conditioning) throws(AetherError) {
                try require(maxImages >= 1, "krea2.references.maxImages", "must be >= 1")
                try require(maxImageTokens >= 1, "krea2.references.maxImageTokens", "must be >= 1")
                try require(label.contains("{n}"), "krea2.references.label", "must contain {n}")
                try require(Set([tokens.pad, tokens.start, tokens.end]).count == 3 && tokens.pad >= 0,
                            "krea2.references.tokens", "must be three distinct ids")
                try require(extensionBase > Int(max(tokens.pad, tokens.start, tokens.end)), "krea2.references.extensionBase",
                            "must lie above the vocabulary ids")
                guard let range = rowRange, let text = conditioning.rowRange else {
                    throw .manifestInvalid(field: "krea2.references.rows", reason: "must be \"lower...upper\" with 1 <= lower <= upper")
                }
                try require(range.upperBound >= text.upperBound, "krea2.references.rows",
                            "must reach at least the text-only rows (\(text.upperBound))")
                try require(positions == "mrope", "krea2.references.positions", "only mrope is implemented")
                try require(deepstackLevels >= 0, "krea2.references.deepstackLevels", "must be >= 0")
                try require(graphs[vision] != nil, "krea2.references.vision", "names graph '\(vision)', which is not in graphs")
                try require(graphs["textEncoder"]?.phases[function] != nil, "krea2.references.function",
                            "the text encoder has no '\(function)' function")
                let p = preprocess
                try require(p.fit == Preprocess.fitRounded, "krea2.references.preprocess.fit", "only \(Preprocess.fitRounded) is implemented")
                try require(p.filter == Preprocess.keysBicubic, "krea2.references.preprocess.filter",
                            "only \(Preprocess.keysBicubic) is implemented")
                try require(p.order == Preprocess.mergeWindowOrder, "krea2.references.preprocess.order",
                            "only \(Preprocess.mergeWindowOrder) is implemented")
                try require(p.patch >= 1 && p.merge >= 1 && p.temporalPatch >= 1 && p.factor == p.patch * p.merge,
                            "krea2.references.preprocess", "factor must be patch · merge")
                try require(p.maxSide >= p.factor && p.minPixels >= 1 && p.minPixels <= p.maxPixels && p.maxAspect >= 1 && p.std > 0,
                            "krea2.references.preprocess", "sizes, pixel bounds, aspect or std out of range")
            }
        }

        public var prompt: Prompt
        public var conditioning: Conditioning
        public var rotary: Rotary
        public var references: References? = nil
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
            try references?.validate(graphs: graphs, conditioning: conditioning)
            try require(contentFilter.required, "krea2.contentFilter.required",
                        "must be true: the Krea 2 Community License (§4.2) requires a content filter")
        }
    }
}
