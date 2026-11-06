// MANIFEST.md §2.4 `tokenizer.text`: how an encoder's texts become fixed-shape id rows (padding, truncation, the
// attention mask, unused batch rows). The manifest keeps the block as JSON (`Manifest.Tokenizer.text`); a driver that
// reads it takes this typed view, which closes the vocabularies and checks the values.

extension Manifest {
    /// `tokenizer.text`'s padding and batching fields. Fields this view doesn't name (`prompts`, `template`) stay in the
    /// JSON for the drivers that read them.
    public struct TokenizerText: Sendable, Hashable, Codable {
        public enum Side: String, Sendable, Hashable, Codable {
            case right, left
        }

        public enum Truncation: String, Sendable, Hashable, Codable {
            /// Pieces past the limit are dropped from the end; the closing special tokens stay.
            case right
            /// At a word boundary.
            case word
            /// Only the template's body is cut (Shieldstral).
            case bodyOnly
        }

        /// Which positions of a row the graph's `attentionMask` marks 1.
        public enum MaskRule: String, Sendable, Hashable, Codable {
            /// `"none"`: the graph takes no mask (CLIP: causal attention pooled at the first EOS makes one redundant).
            case noMask = "none"
            /// Every id that isn't ``TokenizerText/padId``.
            case nonPad
            /// Everything up to and including the first EOS (a pad id equal to EOS, as CLIP's).
            case throughFirstEos
        }

        /// The id rows are padded with.
        public var padId: Int
        /// Default `right`.
        public var padSide: Side?
        /// The row length every text is padded (and truncated) to.
        public var padTo: Int?
        /// Default `right`.
        public var truncation: Truncation?
        /// Default `none` (``MaskRule/noMask``).
        public var maskRule: MaskRule?
        /// A static batch's rows beyond the texts: these ids, then ``padId`` to ``padTo`` (CLIP: `[BOS, EOS]`). Their
        /// outputs are discarded. Absent: rows of ``padId`` only.
        public var unusedBatchRows: [Int]?

        public init(
            padId: Int, padSide: Side? = nil, padTo: Int? = nil, truncation: Truncation? = nil, maskRule: MaskRule? = nil,
            unusedBatchRows: [Int]? = nil
        ) {
            self.padId = padId
            self.padSide = padSide
            self.padTo = padTo
            self.truncation = truncation
            self.maskRule = maskRule
            self.unusedBatchRows = unusedBatchRows
        }

        func validate() throws(AetherError) {
            try require(padId >= 0, "tokenizer.text.padId", "must be a token id (>= 0)")
            if let padTo { try require(padTo >= 1, "tokenizer.text.padTo", "must be at least 1") }
            if let rows = unusedBatchRows {
                try require(rows.allSatisfy { $0 >= 0 }, "tokenizer.text.unusedBatchRows", "must be token ids (>= 0)")
                if let padTo { try require(rows.count <= padTo, "tokenizer.text.unusedBatchRows", "must fit in padTo (\(padTo))") }
            }
        }
    }
}

extension Manifest.Tokenizer {
    /// The typed view of ``text``, or nil when the manifest has no `text` block. Errors name the field under
    /// `tokenizer.text`.
    public func textBlock() throws(AetherError) -> Manifest.TokenizerText? {
        guard let text, text != .null else { return nil }
        let block: Manifest.TokenizerText
        do {
            block = try decodeSanitized(Manifest.TokenizerText.self, from: canonicalJSON(text))
        } catch {
            if case .manifestInvalid(let field, let reason) = error {
                throw .manifestInvalid(field: field.isEmpty ? "tokenizer.text" : "tokenizer.text.\(field)", reason: reason)
            }
            throw error
        }
        try block.validate()
        return block
    }
}
