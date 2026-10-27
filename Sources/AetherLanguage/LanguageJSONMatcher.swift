/// Where a constrained reply is in its grammar, as a set of parallel stacks (one per live `anyOf` alternative). Fed one
/// byte at a time; a byte no stack accepts is refused. Hashable, so allowed-token sets can be cached by state.
struct LanguageJSONMatcher: Hashable, Sendable {
    enum Frame: Hashable, Sendable {
        /// A value of this node, not started.
        case value(Int)
        /// An object; `member` is the next member to write (members.count: the closing brace).
        case object(node: Int, member: Int)
        /// Fixed text being matched: a member's lead-in, or an object's closing brace.
        case text(Text, position: Int)
        case string(StringPhase)
        /// A choice among literals, `alive` the options still matching.
        case choice(node: Int, position: Int, alive: [Int])
        case number(node: Int, NumberState)
        case array(node: Int, count: Int, phase: ArrayPhase)
    }

    enum Text: Hashable, Sendable {
        case member(node: Int, index: Int)
        case close(node: Int)
    }

    enum StringPhase: Hashable, Sendable {
        /// Inside a string; `pending` continuation bytes are due for the current UTF-8 scalar, the next one in
        /// `low...high` (RFC 3629: no overlong forms, surrogates or code points past U+10FFFF).
        case body(pending: Int, low: UInt8 = 0x80, high: UInt8 = 0xBF)
        case escape
        case unicode(remaining: Int)
    }

    enum ArrayPhase: Hashable, Sendable { case open, afterOpen, afterItem, afterComma }

    private(set) var threads: [[Frame]]

    init(grammar: LanguageJSONGrammar) {
        threads = [[.value(grammar.root)]]
    }

    /// The matcher after `byte`, or nil if no stack accepts it.
    func feeding(_ byte: UInt8, _ grammar: LanguageJSONGrammar) -> LanguageJSONMatcher? {
        var next: [[Frame]] = []
        for thread in threads {
            for result in Self.advance(thread, byte, grammar) where !next.contains(result) { next.append(result) }
        }
        guard !next.isEmpty else { return nil }
        var matcher = self
        matcher.threads = next
        return matcher
    }

    func feeding(_ bytes: some Sequence<UInt8>, _ grammar: LanguageJSONGrammar) -> LanguageJSONMatcher? {
        var matcher = self
        for byte in bytes {
            guard let next = matcher.feeding(byte, grammar) else { return nil }
            matcher = next
        }
        return matcher
    }

    /// Whether the reply can end here: some stack has a complete value (a trailing number may end without a delimiter).
    func canEnd(_ grammar: LanguageJSONGrammar) -> Bool {
        threads.contains { thread in
            guard let top = thread.last else { return true }
            if case .number(let node, let state) = top, thread.count == 1 { return state.canEnd(grammar.nodes[node]) }
            return false
        }
    }

    /// The UTF-8 state inside a string body, when every stack is there with the same parents: the fast path for
    /// allowed tokens (any run of plain string bytes keeps the same parents).
    var stringBodyPending: Int? {
        guard threads.count == 1, case .string(.body(let pending, _, _))? = threads[0].last else { return nil }
        return pending
    }

    private static func advance(_ stack: [Frame], _ byte: UInt8, _ grammar: LanguageJSONGrammar) -> [[Frame]] {
        var stack = stack
        guard let top = stack.popLast() else { return [] }
        switch top {
        case .value(let node):
            switch grammar.nodes[node] {
            case .object: stack.append(.object(node: node, member: 0))
            case .string:
                guard byte == UInt8(ascii: "\"") else { return [] }
                stack.append(.string(.body(pending: 0)))
                return [stack]
            case .choice(let options): stack.append(.choice(node: node, position: 0, alive: Array(options.indices)))
            case .integer, .number: stack.append(.number(node: node, NumberState()))
            case .array: stack.append(.array(node: node, count: 0, phase: .open))
            case .anyOf(let alternatives):
                return alternatives.flatMap { advance(stack + [.value($0)], byte, grammar) }
            }
            return advance(stack, byte, grammar)

        case .object(let node, let member):
            guard case .object(let members) = grammar.nodes[node] else { return [] }
            if member < members.count {
                stack.append(.object(node: node, member: member + 1))
                stack.append(.value(members[member].value))
                stack.append(.text(.member(node: node, index: member), position: 0))
            } else {
                stack.append(.text(.close(node: node), position: 0))
            }
            return advance(stack, byte, grammar)

        case .text(let text, let position):
            let bytes: [UInt8]
            switch text {
            case .member(let node, let index):
                guard case .object(let members) = grammar.nodes[node] else { return [] }
                bytes = members[index].text
            case .close(let node):
                guard case .object(let members) = grammar.nodes[node] else { return [] }
                bytes = Array((members.isEmpty ? "{}" : "}").utf8)
            }
            guard bytes[position] == byte else { return [] }
            if position + 1 < bytes.count { stack.append(.text(text, position: position + 1)) }
            return [stack]

        case .string(let phase):
            switch phase {
            case .body(let pending, let low, let high):
                if pending > 0 {
                    guard byte >= low, byte <= high else { return [] }
                    stack.append(.string(.body(pending: pending - 1)))
                } else if byte == UInt8(ascii: "\"") {
                    // The string ends.
                } else if byte == UInt8(ascii: "\\") {
                    stack.append(.string(.escape))
                } else if byte < 0x20 {
                    return []
                } else if byte < 0x80 {
                    stack.append(.string(.body(pending: 0)))
                } else if let next = Self.continuation(lead: byte) {
                    stack.append(.string(.body(pending: next.count, low: next.low, high: next.high)))
                } else {
                    return []
                }
            case .escape:
                if byte == UInt8(ascii: "u") { stack.append(.string(.unicode(remaining: 4))) }
                else if Array("\"\\/bfnrt".utf8).contains(byte) { stack.append(.string(.body(pending: 0))) }
                else { return [] }
            case .unicode(let remaining):
                guard Self.isHex(byte) else { return [] }
                stack.append(.string(remaining > 1 ? .unicode(remaining: remaining - 1) : .body(pending: 0)))
            }
            return [stack]

        case .choice(let node, let position, let alive):
            guard case .choice(let options) = grammar.nodes[node] else { return [] }
            let matching = alive.filter { position < options[$0].count && options[$0][position] == byte }
            guard !matching.isEmpty else { return [] }
            // Options are prefix-free, so a completed option is the only one left.
            if matching.count == 1, options[matching[0]].count == position + 1 { return [stack] }
            stack.append(.choice(node: node, position: position + 1, alive: matching))
            return [stack]

        case .number(let node, let state):
            let kind = grammar.nodes[node]
            if let next = state.consuming(byte, kind) {
                stack.append(.number(node: node, next))
                return [stack]
            }
            // Any other byte ends the number, if it can end here, and belongs to the parent.
            guard state.canEnd(kind) else { return [] }
            return advance(stack, byte, grammar)

        case .array(let node, let count, let phase):
            guard case .array(let item, let minimum, let maximum) = grammar.nodes[node] else { return [] }
            switch phase {
            case .open:
                guard byte == UInt8(ascii: "[") else { return [] }
                stack.append(.array(node: node, count: 0, phase: .afterOpen))
                return [stack]
            case .afterOpen:
                if byte == UInt8(ascii: "]") { return minimum == 0 ? [stack] : [] }
                guard maximum.map({ $0 > 0 }) ?? true else { return [] }
                stack.append(.array(node: node, count: 1, phase: .afterItem))
                stack.append(.value(item))
                return advance(stack, byte, grammar)
            case .afterItem:
                if byte == UInt8(ascii: ","), maximum.map({ count < $0 }) ?? true {
                    stack.append(.array(node: node, count: count, phase: .afterComma))
                    return [stack]
                }
                return byte == UInt8(ascii: "]") && count >= minimum ? [stack] : []
            case .afterComma:
                guard byte == UInt8(ascii: " ") else { return [] }
                stack.append(.array(node: node, count: count + 1, phase: .afterItem))
                stack.append(.value(item))
                return [stack]
            }
        }
    }

    /// The continuation bytes a UTF-8 lead byte needs and the range of the first one (RFC 3629), or nil for a byte
    /// that can't lead a scalar.
    static func continuation(lead byte: UInt8) -> (count: Int, low: UInt8, high: UInt8)? {
        switch byte {
        case 0xC2...0xDF: (1, 0x80, 0xBF)
        case 0xE0: (2, 0xA0, 0xBF)
        case 0xED: (2, 0x80, 0x9F)
        case 0xE1...0xEF: (2, 0x80, 0xBF)
        case 0xF0: (3, 0x90, 0xBF)
        case 0xF1...0xF3: (3, 0x80, 0xBF)
        case 0xF4: (3, 0x80, 0x8F)
        default: nil
        }
    }

    private static func isHex(_ byte: UInt8) -> Bool {
        (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte) || (UInt8(ascii: "a")...UInt8(ascii: "f")).contains(byte)
            || (UInt8(ascii: "A")...UInt8(ascii: "F")).contains(byte)
    }
}

/// A number being written, and whether it can still land within its bounds: `-?(0|[1-9][0-9]*)(\.[0-9]+)?`, with
/// exact fixed-point arithmetic (integers as is, numbers in billionths), so a prefix is accepted only if some
/// completion within the digit limits is in range, and the reply never reaches a dead end.
struct NumberState: Hashable, Sendable {
    enum Stage: Hashable, Sendable { case start, sign, zero, integer, dot, fraction }
    var stage = Stage.start
    var negative = false
    var whole: Int64 = 0
    var wholeDigits = 0
    var fraction: Int64 = 0
    var fractionDigits = 0

    private static let fractionLimit = 9

    func consuming(_ byte: UInt8, _ kind: LanguageJSONGrammar.Node) -> NumberState? {
        let isNumber: Bool
        let minimum: Int64?, maximum: Int64?
        switch kind {
        case .integer(let low, let high): (isNumber, minimum, maximum) = (false, low, high)
        case .number(let low, let high): (isNumber, minimum, maximum) = (true, low, high)
        default: return nil
        }
        let wholeLimit = isNumber ? 9 : 18
        let digit = byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9") ? Int64(byte - UInt8(ascii: "0")) : nil
        var next = self
        switch (stage, digit, byte) {
        case (.start, _, UInt8(ascii: "-")): next.stage = .sign; next.negative = true
        case (.start, 0?, _), (.sign, 0?, _): next.stage = .zero; next.wholeDigits = 1
        case (.start, let d?, _), (.sign, let d?, _): next.stage = .integer; next.whole = d; next.wholeDigits = 1
        case (.integer, let d?, _) where wholeDigits < wholeLimit: next.whole = whole * 10 + d; next.wholeDigits += 1
        case (.zero, nil, UInt8(ascii: ".")) where isNumber, (.integer, nil, UInt8(ascii: ".")) where isNumber: next.stage = .dot
        case (.dot, let d?, _): next.stage = .fraction; next.fraction = d; next.fractionDigits = 1
        case (.fraction, let d?, _) where fractionDigits < Self.fractionLimit:
            next.fraction = fraction * 10 + d; next.fractionDigits += 1
        default: return nil
        }
        return next.feasible(isNumber: isNumber, minimum: minimum, maximum: maximum) ? next : nil
    }

    func canEnd(_ kind: LanguageJSONGrammar.Node) -> Bool {
        guard stage == .zero || stage == .integer || stage == .fraction else { return false }
        switch kind {
        case .integer(let minimum, let maximum): return Self.within(value(isNumber: false), minimum, maximum)
        case .number(let minimum, let maximum): return Self.within(value(isNumber: true), minimum, maximum)
        default: return false
        }
    }

    /// The value written so far (in billionths for numbers).
    private func value(isNumber: Bool) -> Int64 {
        let scale = isNumber ? LanguageJSONGrammar.numberScale : 1
        var magnitude = whole * scale
        if stage == .fraction { magnitude += fraction * Self.power(Self.fractionLimit - fractionDigits) }
        return negative ? -magnitude : magnitude
    }

    /// Whether some completion within the digit limits lies in [minimum, maximum].
    private func feasible(isNumber: Bool, minimum: Int64?, maximum: Int64?) -> Bool {
        let scale = isNumber ? LanguageJSONGrammar.numberScale : 1
        let wholeLimit = isNumber ? 9 : 18
        var magnitudes: [ClosedRange<Int64>] = []
        switch stage {
        case .start: return true
        case .sign: return Self.overlaps(Int64.min...0, minimum, maximum)
        case .zero: magnitudes = [0...(isNumber ? scale - 1 : 0)]
        case .integer:
            if !isNumber { magnitudes.append(whole...whole) }
            for extra in (isNumber ? 0 : 1)...max(isNumber ? 0 : 1, wholeLimit - wholeDigits) where wholeDigits + extra <= wholeLimit {
                let low = whole * Self.power(extra), high = (whole + 1) * Self.power(extra) - 1
                magnitudes.append(isNumber ? (low * scale)...((high + 1) * scale - 1) : low...high)
            }
        case .dot: magnitudes = [(whole * scale)...((whole + 1) * scale - 1)]
        case .fraction:
            let step = Self.power(Self.fractionLimit - fractionDigits)
            let low = whole * scale + fraction * step
            magnitudes = [low...(low + step - 1)]
        }
        return magnitudes.contains { range in
            Self.overlaps(negative ? (-range.upperBound)...(-range.lowerBound) : range, minimum, maximum)
        }
    }

    private static func overlaps(_ range: ClosedRange<Int64>, _ minimum: Int64?, _ maximum: Int64?) -> Bool {
        range.upperBound >= (minimum ?? .min) && range.lowerBound <= (maximum ?? .max)
    }

    private static func within(_ value: Int64, _ minimum: Int64?, _ maximum: Int64?) -> Bool {
        value >= (minimum ?? .min) && value <= (maximum ?? .max)
    }

    private static func power(_ exponent: Int) -> Int64 {
        var result: Int64 = 1
        for _ in 0..<max(0, exponent) { result *= 10 }
        return result
    }
}
