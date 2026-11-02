@testable import AetherLanguage
import AetherLanguageRuntime
import AetherLanguageTokenizers
import Foundation
import FoundationModels
import Testing

private let environment = ProcessInfo.processInfo.environment

/// A synthetic byte-level vocabulary: every single byte, then multi-byte tokens that cross JSON structure (`", "`,
/// `{"`, `"]`), whole words, digits runs, a scalar split across two tokens, and two special tokens.
enum JSONVocabulary {
    static let pieces: [[UInt8]] = (0...255).map { [UInt8($0)] } + [
        "\", \"", "\": ", "{\"", "\"}", "\"]", "[\"", ", ", "true", "false", "null", "12", "3.", ".5", "00", "-1",
        "hello", " world", "red", "ed\"", "\"red", "name", "size", "\": \"", "}, {", "\\n", "\\u00", "é", "日本",
    ].map { Array($0.utf8) } + [[0x97, 0xA5], [0xF0, 0x9F], [0x8C, 0x99]]  // (lone lead bytes are already single-byte tokens)
    static let stop = pieces.count, other = pieces.count + 1
    static let headRows = pieces.count + 4

    static let bytes: LanguageTokenBytes = {
        let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var shifted: [Int: Int] = [:]
        var next = 256
        for byte in 0...255 where !printable.contains(byte) { shifted[byte] = next; next += 1 }
        func spelled(_ bytes: [UInt8]) -> String {
            String(String.UnicodeScalarView(bytes.map { Unicode.Scalar(UInt32(shifted[Int($0)] ?? Int($0)))! }))
        }
        var vocab: [String: Int] = [:]
        for (id, piece) in pieces.enumerated() { vocab[spelled(piece)] = id }
        let json: [String: Any] = ["model": ["vocab": vocab], "decoder": ["type": "ByteLevel"], "added_tokens": [
            ["id": stop, "content": "<|im_end|>", "special": true], ["id": other, "content": "<|x|>", "special": true],
        ]]
        return try! LanguageTokenBytes(tokenizerJSON: try! JSONSerialization.data(withJSONObject: json))
    }()

    static let vocabulary = LanguageJSONVocabulary(bytes: bytes, headRows: headRows, excluded: [])
}

struct Layer: AetherGenerable, Equatable {
    var item: String
    var warm: Bool
    static var generationSchema: AetherGenerationSchema {
        .object(name: "Layer", properties: [.init(name: "item", type: String.self), .init(name: "warm", type: Bool.self)])
    }
}

enum Colour: String, AetherGenerable {
    case red, green, blue
    static var generationSchema: AetherGenerationSchema { .enumeration(name: "Colour", cases: ["red", "green", "blue"]) }
}

struct Outfit: AetherGenerable, Equatable {
    var name: String
    var size: Int
    var price: Double
    var colours: [Colour]
    var layers: [Layer]
    var note: String?
    var grade: String
    static var generationSchema: AetherGenerationSchema {
        .object(name: "Outfit", description: "An outfit", properties: [
            .init(name: "name", type: String.self),
            .init(name: "size", type: Int.self, guides: [.range(1...10)]),
            .init(name: "price", type: Double.self, guides: [.range(0.5...100)]),
            .init(name: "colours", type: [Colour].self, guides: [.count(1...3)]),
            .init(name: "layers", type: [Layer].self, guides: [.maximumCount(2)]),
            .init(name: "note", type: String?.self),
            .init(name: "grade", type: String.self, guides: [.anyOf(["A", "B"])]),
        ])
    }
}

private func grammar(_ schema: AetherGenerationSchema) throws -> LanguageJSONGrammar { try LanguageJSONGrammar(schema: schema) }

private func accepts(_ text: String, _ schema: AetherGenerationSchema) throws -> (prefix: Bool, complete: Bool) {
    let grammar = try grammar(schema)
    guard let matcher = LanguageJSONMatcher(grammar: grammar).feeding(Array(text.utf8), grammar) else { return (false, false) }
    return (true, matcher.canEnd(grammar))
}

/// Every token the matcher accepts, by simulating each one: the reference for the trie walk.
private func bruteForce(_ matcher: LanguageJSONMatcher, _ grammar: LanguageJSONGrammar) -> LanguageTokenSet {
    var set = LanguageTokenSet(vocabularySize: JSONVocabulary.headRows)
    for id in 0..<JSONVocabulary.pieces.count where matcher.feeding(JSONVocabulary.pieces[id], grammar) != nil { set.insert(id) }
    if matcher.canEnd(grammar) { set.insert(JSONVocabulary.stop) }
    return set
}

struct LanguageJSONSchemaConstraintTests {
    @Test func unsupportedSchemasAreRefused() throws {
        func refuses(_ json: String) -> Bool {
            guard let schema = try? AetherGenerationSchema(json: Data(json.utf8)) else { return true }
            return (try? LanguageJSONGrammar(schema: schema)) == nil
        }
        #expect(refuses(#"{"type": "string", "pattern": "[a-z]+"}"#))
        #expect(refuses(#"{"type": "object", "properties": {}, "additionalProperties": true}"#))
        #expect(refuses(#"{"type": "string", "minLength": 3}"#))
        #expect(refuses(#"{"type": "string", "enum": [1, 2]}"#))
        #expect(refuses(#"{"type": "string", "enum": []}"#))
        #expect(refuses(#"{"type": "integer", "minimum": 5, "maximum": 1}"#))
        #expect(refuses(##"{"$ref": "#/$defs/Missing"}"##))
        #expect(refuses(#"{"type": "object", "properties": {"a": {"type": "string"}}, "x-order": ["b"]}"#))
        #expect(!refuses(Outfit.generationSchema.json))
        #expect(!refuses(KitCharacter.generationSchema.json))
    }

    @Test func theMatcherAcceptsTheCanonicalLayoutOnly() throws {
        let schema = Outfit.generationSchema
        let good = #"{"name": "Mara \"M\" é 日本", "size": 10, "price": 99.95, "colours": ["red", "blue"], "layers": [{"item": "coat", "warm": true}], "note": null, "grade": "B"}"#
        #expect(try accepts(good, schema) == (true, true))
        #expect(try accepts(good.replacingOccurrences(of: #""note": null"#, with: #""note": "dry clean""#), schema) == (true, true))
        #expect(try accepts(String(good.dropLast()), schema) == (true, false), "a prefix is accepted but not complete")
        for bad in [
            good.replacingOccurrences(of: "\"size\": 10", with: "\"size\":10"),        // layout
            good.replacingOccurrences(of: "\"size\": 10", with: "\"size\": 11"),       // range
            good.replacingOccurrences(of: "\"size\": 10", with: "\"size\": 010"),      // leading zero
            good.replacingOccurrences(of: "99.95", with: "100.5"),                    // number range
            good.replacingOccurrences(of: "99.95", with: "0.4"),                      // below the minimum
            good.replacingOccurrences(of: "99.95", with: "1e2"),                      // exponents aren't allowed
            good.replacingOccurrences(of: #"["red", "blue"]"#, with: "[]"),          // too few items
            good.replacingOccurrences(of: #"["red", "blue"]"#, with: #"["red", "red", "red", "red"]"#),
            good.replacingOccurrences(of: #""red""#, with: #""pink""#),               // not in the enum
            good.replacingOccurrences(of: #""grade": "B""#, with: #""grade": "C""#),
            good.replacingOccurrences(of: #"é"#, with: #"\x"#),                  // bad escape
            good.replacingOccurrences(of: "Mara", with: "Ma\nra"),                    // control character
            good.replacingOccurrences(of: #""name": "#, with: #""nom": "#),           // wrong key
            #"{"size": 1, "name": "x"}"#,                                             // wrong order
        ] {
            #expect(try accepts(bad, schema).complete == false, "\(bad)")
        }
        #expect(try accepts("{\"name\": \"日\", ", schema).prefix)
        // Invalid UTF-8: a truncated scalar, an overlong form, a surrogate, a code point past U+10FFFF.
        for invalid: [UInt8] in [[0xE6, 0x41], [0xE0, 0x80, 0x80], [0xED, 0xA0, 0x80], [0xF4, 0x90, 0x80, 0x80], [0xC0, 0x80]] {
            let text = Array("{\"name\": \"".utf8) + invalid
            #expect(LanguageJSONMatcher(grammar: try grammar(schema)).feeding(text, try grammar(schema)) == nil, "\(invalid)")
        }
    }

    @Test func numbersStayFeasibleSoTheReplyNeverDeadEnds() throws {
        func integer(_ text: String) throws -> (Bool, Bool) {
            try accepts(text, AetherGenerationSchema(json: Data(#"{"type": "integer", "minimum": 18, "maximum": 90}"#.utf8)))
        }
        #expect(try integer("1") == (true, false) && integer("18") == (true, true) && integer("9") == (true, false))
        #expect(try integer("90") == (true, true) && integer("91").0 == false && integer("17").0 == false && integer("0").0 == false)
        #expect(try integer("-").0 == false)
        func number(_ text: String) throws -> (Bool, Bool) {
            try accepts(text, AetherGenerationSchema(json: Data(#"{"type": "number", "minimum": 0.5, "maximum": 1}"#.utf8)))
        }
        #expect(try number("0.5") == (true, true) && number("1") == (true, true) && number("1.0") == (true, true))
        #expect(try number("0.4").0 == false && number("0.49").0 == false && number("1.5").0 == false && number("2").0 == false)
        #expect(try number("0.") == (true, false) && number("0") == (true, false))
        let negative = try AetherGenerationSchema(json: Data(#"{"type": "integer", "minimum": -5, "maximum": -2}"#.utf8))
        #expect(try accepts("-3", negative) == (true, true) && accepts("-1", negative).0 == false && accepts("3", negative).0 == false)
    }

    /// Bounds past what the grammar writes (18 integer digits, or 9 + 9 for numbers) and counts past `Int` compile:
    /// they bound nothing, or are refused as empty ranges, instead of trapping in an `Int64` or `Int` conversion.
    @Test func extremeGuidesCompileInsteadOfTrapping() throws {
        let open = AetherGenerationSchema.object(name: "Extremes", properties: [
            .init(name: "a", type: Int.self, guides: [.maximum(Int.max)]), .init(name: "b", type: Int.self, guides: [.range(Int.min...Int.max)]),
            .init(name: "c", type: Double.self, guides: [.maximum(1e12)]), .init(name: "d", type: Double.self, guides: [.minimum(-1e300)]),
            .init(name: "e", type: [String].self, guides: [.maximumCount(Int.max)]),
        ])
        let reply = #"{"a": 999999999999999999, "b": -999999999999999999, "c": 999999999.5, "d": -3, "e": ["red"]}"#
        #expect(try accepts(reply, open) == (true, true))
        func refuses(_ json: String) -> Bool {
            guard let schema = try? AetherGenerationSchema(json: Data(json.utf8)) else { Issue.record("unparsed \(json)"); return false }
            return (try? LanguageJSONGrammar(schema: schema)) == nil
        }
        #expect(refuses(#"{"type": "integer", "minimum": 1e19}"#))
        #expect(refuses(#"{"type": "integer", "maximum": -1e30}"#))
        #expect(refuses(#"{"type": "number", "minimum": 1e12}"#))
        #expect(refuses(#"{"type": "array", "items": {"type": "string"}, "minItems": 1e300}"#))
        #expect(refuses(#"{"type": "array", "items": {"type": "string"}, "minItems": 1.5}"#))
        #expect(!refuses(#"{"type": "array", "items": {"type": "string"}, "maxItems": 1e300}"#))
    }

    /// Review M4: two enums with a case of one name both define `DiscriminatedValue`, and two types can share a name;
    /// each keeps its own schema instead of the last one overwriting the others.
    @Test func definitionsWithOneNameKeepTheirOwnSchemas() throws {
        func union(_ name: String, _ type: (some AetherGenerable).Type) -> AetherGenerationSchema {
            .union(name: name, cases: [.init(name: "value", values: [.init(name: "value", type: type)])])
        }
        struct Named<Payload: AetherGenerable>: AetherGenerable {
            init(from decoder: any Decoder) throws {}
            static var generationSchema: AetherGenerationSchema { .object(name: "Item", properties: [.init(name: "x", type: Payload.self)]) }
        }
        enum First: AetherGenerable {
            init(from decoder: any Decoder) throws { self = .none }
            case none
            static var generationSchema: AetherGenerationSchema { union("First", Int.self) }
        }
        enum Second: AetherGenerable {
            init(from decoder: any Decoder) throws { self = .none }
            case none
            static var generationSchema: AetherGenerationSchema { union("Second", String.self) }
        }
        enum Outer: AetherGenerable {  // its `value` case holds an enum with a `value` case of its own
            init(from decoder: any Decoder) throws { self = .none }
            case none
            static var generationSchema: AetherGenerationSchema { union("Outer", Second.self) }
        }
        let schema = AetherGenerationSchema.object(name: "Root", properties: [
            .init(name: "first", type: First.self), .init(name: "second", type: Second.self), .init(name: "outer", type: Outer.self),
            .init(name: "number", type: Named<Int>.self), .init(name: "text", type: Named<String>.self),
            .init(name: "again", type: First.self),
        ])
        let good = #"{"first": {"type": "value", "value": 1}, "second": {"type": "value", "value": "a"}, "#
            + #""outer": {"type": "value", "value": {"type": "value", "value": "b"}}, "number": {"x": 2}, "text": {"x": "c"}, "#
            + #""again": {"type": "value", "value": 3}}"#
        #expect(try accepts(good, schema) == (true, true))
        for bad in [
            good.replacingOccurrences(of: #""first": {"type": "value", "value": 1}"#, with: #""first": {"type": "value", "value": "1"}"#),
            good.replacingOccurrences(of: #""second": {"type": "value", "value": "a"}"#, with: #""second": {"type": "value", "value": 1}"#),
            good.replacingOccurrences(of: #""value": {"type": "value", "value": "b"}"#, with: #""value": "b""#),
            good.replacingOccurrences(of: #""number": {"x": 2}"#, with: #""number": {"x": "2"}"#),
            good.replacingOccurrences(of: #""text": {"x": "c"}"#, with: #""text": {"x": 3}"#),
        ] {
            #expect(try accepts(bad, schema).complete == false, "\(bad)")
        }
        // Distinct schemas under one name are renamed apart; the same schema used twice is defined once.
        guard case .object(let definitions)? = schema.document["$defs"] else { Issue.record("no $defs"); return }
        #expect(Set(definitions.keys) == ["First", "Second", "Outer", "Item", "Item2",
                                          "DiscriminatedValue", "DiscriminatedValue2", "DiscriminatedValue3"])
    }

    /// Review M5: `\u` escapes pair UTF-16 surrogates as JSON decoders require. A string the matcher completes always
    /// decodes, and one that decodes always completes; a prefix that can't pair is refused at once, never a dead end.
    @Test func unicodeEscapesPairSurrogatesAsDecodersDo() throws {
        let schema = try AetherGenerationSchema(json: Data(#"{"type": "string"}"#.utf8))
        let units = ["0041", "00e9", "D800", "dbff", "D83D", "DC00", "dfff", "DE00", "E000", "FFFF"]
        let escapes = units.map { "\\u\($0)" } + units.flatMap { a in units.map { b in "\\u\(a)\\u\(b)" } }
            + units.map { "\\u\($0)x" } + units.map { "\\u\($0)\\n" }
        for escape in escapes {
            let text = "\"\(escape)\""
            let decodes = (try? JSONDecoder().decode(String.self, from: Data(text.utf8))) != nil
            #expect(try accepts(text, schema) == (decodes, decodes), "\(text)")
        }
        #expect(try accepts(#""\uD83D\uDE00""#, schema) == (true, true))
        #expect(try accepts(#""\uD83D"#, schema) == (true, false), "a high surrogate is a prefix of a pair")
        #expect(try accepts(#""\uD83D\"#, schema) == (true, false) && accepts(#""\uD83D\uD"#, schema) == (true, false))
        for deadEnd in [#""\uD83D""#, #""\uD83D\n"#, #""\uD83D\u0"#, #""\uD83D\uDB"#, #""\uDC"#, #""\udf"#] {
            #expect(try accepts(deadEnd, schema).prefix == false, "\(deadEnd)")
        }
    }

    /// Seeded random walks over the synthetic vocabulary: at every step the trie walk equals simulating every token,
    /// the allowed set is never empty, and every walk ends in JSON that parses and decodes into the type.
    @Test func randomWalksMatchBruteForceAndDecode() throws {
        let schema = Outfit.generationSchema
        let grammar = try grammar(schema)
        var generator = LanguageRandomGenerator(seed: 77)
        var decoded = 0
        let stops = LanguageTokenSet([JSONVocabulary.stop], vocabularySize: JSONVocabulary.headRows)
        for walk in 0..<60 {
            var matcher = LanguageJSONMatcher(grammar: grammar)
            var text: [UInt8] = []
            for _ in 0..<4000 {
                let allowed = LanguageJSONSchemaConstraint.allowedTokens(matcher, grammar: grammar, vocabulary: JSONVocabulary.vocabulary,
                                                                         stopTokens: stops)
                let reference = bruteForce(matcher, grammar)
                #expect(allowed == reference, "walk \(walk) at \(String(decoding: text, as: UTF8.self))")
                let ids = allowed.ids
                guard !ids.isEmpty else { Issue.record("dead end at \(String(decoding: text, as: UTF8.self))"); break }
                // Favour structure over long strings, so walks finish.
                let structural = ids.filter { id in id == JSONVocabulary.stop || JSONVocabulary.pieces[id].contains { "\"]},".utf8.contains($0) } }
                let pool = !structural.isEmpty && generator.next() % 3 != 0 ? structural : ids
                let token = pool[Int(generator.next() % UInt64(pool.count))]
                if token == JSONVocabulary.stop { break }
                text += JSONVocabulary.pieces[token]
                matcher = try #require(matcher.feeding(JSONVocabulary.pieces[token], grammar))
            }
            #expect(matcher.canEnd(grammar), "walk \(walk) unfinished: \(String(decoding: text, as: UTF8.self))")
            let outfit = try JSONDecoder().decode(Outfit.self, from: Data(text))
            #expect((1...10).contains(outfit.size) && (0.5...100).contains(outfit.price) && (1...3).contains(outfit.colours.count))
            #expect(outfit.layers.count <= 2 && ["A", "B"].contains(outfit.grade))
            decoded += 1
        }
        #expect(decoded == 60)
    }

    /// FoundationModels' own schema for a type with every supported feature (unions, consts, nested definitions, guides,
    /// optionals): walks end in JSON that FoundationModels itself decodes into the type.
    @Test func walksOnFoundationModelsSchemasDecodeWithFoundationModels() throws {
        let schema = try AetherGenerationSchema(json: try JSONEncoder().encode(FMCharacter.generationSchema))
        let grammar = try grammar(schema)
        let stops = LanguageTokenSet([JSONVocabulary.stop], vocabularySize: JSONVocabulary.headRows)
        var generator = LanguageRandomGenerator(seed: 5)
        for walk in 0..<25 {
            var matcher = LanguageJSONMatcher(grammar: grammar)
            var text: [UInt8] = []
            while text.count < 20_000 {
                let ids = LanguageJSONSchemaConstraint.allowedTokens(matcher, grammar: grammar, vocabulary: JSONVocabulary.vocabulary,
                                                                     stopTokens: stops).ids
                guard !ids.isEmpty else { Issue.record("dead end"); break }
                let structural = ids.filter { id in id == JSONVocabulary.stop || JSONVocabulary.pieces[id].contains { "\"]},".utf8.contains($0) } }
                let pool = !structural.isEmpty && generator.next() % 4 != 0 ? structural : ids
                let token = pool[Int(generator.next() % UInt64(pool.count))]
                if token == JSONVocabulary.stop { break }
                text += JSONVocabulary.pieces[token]
                matcher = try #require(matcher.feeding(JSONVocabulary.pieces[token], grammar))
            }
            let json = String(decoding: text, as: UTF8.self)
            let character = try FMCharacter(GeneratedContent(json: json))
            #expect((18...90).contains(character.age) && (2...4).contains(character.tags.count) && character.version == "v1", "walk \(walk)")
        }
    }

    /// Opt-in: the real Qwen3 vocabulary (a pinned hub snapshot under `AETHER_LANGUAGE_TOKENIZER_HUB`): index build time,
    /// allowed-set cost per kind of position, and a walk (preferring tokens that close things) that ends in JSON decoding
    /// into the type.
    @Test(.enabled(if: environment["AETHER_LANGUAGE_TOKENIZER_HUB"] != nil))
    func realVocabularyCostAndWalk() throws {
        let hub = URL(fileURLWithPath: try #require(environment["AETHER_LANGUAGE_TOKENIZER_HUB"]))
        let snapshot = hub.appendingPathComponent("models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
        let bytes = try LanguageTokenizer.tokenBytes(directory: snapshot)
        var start = ContinuousClock.now
        let vocabulary = LanguageJSONVocabulary(bytes: bytes, headRows: 151_936, excluded: [151_667, 151_668])
        print("COST vocabulary index", start.duration(to: .now))
        let grammar = try grammar(Outfit.generationSchema)
        let stops = LanguageTokenSet([151_645], vocabularySize: 151_936)
        func cost(_ prefix: String) throws -> (Duration, Int) {
            let matcher = try #require(LanguageJSONMatcher(grammar: grammar).feeding(Array(prefix.utf8), grammar))
            start = ContinuousClock.now
            let set = LanguageJSONSchemaConstraint.allowedTokens(matcher, grammar: grammar, vocabulary: vocabulary, stopTokens: stops)
            return (start.duration(to: .now), set.count)
        }
        for prefix in ["", #"{"name": "#, #"{"name": "Mara"#, #"{"name": "Mara", "size": "#, #"{"name": "Mara", "size": 7, "price": 12."#,
                       #"{"name": "Mara", "size": 7, "price": 12.5, "colours": ["#, #"{"name": "Mara", "size": 7, "price": 12.5, "colours": ["red"]"#] {
            let (elapsed, count) = try cost(prefix)
            print("COST", prefix.debugDescription, elapsed, count)
        }
        // A walk through every position kind.
        var matcher = LanguageJSONMatcher(grammar: grammar)
        var text: [UInt8] = []
        var longest = Duration.zero, steps = 0
        while steps < 2000 {
            start = ContinuousClock.now
            let ids = LanguageJSONSchemaConstraint.allowedTokens(matcher, grammar: grammar, vocabulary: vocabulary, stopTokens: stops).ids
            longest = max(longest, start.duration(to: .now))
            // Prefer ending, then closing a string, then closing a container, so the walk finishes.
            let closing = ids.contains(151_645) ? [151_645] : ["\"", "]}"].lazy.map { marks in
                ids.filter { bytes.bytes(of: $0)?.contains { marks.utf8.contains($0) } ?? false }
            }.first { !$0.isEmpty } ?? ids
            let token = try #require(closing.last)
            if token == 151_645 { break }
            text += bytes.bytes(of: token)!
            matcher = try #require(matcher.feeding(bytes.bytes(of: token)!, grammar))
            steps += 1
        }
        print("COST walk", steps, "steps, longest", longest, String(decoding: text, as: UTF8.self))
        _ = try JSONDecoder().decode(Outfit.self, from: Data(text))
    }

    @Test func theProcessorConstrainsAfterReasoningAndCaches() async throws {
        let constraint = try LanguageJSONSchemaConstraint(schema: Layer.generationSchema, vocabulary: JSONVocabulary.vocabulary,
                                                          stopTokenIDs: [JSONVocabulary.stop], reasoningEnd: JSONVocabulary.other - 1)
        #expect(await constraint.begin(prompt: [1, 2]) == .unconstrained)
        #expect(try await constraint.accept(5) == .unconstrained, "still thinking")
        guard case .allowed(let first) = try await constraint.accept(JSONVocabulary.other - 1) else { Issue.record("expected a set"); return }
        #expect(first.contains(Int(UInt8(ascii: "{"))) && first.contains(JSONVocabulary.pieces.firstIndex(of: Array("{\"".utf8))!))
        #expect(!first.contains(Int(UInt8(ascii: "["))) && !first.contains(JSONVocabulary.stop))
        var last = LanguageTokenConstraint.unconstrained
        for byte in Array(#"{"item": "scarf", "warm": false}"#.utf8) { last = try await constraint.accept(Int(byte)) }
        #expect(last == .allowed(LanguageTokenSet([JSONVocabulary.stop], vocabularySize: JSONVocabulary.headRows)), "only the stop once complete")
        await #expect(throws: AetherLanguageError.self) { try await constraint.accept(Int(UInt8(ascii: "x"))) }
        let statistics = await constraint.statistics
        #expect(statistics.calls > 30 && statistics.cacheHits > 0)

        let plain = try LanguageJSONSchemaConstraint(schema: Layer.generationSchema, vocabulary: JSONVocabulary.vocabulary,
                                                     stopTokenIDs: [JSONVocabulary.stop])
        guard case .allowed = await plain.begin(prompt: []) else { Issue.record("constrained from the start"); return }
    }
}
