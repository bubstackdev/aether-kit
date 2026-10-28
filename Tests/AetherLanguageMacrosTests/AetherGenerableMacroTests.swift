import AetherLanguageMacros
import AetherLanguageMacrosPlugin
import Foundation
import FoundationModels
import SwiftSyntaxMacroExpansion
import SwiftSyntaxMacrosGenericTestSupport
import Testing

// Paired types: the same declaration under FoundationModels' @Generable and Kit's @AetherGenerable.

@Generable(description: "Moods") enum FMMood { case calm, curious, stormy }
@AetherGenerable(description: "Moods") enum KitMood { case calm, curious, stormy }

@Generable enum FMWeapon { case sword(edge: Int), bow, staff(Double) }
@AetherGenerable enum KitWeapon: Equatable { case sword(edge: Int), bow, staff(Double) }

@Generable(description: "A colour swatch") struct FMSwatch {
    var name: String
    @Guide(description: "0-255", .range(0...255)) var red: Int
}
@AetherGenerable(description: "A colour swatch") struct KitSwatch: Equatable {
    var name: String
    @AetherGuide(description: "0-255", .range(0...255)) var red: Int
}

@Generable(description: "A fictional character") struct FMCharacter {
    @Guide(description: "Full name") var name: String
    @Guide(.range(18...90)) var age: Int
    @Guide(.minimum(0.5)) var height: Double
    var alive: Bool
    @Guide(description: "described enum") var mood: FMMood
    @Guide(description: "Tags", .count(2...4)) var tags: [String]
    @Guide(.anyOf(["red", "green"])) var colour: String
    @Guide(.constant("v1")) var version: String
    @Guide(description: "described nested") var swatch: FMSwatch
    var swatches: [FMSwatch]
    var nickname: String?
    @Guide(.maximumCount(3)) var scores: [Int]
    @Guide(.element(.range(1...5))) var ratings: [Int]
    var ratio: Float
    var weapon: FMWeapon
    var maybeSwatch: Optional<FMSwatch>
    var moods: [FMMood]
    @Guide(.count(3)) var three: [String]
    var nested: [[Int]]
    @Guide(.range(0.0...1.0)) var unit: Double
    var computed: Int { 3 }
    static let shared = 0
}
@AetherGenerable(description: "A fictional character") struct KitCharacter {
    @AetherGuide(description: "Full name") var name: String
    @AetherGuide(.range(18...90)) var age: Int
    @AetherGuide(.minimum(0.5)) var height: Double
    var alive: Bool
    @AetherGuide(description: "described enum") var mood: KitMood
    @AetherGuide(description: "Tags", .count(2...4)) var tags: [String]
    @AetherGuide(.anyOf(["red", "green"])) var colour: String
    @AetherGuide(.constant("v1")) var version: String
    @AetherGuide(description: "described nested") var swatch: KitSwatch
    var swatches: [KitSwatch]
    var nickname: String?
    @AetherGuide(.maximumCount(3)) var scores: [Int]
    @AetherGuide(.element(.range(1...5))) var ratings: [Int]
    var ratio: Float
    var weapon: KitWeapon
    var maybeSwatch: Optional<KitSwatch>
    var moods: [KitMood]
    @AetherGuide(.count(3)) var three: [String]
    var nested: [[Int]]
    @AetherGuide(.range(0.0...1.0)) var unit: Double
    var computed: Int { 3 }
    static let shared = 0
}

/// A schema's JSON object with Kit type names replaced by FoundationModels' (the only intended difference).
private func normalized(_ data: Data) throws -> NSDictionary {
    var text = String(decoding: data, as: UTF8.self)
    for name in ["Mood", "Weapon", "Swatch", "Character"] { text = text.replacingOccurrences(of: "Kit\(name)", with: "FM\(name)") }
    return try JSONSerialization.jsonObject(with: Data(text.utf8)) as! NSDictionary
}

struct AetherGenerableMacroTests {
    @Test func macroSchemasMatchFoundationModelsForTheSameDeclarations() throws {
        let pairs: [(String, Data, AetherGenerationSchema)] = [
            ("mood", try JSONEncoder().encode(FMMood.generationSchema), KitMood.generationSchema),
            ("weapon", try JSONEncoder().encode(FMWeapon.generationSchema), KitWeapon.generationSchema),
            ("swatch", try JSONEncoder().encode(FMSwatch.generationSchema), KitSwatch.generationSchema),
            ("character", try JSONEncoder().encode(FMCharacter.generationSchema), KitCharacter.generationSchema),
        ]
        for (name, foundation, kit) in pairs {
            #expect(try normalized(Data(kit.json.utf8)) == (try normalized(foundation)),
                    "\(name)\nkit \(kit.json)\nfm  \(String(decoding: foundation, as: UTF8.self))")
        }
    }

    /// JSON in the constraint's layout decodes into the macro's types, as it does into FoundationModels'.
    @Test func macroTypesDecodeWhatFoundationModelsDecodes() throws {
        let json = #"{"name": "Aria", "age": 30, "height": 1.7, "alive": true, "mood": "curious", "tags": ["a", "b"], "colour": "red", "version": "v1", "swatch": {"name": "teal", "red": 0}, "swatches": [], "nickname": null, "scores": [1], "ratings": [5], "ratio": 0.5, "weapon": {"type": "sword", "edge": 3}, "maybeSwatch": {"name": "x", "red": 255}, "moods": ["calm", "stormy"], "three": ["x", "y", "z"], "nested": [[1], []], "unit": 0.25}"#
        let kit = try JSONDecoder().decode(KitCharacter.self, from: Data(json.utf8))
        let foundation = try FMCharacter(GeneratedContent(json: json))
        #expect(kit.name == foundation.name && kit.age == foundation.age && kit.tags == foundation.tags && kit.nickname == nil)
        #expect(kit.mood == .curious && kit.weapon == .sword(edge: 3) && kit.swatch == KitSwatch(name: "teal", red: 0))
        #expect(kit.maybeSwatch == KitSwatch(name: "x", red: 255) && kit.moods == [.calm, .stormy] && kit.nested == [[1], []])
        for (text, expected) in [(#"{"type": "bow"}"#, KitWeapon.bow), (#"{"type": "staff", "value": 1.5}"#, .staff(1.5))] {
            #expect(try JSONDecoder().decode(KitWeapon.self, from: Data(text.utf8)) == expected)
        }
        #expect(throws: DecodingError.self) { try JSONDecoder().decode(KitWeapon.self, from: Data(#"{"type": "axe"}"#.utf8)) }
        #expect(throws: DecodingError.self) { try JSONDecoder().decode(KitMood.self, from: Data(#""angry""#.utf8)) }
    }

    @Test func expansionOfAStructAndItsDiagnostics() {
        let macros: [String: MacroSpec] = ["AetherGenerable": MacroSpec(type: AetherGenerableMacro.self),
                                           "AetherGuide": MacroSpec(type: AetherGuideMacro.self)]
        assertMacroExpansion("""
            @AetherGenerable(description: "A layer")
            public struct Layer {
                @AetherGuide(description: "What it is") var item: String
                @AetherGuide(.range(1...3)) var weight: Int
                var note: String?
            }
            """, expandedSource: """
            public struct Layer {
                var item: String
                var weight: Int
                var note: String?
            }

            extension Layer: AetherLanguage.AetherGenerable {
                public static var generationSchema: AetherLanguage.AetherGenerationSchema {
                    .object(name: "Layer", description: "A layer", properties: [
                        .init(name: "item", description: "What it is", type: String.self, guides: []),
                        .init(name: "weight", description: nil, type: Int.self, guides: [.range(1 ... 3)]),
                        .init(name: "note", description: nil, type: String?.self, guides: []),
                    ])
                }
            }
            """, macroSpecs: macros, failureHandler: { Issue.record("\($0.message)") })
        assertMacroExpansion("""
            @AetherGenerable class Box {
                var x: Int = 0
            }
            """, expandedSource: """
            class Box {
                var x: Int = 0
            }
            """, diagnostics: [DiagnosticSpec(message: "@AetherGenerable applies to structs and enums", line: 1, column: 1)],
            macroSpecs: macros, failureHandler: { Issue.record("\($0.message)") })
        assertMacroExpansion("""
            @AetherGenerable enum Pair {
                case both(Int, Int)
            }
            """, expandedSource: """
            enum Pair {
                case both(Int, Int)
            }
            """, diagnostics: [DiagnosticSpec(message: "@AetherGenerable needs labels on case 'both''s values", line: 1, column: 1)],
            macroSpecs: macros, failureHandler: { Issue.record("\($0.message)") })
    }
}
