@testable import AetherLanguage
import Foundation
import FoundationModels
import Testing

// Paired types: FoundationModels' @Generable on one side, a hand-written AetherGenerable schema on the other. The
// macro tests pair macro-generated schemas the same way.

@Generable(description: "Moods") enum FMMood { case calm, curious }
enum KitMood: String, AetherGenerable {
    case calm, curious
    static var generationSchema: AetherGenerationSchema { .enumeration(name: "KitMood", description: "Moods", cases: ["calm", "curious"]) }
}

@Generable(description: "A colour swatch") struct FMSwatch {
    var name: String
    @Guide(description: "0-255", .range(0...255)) var red: Int
}
struct KitSwatch: AetherGenerable {
    var name: String
    var red: Int
    static var generationSchema: AetherGenerationSchema {
        .object(name: "KitSwatch", description: "A colour swatch", properties: [
            .init(name: "name", type: String.self),
            .init(name: "red", description: "0-255", type: Int.self, guides: [.range(0...255)]),
        ])
    }
}

@Generable enum FMShape { case circle(Double), square(side: Double, filled: Bool), none }
enum KitShape: AetherGenerable {
    case circle(Double), square(side: Double, filled: Bool), none
    init(from decoder: any Decoder) throws { self = .none }
    static var generationSchema: AetherGenerationSchema {
        .union(name: "KitShape", cases: [
            .init(name: "circle", values: [.init(name: "value", type: Double.self)]),
            .init(name: "square", values: [.init(name: "side", type: Double.self), .init(name: "filled", type: Bool.self)]),
            .init(name: "none"),
        ])
    }
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
    var shape: FMShape
    var maybeSwatch: FMSwatch?
    var moods: [FMMood]
    @Guide(.count(3)) var three: [String]
    var nested: [[Int]]
    @Guide(.range(0.0...1.0)) var unit: Double
}
struct KitCharacter: AetherGenerable {
    static var generationSchema: AetherGenerationSchema {
        .object(name: "KitCharacter", description: "A fictional character", properties: [
            .init(name: "name", description: "Full name", type: String.self),
            .init(name: "age", type: Int.self, guides: [.range(18...90)]),
            .init(name: "height", type: Double.self, guides: [.minimum(0.5)]),
            .init(name: "alive", type: Bool.self),
            .init(name: "mood", description: "described enum", type: KitMood.self),
            .init(name: "tags", description: "Tags", type: [String].self, guides: [.count(2...4)]),
            .init(name: "colour", type: String.self, guides: [.anyOf(["red", "green"])]),
            .init(name: "version", type: String.self, guides: [.constant("v1")]),
            .init(name: "swatch", description: "described nested", type: KitSwatch.self),
            .init(name: "swatches", type: [KitSwatch].self),
            .init(name: "nickname", type: String?.self),
            .init(name: "scores", type: [Int].self, guides: [.maximumCount(3)]),
            .init(name: "ratings", type: [Int].self, guides: [.element(.range(1...5))]),
            .init(name: "ratio", type: Float.self),
            .init(name: "shape", type: KitShape.self),
            .init(name: "maybeSwatch", type: KitSwatch?.self),
            .init(name: "moods", type: [KitMood].self),
            .init(name: "three", type: [String].self, guides: [.count(3)]),
            .init(name: "nested", type: [[Int]].self),
            .init(name: "unit", type: Double.self, guides: [.range(0.0...1.0)]),
        ])
    }
    init(from decoder: any Decoder) throws {}
}

/// The JSON object of a schema with Kit type names replaced by FoundationModels' (the only intended difference).
func normalizedSchema(_ data: Data) throws -> NSDictionary {
    var text = String(decoding: data, as: UTF8.self)
    for name in ["Mood", "Swatch", "Shape", "Character", "Look", "Outfit", "Palette", "Weapon"] {
        text = text.replacingOccurrences(of: "Kit\(name)", with: "FM\(name)")
    }
    return try JSONSerialization.jsonObject(with: Data(text.utf8)) as! NSDictionary
}

func fmSchemaJSON<T: Generable>(_ type: T.Type) throws -> Data { try JSONEncoder().encode(T.generationSchema) }

struct AetherGenerationSchemaTests {
    @Test func schemasMatchFoundationModelsForTheSameTypes() throws {
        let pairs: [(String, Data, Data)] = [
            ("mood", try fmSchemaJSON(FMMood.self), Data(KitMood.generationSchema.json.utf8)),
            ("swatch", try fmSchemaJSON(FMSwatch.self), Data(KitSwatch.generationSchema.json.utf8)),
            ("shape", try fmSchemaJSON(FMShape.self), Data(KitShape.generationSchema.json.utf8)),
            ("character", try fmSchemaJSON(FMCharacter.self), Data(KitCharacter.generationSchema.json.utf8)),
        ]
        for (name, foundation, kit) in pairs {
            let expected = try normalizedSchema(foundation), actual = try normalizedSchema(kit)
            #expect(actual == expected, "\(name):\nkit \(String(decoding: kit, as: UTF8.self))\nfm  \(String(decoding: foundation, as: UTF8.self))")
        }
    }

    @Test func foundationModelsSchemasRoundTrip() throws {
        let data = try fmSchemaJSON(FMCharacter.self)
        let schema = try AetherGenerationSchema(json: data)
        #expect(schema.name == "FMCharacter")
        #expect(try normalizedSchema(Data(schema.json.utf8)) == (try normalizedSchema(data)))
        let decoded = try JSONDecoder().decode(AetherGenerationSchema.self, from: try JSONEncoder().encode(schema))
        #expect(decoded == schema)
    }
}
