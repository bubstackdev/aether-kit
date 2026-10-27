@_exported import AetherLanguage

/// Makes a struct or enum ``AetherGenerable``: writes its `generationSchema` (the same JSON Schema FoundationModels'
/// `@Generable` emits for the same type) and, for enums, the decoding of the JSON a constrained model writes.
///
/// - Structs: every stored property, in declaration order; optional properties may be `null`. Properties are decoded
///   by the compiler-synthesized `Decodable`, so their types must be generable (`String`, `Int`, `Double`, `Float`,
///   `Bool`, arrays and optionals of them, and other `@AetherGenerable` types).
/// - Enums without associated values: one of the case names.
/// - Enums with associated values: an object with the case name under `"type"` and the values by label (a lone
///   unlabelled value is `"value"`).
///
/// ```swift
/// @AetherGenerable(description: "A fictional character")
/// struct Character {
///     @AetherGuide(description: "Full name") var name: String
///     @AetherGuide(.range(18...90)) var age: Int
///     @AetherGuide(.count(2...4)) var tags: [String]
///     var nickname: String?
/// }
/// let character = try await session.respond(to: "Invent one.", generating: Character.self).content
/// ```
@attached(extension, conformances: AetherGenerable, names: named(generationSchema), named(init(from:)))
public macro AetherGenerable(description: String? = nil) =
    #externalMacro(module: "AetherLanguageMacrosPlugin", type: "AetherGenerableMacro")

/// Describes and narrows one property of an `@AetherGenerable` struct, as FoundationModels' `@Guide`: a description for
/// the model, and guides (`.range`, `.minimum`, `.maximum`, `.count`, `.minimumCount`, `.maximumCount`, `.element`,
/// `.anyOf`, `.constant`).
@attached(peer)
public macro AetherGuide<T>(description: String? = nil, _ guides: AetherGenerationGuide<T>...) =
    #externalMacro(module: "AetherLanguageMacrosPlugin", type: "AetherGuideMacro")

@attached(peer)
public macro AetherGuide(description: String) = #externalMacro(module: "AetherLanguageMacrosPlugin", type: "AetherGuideMacro")
