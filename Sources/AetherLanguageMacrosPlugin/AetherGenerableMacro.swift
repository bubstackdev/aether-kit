import SwiftCompilerPlugin
import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

@main
struct AetherLanguageMacrosPlugin: CompilerPlugin {
    let providingMacros: [any Macro.Type] = [AetherGenerableMacro.self, AetherGuideMacro.self]
}

/// `@AetherGuide` only marks a property; `@AetherGenerable` reads it.
public struct AetherGuideMacro: PeerMacro {
    public static func expansion(of node: AttributeSyntax, providingPeersOf declaration: some DeclSyntaxProtocol,
                                 in context: some MacroExpansionContext) throws -> [DeclSyntax] { [] }
}

struct MacroError: Error, CustomStringConvertible {
    let description: String
}

public struct AetherGenerableMacro: ExtensionMacro {
    public static func expansion(of node: AttributeSyntax, attachedTo declaration: some DeclGroupSyntax,
                                 providingExtensionsOf type: some TypeSyntaxProtocol, conformingTo protocols: [TypeSyntax],
                                 in context: some MacroExpansionContext) throws -> [ExtensionDeclSyntax] {
        let description = Self.stringArgument("description", of: node)
        let access = Self.access(of: declaration)
        let members: String
        if let structure = declaration.as(StructDeclSyntax.self) {
            members = try structMembers(structure, description: description, access: access)
        } else if let enumeration = declaration.as(EnumDeclSyntax.self) {
            members = try enumMembers(enumeration, description: description, access: access)
        } else {
            throw MacroError(description: "@AetherGenerable applies to structs and enums")
        }
        let extensionDecl: DeclSyntax = """
            extension \(type.trimmed): AetherLanguage.AetherGenerable {
            \(raw: members)
            }
            """
        return [extensionDecl.cast(ExtensionDeclSyntax.self)]
    }

    // MARK: Structs

    private static func structMembers(_ structure: StructDeclSyntax, description: String?, access: String) throws -> String {
        var properties: [String] = []
        for member in structure.memberBlock.members {
            guard let variable = member.decl.as(VariableDeclSyntax.self),
                  !variable.modifiers.contains(where: { ["static", "class"].contains($0.name.text) }) else { continue }
            for binding in variable.bindings {
                // Computed properties aren't stored, so they aren't generated.
                if let accessors = binding.accessorBlock, !Self.isObserverOnly(accessors) { continue }
                guard let name = binding.pattern.as(IdentifierPatternSyntax.self)?.identifier.text else { continue }
                guard let type = binding.typeAnnotation?.type.trimmed else {
                    throw MacroError(description: "@AetherGenerable needs an explicit type on '\(name)'")
                }
                let guide = variable.attributes.compactMap { $0.as(AttributeSyntax.self) }
                    .first { $0.attributeName.trimmedDescription.hasSuffix("AetherGuide") }
                properties.append(property(name: name, type: type, guide: guide))
            }
        }
        let name = structure.name.text
        return """
            \(access)static var generationSchema: AetherLanguage.AetherGenerationSchema {
                .object(name: \(literal(name)), description: \(description.map(literal) ?? "nil"), properties: [
            \(properties.map { "        \($0)," }.joined(separator: "\n"))
                ])
            }
            """
    }

    private static func property(name: String, type: TypeSyntax, guide: AttributeSyntax?) -> String {
        var description = "nil"
        var guides: [String] = []
        if case .argumentList(let arguments)? = guide?.arguments {
            for argument in arguments {
                if argument.label?.text == "description" { description = argument.expression.trimmedDescription }
                else { guides.append(argument.expression.trimmedDescription) }
            }
        }
        return ".init(name: \(literal(name)), description: \(description), type: \(typeExpression(type)).self, guides: [\(guides.joined(separator: ", "))])"
    }

    /// The type as an expression for `.self`: `Optional<T>` and `T?` alike become `T?`.
    private static func typeExpression(_ type: TypeSyntax) -> String {
        if let optional = type.as(OptionalTypeSyntax.self) { return "\(optional.wrappedType.trimmedDescription)?" }
        if let identifier = type.as(IdentifierTypeSyntax.self), identifier.name.text == "Optional",
           let wrapped = identifier.genericArgumentClause?.arguments.first {
            return "\(wrapped.argument.trimmedDescription)?"
        }
        return type.trimmedDescription
    }

    private static func isObserverOnly(_ accessors: AccessorBlockSyntax) -> Bool {
        guard case .accessors(let list) = accessors.accessors else { return false }
        return list.allSatisfy { ["willSet", "didSet"].contains($0.accessorSpecifier.text) }
    }

    // MARK: Enums

    private struct EnumCase {
        let name: String
        let values: [(key: String, type: TypeSyntax)]
    }

    private static func enumMembers(_ enumeration: EnumDeclSyntax, description: String?, access: String) throws -> String {
        var cases: [EnumCase] = []
        for member in enumeration.memberBlock.members {
            guard let caseDecl = member.decl.as(EnumCaseDeclSyntax.self) else { continue }
            for element in caseDecl.elements {
                let parameters = element.parameterClause?.parameters ?? []
                if parameters.count > 1, parameters.contains(where: { $0.firstName == nil }) {
                    throw MacroError(description: "@AetherGenerable needs labels on case '\(element.name.text)''s values")
                }
                let values = parameters.map { (key: $0.firstName?.text ?? "value", type: $0.type.trimmed) }
                cases.append(EnumCase(name: element.name.text, values: values))
            }
        }
        guard !cases.isEmpty else { throw MacroError(description: "@AetherGenerable needs at least one case") }
        let name = enumeration.name.text
        let typeDescription = description.map(literal) ?? "nil"
        let unknownValue = "throw DecodingError.dataCorruptedError(in: container, debugDescription: \"Unknown case \\(value)\")"
        let unknownType = "throw DecodingError.dataCorruptedError(forKey: \"type\", in: container, debugDescription: \"Unknown case \\(value)\")"
        if cases.allSatisfy({ $0.values.isEmpty }) {
            let names = cases.map { literal($0.name) }.joined(separator: ", ")
            let switchCases = cases.map { "        case \(literal($0.name)): self = .\($0.name)" }.joined(separator: "\n")
            return """
                \(access)static var generationSchema: AetherLanguage.AetherGenerationSchema {
                    .enumeration(name: \(literal(name)), description: \(typeDescription), cases: [\(names)])
                }
                \(access)init(from decoder: any Decoder) throws {
                    let container = try decoder.singleValueContainer()
                    let value = try container.decode(String.self)
                    switch value {
                \(switchCases)
                    default: \(unknownValue)
                    }
                }
                """
        }
        let schemaCases = cases.map { item -> String in
            let values = item.values.map { ".init(name: \(literal($0.key)), type: \(typeExpression($0.type)).self)" }
            return "        .init(name: \(literal(item.name)), values: [\(values.joined(separator: ", "))]),"
        }.joined(separator: "\n")
        let switchCases = cases.map { item -> String in
            guard !item.values.isEmpty else { return "        case \(literal(item.name)): self = .\(item.name)" }
            let arguments = item.values.map { value -> String in
                let decode = value.type.is(OptionalTypeSyntax.self)
                    ? "try container.decodeIfPresent(\(typeExpression(value.type).dropLast()).self, forKey: \(literal(value.key)))"
                    : "try container.decode(\(value.type.trimmedDescription).self, forKey: \(literal(value.key)))"
                return value.key == "value" && item.values.count == 1 ? decode : "\(value.key): \(decode)"
            }
            return "        case \(literal(item.name)): self = .\(item.name)(\(arguments.joined(separator: ", ")))"
        }.joined(separator: "\n")
        return """
            \(access)static var generationSchema: AetherLanguage.AetherGenerationSchema {
                .union(name: \(literal(name)), description: \(typeDescription), cases: [
            \(schemaCases)
                ])
            }
            \(access)init(from decoder: any Decoder) throws {
                let container = try decoder.container(keyedBy: AetherLanguage.AetherGenerableCodingKey.self)
                let value = try container.decode(String.self, forKey: "type")
                switch value {
            \(switchCases)
                default: \(unknownType)
                }
            }
            """
    }

    // MARK: Helpers

    private static func stringArgument(_ label: String, of node: AttributeSyntax) -> String? {
        guard case .argumentList(let arguments)? = node.arguments,
              let argument = arguments.first(where: { $0.label?.text == label }),
              let string = argument.expression.as(StringLiteralExprSyntax.self) else { return nil }
        return string.segments.compactMap { $0.as(StringSegmentSyntax.self)?.content.text }.joined()
    }

    /// The members' access: as visible as the type, for public and package types.
    private static func access(of declaration: some DeclGroupSyntax) -> String {
        for modifier in declaration.modifiers where ["public", "package", "open"].contains(modifier.name.text) {
            return modifier.name.text == "open" ? "public " : "\(modifier.name.text) "
        }
        return ""
    }

    /// A Swift string literal for `text`.
    private static func literal(_ text: String) -> String {
        var out = "\""
        for character in text {
            switch character {
            case "\\": out += "\\\\"
            case "\"": out += "\\\""
            case "\n": out += "\\n"
            default: out.append(character)
            }
        }
        return out + "\""
    }
}
