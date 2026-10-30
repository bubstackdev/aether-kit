#!/usr/bin/env python3
"""Helpers for Scripts/check-api.sh, the 0.0.x API freeze (docs/api-stability.md).

  api-freeze.py prune FREEZE_JSON MODULE FULL_DUMP OUT
      Reduces a swift-api-digester dump to the module's frozen declarations: drops imports, internal and package
      declarations (`isInternal`), every top-level declaration the freeze doesn't name, and the tool's arguments (they
      hold local paths). Fails if a frozen name is missing, nothing is left, or what's left exposes an unfrozen Aether
      type (in a signature, a generic constraint or a conformance).
  api-freeze.py diagnose BASELINE DIAGNOSTICS
      Reads `swift-api-digester -diagnose-sdk` output and fails on any finding, except a case added to an enum the
      baseline marks @nonexhaustive, which clients already handle with `@unknown default`.
"""
import json
import re
import sys


def fail(message):
    print(f"api-freeze: {message}", file=sys.stderr)
    sys.exit(1)


def strip_internal(node):
    children = [strip_internal(child) for child in node.get("children", []) if not child.get("isInternal")]
    if "children" in node:
        node["children"] = children
    return node


def is_frozen(qualified, freeze):
    """Whether an Aether type, "Module.Outer[.Inner...]", is frozen: its module is frozen whole, its outer type is
    named, or "Outer.Inner" is named and the type is Inner or nested in it."""
    module, _, path = qualified.partition(".")
    if module not in freeze:
        return False
    names = freeze[module]
    if names == ["*"]:
        return True
    parts = path.split(".")
    return parts[0] in names or (len(parts) > 1 and f"{parts[0]}.{parts[1]}" in names)


def qualified_names(text):
    """The module-qualified Aether types ("AetherKit.Outer.Inner") a printed type or generic signature names."""
    return {re.sub(r"\.Type$", "", m.group(0)) for m in re.finditer(r"\bAether\w+(?:\.\w+)+", text)}


def mangled_words(text):
    """The words Swift's mangling can refer back to from an identifier's literal text, as the demangler collects them:
    runs of two or more characters that don't start with a digit or underscore, ending at an underscore or at an
    uppercase letter after a non-uppercase one ("AetherLanguage": Aether, Language; "Model2X" is one word)."""
    words, start = [], None
    for index, char in enumerate(text + "\0"):
        if start is not None and (char in "_\0" or (char.isupper() and not text[index - 1].isupper())):
            if index - start >= 2:
                words.append(text[start:index])
            start = None
        if start is None and not char.isdigit() and char not in "_\0":
            start = index
    return words


def demangled_name(mangled):
    """The qualified name of a conformance's protocol from its mangled name, "$s" then identifiers each followed by a
    kind letter: "$s12AetherAssets14ModelTransportP" is "AetherAssets.ModelTransport". An identifier is a length and
    its text, or, after a "0", any mix of references to earlier words (a-z, or A-Z for the last) and lengths with text,
    ended by an uppercase reference or a "0": "$s14AetherLanguage0A9GenerableP" is "AetherLanguage.AetherGenerable",
    "$s14AetherLanguage06GuidedA5ThingP" is "AetherLanguage.GuidedAetherThing". None when the module isn't an Aether
    one; "?" when it is but the name uses a form this doesn't read (punycode, other substitutions, generics), so the
    caller fails closed."""
    if not re.match(r"\$s\d", mangled):
        return None
    position, parts, words = 2, [], []

    def literal():
        nonlocal position
        length = re.match(r"\d+", mangled[position:])
        if not length or length.group(0).startswith("0"):
            return None
        start = position + len(length.group(0))
        text = mangled[start:start + int(length.group(0))]
        if len(text) < int(length.group(0)):
            return None
        position = start + len(text)
        # The demangler keeps at most 26 words, one per reference letter.
        words.extend(mangled_words(text)[:26 - len(words)])
        return text

    def identifier():
        """The next identifier, or None if it isn't in a form this reads."""
        nonlocal position
        if mangled[position] != "0":
            return literal()
        position += 1
        if position >= len(mangled) or mangled[position] == "0":
            return None  # punycode
        name, substituting = "", True
        while True:
            while substituting and position < len(mangled) and mangled[position].isalpha():
                char = mangled[position]
                index = ord(char.lower()) - ord("a")
                if index >= len(words):
                    return None
                name += words[index]
                position += 1
                substituting = char.islower()
            if position < len(mangled) and mangled[position] == "0":
                position += 1
                return name or None
            text = literal()
            if text is None:
                return None
            name += text
            if not substituting:
                return name

    while position < len(mangled) and mangled[position].isdigit():
        name = identifier()
        if name is None:
            return "?" if parts and parts[0].startswith("Aether") else None
        parts.append(name)
        if not parts[0].startswith("Aether"):
            return None
        if position < len(mangled) and mangled[position] in "VOCPa":
            position += 1
    if not parts or not parts[0].startswith("Aether"):
        return None
    if position != len(mangled) or len(parts) < 2:
        return "?"
    return ".".join(parts)


def unfrozen_references(node, freeze, module):
    """The Aether types a pruned dump of `module` names that the freeze doesn't cover: in signatures, in generic
    constraints, and as protocols the frozen types conform to. A frozen declaration that exposes an unfrozen type would
    let that type break the frozen API. An extension of another module's type (`isExternal`) lists every conformance
    visible to the build; only those to `module`'s own protocols are its API. The dump doesn't say which module declared
    a conformance, so `module`'s own conformance of another module's type to a third module's protocol isn't seen."""
    names = set()
    if node.get("kind") == "TypeNominal":
        names |= qualified_names(node.get("printedName", ""))
    for key in ("genericSig", "sugared_genericSig"):
        names |= qualified_names(node.get(key, ""))
    found = {name for name in names if not is_frozen(name, freeze)}
    for conformance in node.get("conformances", []):
        name = demangled_name(conformance.get("mangledName", ""))
        if name == "?":
            found.add(f"{conformance.get('name')} (conformance {conformance.get('mangledName')} unreadable)")
        elif name and not (node.get("isExternal") and name.partition(".")[0] != module) and not is_frozen(name, freeze):
            found.add(name)
    # A property's accessors sit under their own key; in a constrained extension only they carry the constraint.
    for child in node.get("children", []) + node.get("accessors", []):
        found |= unfrozen_references(child, freeze, module)
    return found


def prune(freeze_path, module, dump_path, out_path):
    with open(freeze_path) as f:
        freeze = json.load(f)["modules"]
    if module not in freeze:
        fail(f"{module} is not in {freeze_path}")
    names = freeze[module]
    with open(dump_path) as f:
        dump = json.load(f)
    root = dump.get("ABIRoot")
    if not root or root.get("name") != module:
        fail(f"{dump_path} is not a dump of {module}")
    root.pop("tool_arguments", None)
    declarations = [c for c in root.get("children", []) if c.get("kind") != "Import" and not c.get("isInternal")]
    if names != ["*"]:
        # "Outer.Inner" keeps only Inner inside Outer; "Outer" keeps all of Outer.
        wanted = {}
        for name in names:
            outer, _, inner = name.partition(".")
            wanted.setdefault(outer, set()).add(inner or "*")
        kept = []
        for declaration in declarations:
            # An extension of another module's type (isExternal) is frozen only with its module's whole freeze.
            if declaration.get("isExternal") or declaration["name"] not in wanted:
                continue
            inner = wanted[declaration["name"]]
            if "*" not in inner:
                declaration["children"] = [c for c in declaration.get("children", []) if c.get("name") in inner]
                missing = inner - {c.get("name") for c in declaration["children"]}
                if missing:
                    fail(f"{module}: frozen {declaration['name']}.{sorted(missing)[0]} is missing")
            kept.append(declaration)
        missing = set(wanted) - {d["name"] for d in kept}
        if missing:
            fail(f"{module}: frozen declaration {sorted(missing)[0]} is missing")
        declarations = kept
    if not declarations:
        fail(f"{module}: the dump has no frozen declarations")
    root["children"] = [strip_internal(d) for d in declarations]
    unfrozen = sorted(unfrozen_references(root, freeze, module))
    if unfrozen:
        fail(f"{module}: frozen declarations expose unfrozen types: {', '.join(unfrozen)} "
             "(freeze them in Scripts/api-freeze.json, or keep them out of frozen signatures)")
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=1, sort_keys=True)
        f.write("\n")


def enums(node, prefix=""):
    """Every enum in a dump, by qualified name ("ModelReadiness.State"), mapped to whether it's @nonexhaustive."""
    found = {}
    for child in node.get("children", []):
        if child.get("kind") != "TypeDecl":
            continue
        name = f"{prefix}{child['name']}"
        if child.get("declKind") == "Enum":
            found[name] = "Nonexhaustive" in child.get("declAttributes", [])
        found.update(enums(child, f"{name}."))
    return found


def is_open(enum_name, known):
    """Whether a case added to `enum_name`, as the digester printed it, is additive. The digester may print a nested
    enum by its full or a shorter qualified name ("State" for "ModelReadiness.State"), so the name matches every enum
    it is a suffix of; the case is additive only if there is at least one and all of them are @nonexhaustive."""
    matches = [open_ for name, open_ in known.items() if name == enum_name or name.endswith("." + enum_name)]
    return bool(matches) and all(matches)


def classify(lines, known):
    """Splits digester findings into (additive, breaking)."""
    findings = [line for line in lines if line and not (line.startswith("/*") and line.endswith("*/"))]
    additive, breaking = [], []
    for line in findings:
        added = re.fullmatch(r"EnumElement (\S+)\.[^.\s]+ has been added as a new enum case", line)
        (additive if added and is_open(added.group(1), known) else breaking).append(line)
    return additive, breaking


def diagnose(baseline_path, diagnostics_path):
    with open(baseline_path) as f:
        known = enums(json.load(f)["ABIRoot"])
    with open(diagnostics_path) as f:
        lines = [line.strip() for line in f]
    additive, breaking = classify(lines, known)
    for line in additive:
        print(f"additive: {line}")
    for line in breaking:
        print(f"::error::API break: {line}")
    if breaking:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "prune":
        prune(*sys.argv[2:])
    elif len(sys.argv) == 4 and sys.argv[1] == "diagnose":
        diagnose(*sys.argv[2:])
    else:
        fail("usage: api-freeze.py prune FREEZE_JSON MODULE FULL_DUMP OUT | diagnose BASELINE DIAGNOSTICS")
