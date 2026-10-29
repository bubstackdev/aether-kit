#!/usr/bin/env python3
"""Helpers for Scripts/check-api.sh, the 0.0.x API freeze (docs/api-stability.md).

  api-freeze.py prune FREEZE_JSON MODULE FULL_DUMP OUT
      Reduces a swift-api-digester dump to the module's frozen declarations: drops imports, internal and package
      declarations (`isInternal`), every top-level declaration the freeze doesn't name, and the tool's arguments (they
      hold local paths). Fails if a frozen name is missing or nothing is left.
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


def unfrozen_references(node, freeze):
    """The Aether types a pruned dump's signatures name that the freeze doesn't cover. A frozen signature that exposes
    an unfrozen type would let that type break the frozen API."""
    found = set()
    if node.get("kind") == "TypeNominal":
        for match in re.finditer(r"\bAether\w+(?:\.\w+)+", node.get("printedName", "")):
            name = re.sub(r"\.Type$", "", match.group(0))
            if not is_frozen(name, freeze):
                found.add(name)
    for child in node.get("children", []):
        found |= unfrozen_references(child, freeze)
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
    unfrozen = sorted(unfrozen_references(root, freeze))
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
