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
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=1, sort_keys=True)
        f.write("\n")


def nonexhaustive_enums(node, prefix=""):
    found = set()
    for child in node.get("children", []):
        if child.get("kind") != "TypeDecl":
            continue
        name = f"{prefix}{child['name']}"
        if child.get("declKind") == "Enum" and "Nonexhaustive" in child.get("declAttributes", []):
            found.add(name)
        found |= nonexhaustive_enums(child, f"{name}.")
    return found


def diagnose(baseline_path, diagnostics_path):
    with open(baseline_path) as f:
        open_enums = nonexhaustive_enums(json.load(f)["ABIRoot"])
    with open(diagnostics_path) as f:
        lines = [line.strip() for line in f]
    findings = [line for line in lines if line and not (line.startswith("/*") and line.endswith("*/"))]
    breaking = []
    for line in findings:
        added = re.fullmatch(r"EnumElement (\S+)\.[^.\s]+ has been added as a new enum case", line)
        if added and added.group(1) in open_enums:
            print(f"additive: {line}")
            continue
        breaking.append(line)
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
