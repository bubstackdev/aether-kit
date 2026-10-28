#!/usr/bin/env python3
"""Gate the Core AI targets behind macOS 27 / iOS 27 so the package can keep its macOS 14 / iOS 17 floor.

    python3 -I Scripts/annotate-floor.py            annotate every gated target in place
    python3 -I Scripts/annotate-floor.py --check    report what is missing and exit 1 if anything is (CI)

SwiftPM has no per-target deployment floor, so every top-level declaration in a target that uses Core AI (or another
OS 27-only API) carries `@available(macOS 27, iOS 27, *)`. Members inherit availability from their type or extension,
so only column-0 declarations are touched: the attribute goes on its own line before the declaration, or before the
first attribute of its attribute run, unless that run already has it. Imports, multi-line strings and block comments
are skipped. The script is idempotent: after a merge or a rebase, take either side of a conflicted file and run it
again. Test targets are never annotated (Swift Testing refuses @Test and @Suite inside a gated type); they build with
`--triple arm64-apple-macosx27.0` instead (docs/deployment-floor.md).
"""
import re
import sys
from pathlib import Path

ATTRIBUTE = "@available(macOS 27, iOS 27, *)"
GATED_TARGETS = [
    "Sources/AetherCoreAI",
    "Sources/AetherLanguageCoreAI",
    "Sources/AetherH3CoreAI",
    "Sources/AetherKreaCoreAI",
    "Sources/AetherFlux2CoreAI",
    "Sources/AetherSpeechCoreAI",
    "Sources/AetherProbe",
]
MODIFIERS = r"(?:(?:public|internal|package|private|fileprivate|final|open|nonisolated|indirect|static)\s+)*"
DECL = re.compile(r"^" + MODIFIERS + r"(?:func|struct|enum|actor|class|extension|protocol|let|var|typealias)\b")
IMPORT = re.compile(r"^(?:@\w+(?:\([^)]*\))?\s+)*import\s")
LEADING_ATTRIBUTES = re.compile(r"^(?:@\w+(?:\((?:[^()]|\([^()]*\))*\))?\s*)+")


def annotate(text: str) -> tuple[str, int]:
    """Returns the annotated text and the number of attributes inserted."""
    out, inserted = [], 0
    in_string = in_comment = False
    run_start = None  # index in `out` where the current attribute run began
    for line in text.split("\n"):
        if in_comment:
            in_comment = "*/" not in line
            out.append(line)
            continue
        if in_string:
            in_string = line.count('"""') % 2 == 0
            out.append(line)
            continue
        at_column0 = bool(line) and not line[0].isspace()
        if at_column0 and line.startswith("/*") and "*/" not in line:
            in_comment = True
            out.append(line)
        elif at_column0 and line.startswith("@") and not IMPORT.match(line):
            if run_start is None:
                run_start = len(out)
            out.append(line)
            rest = LEADING_ATTRIBUTES.sub("", line)  # `@MainActor final class …` on one line
            if rest and DECL.match(rest):
                inserted += finish_run(out, run_start)
                run_start = None
        elif at_column0 and DECL.match(line):
            if run_start is None:
                out.append(ATTRIBUTE)
                inserted += 1
                out.append(line)
            else:
                out.append(line)
                inserted += finish_run(out, run_start)
                run_start = None
        else:
            if at_column0:
                run_start = None
            out.append(line)
        if line.count('"""') % 2:
            in_string = True
    return "\n".join(out), inserted


def finish_run(out: list[str], start: int) -> int:
    if any(line.startswith("@available(") and "macOS 27" in line for line in out[start:]):
        return 0
    out.insert(start, ATTRIBUTE)
    return 1


def main(argv: list[str]) -> int:
    check = "--check" in argv
    repo = Path(__file__).resolve().parent.parent
    total = 0
    for target in GATED_TARGETS:
        files = sorted((repo / target).rglob("*.swift"))
        changed = count = 0
        for path in files:
            new, inserted = annotate(path.read_text())
            if inserted:
                changed += 1
                count += inserted
                if check:
                    print(f"{path.relative_to(repo)}: {inserted} declaration(s) without {ATTRIBUTE}")
                else:
                    path.write_text(new)
        if not check:
            print(f"{target}: {count} declarations in {changed} of {len(files)} files")
        total += count
    if check:
        if total:
            print(f"{total} declaration(s) need the Core AI floor attribute: run python3 -I Scripts/annotate-floor.py")
        else:
            print("Every top-level declaration in the gated targets carries the Core AI floor attribute")
    return 1 if check and total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
