#!/usr/bin/env python3
"""Cross-checks the reference ids against mistral-common, upstream's own tokenizer (optional; not in the Forge lock).

    uv run --project Forge --frozen --with mistral-common==1.11.5 \
        python -I Forge/shieldstral/crosscheck_mistral_common.py --reference <reference dir> [--report <json>]

The model card loads its tokenizer with `MistralCommonBackend` (mistral-common over the checkpoint's tekken.json); the
bundle ships tokenizer.json, which the host reads, and reference_shieldstral.py already checks the flattened template
against tokenizer.json's own `apply_chat_template`. This script closes the remaining gap in an ephemeral uv overlay:
for every reference case, mistral-common's encoding of the chat [system: the fixed prompt, user: the body] (pinned
tekken.json) must equal the reference's `input_ids` through the last real token (for the truncated case, the prefix
and the kept body ids, then [/INST]). Exits nonzero when a case differs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shieldstral_graphs as sg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=sg.DEFAULT_SNAPSHOT)
    args = ap.parse_args()
    sg.verify_sources(args.snapshot, weights=False)
    from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_file(str(args.snapshot / "tekken.json"))
    index = json.loads((args.reference / "references.json").read_text())
    results, passed = [], True
    for entry in index["cases"]:
        folder = args.reference / entry["name"]
        case = json.loads((folder / "case.json").read_text())
        ids = sg.read_i32(folder / "input_ids.i32", index["shapes"]["input_ids"])[0]
        n = case["realTokens"]
        body = sg.body(case["instruction"], case["query"], case["document"])
        request = ChatCompletionRequest(messages=[SystemMessage(content=sg.SYSTEM_PROMPT), UserMessage(content=body)])
        theirs = tok.encode_chat_completion(request).tokens
        ours = ids[:n].tolist()
        if case["truncated"]:
            same = theirs[:n - 1] == ours[:n - 1] and theirs[-1] == ours[-1]
        else:
            same = theirs == ours
        passed &= same
        results.append({"case": case["name"], "same": same, "mistralCommonTokens": len(theirs), "realTokens": n})
        print(f"[crosscheck] {'PASS' if same else 'FAIL'} {case['name']}: {len(theirs)} mistral-common ids, "
              f"{n} reference", file=sys.stderr, flush=True)
    report = {"passed": passed, "mistralCommon": importlib.metadata.version("mistral-common"), "cases": results}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("mistral-common tokenization differs")


if __name__ == "__main__":
    main()
