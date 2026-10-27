#!/usr/bin/env python3
"""Upstream chat-template oracle for AetherLanguageTokenizers.

Renders each case with the checkpoint's own chat template through upstream Transformers
(`apply_chat_template`, generation prompt on, `enable_thinking` as the case says) and records the
rendered text, its token IDs (no special tokens added) and, for the decode check, the plain decode
of a short reply. Vision-language checkpoints render through their processor, whose template emits
one `<|vision_start|><|image_pad|><|vision_end|>` per image before expansion.

Reads local snapshots only (HF_HUB_OFFLINE); writes only `--output`. Development tooling: Python is
absent from shipping targets.

    reference_chat_tokens.py --output Tests/AetherLanguageTokenizersTests/Fixtures/chat-oracle.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

# Repository, pinned revision, and whether the checkpoint is a vision-language model (processor template).
MODELS = [
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", False),
    ("huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated", "c9bd464550d4078c72af0dd22aa18d0437868ce3", False),
    ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", True),
    ("huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated", "ce72a7c22aacb493fb94478de3bfbe834c61844a", True),
]

# Messages use the AetherLanguageTokenizers shape: parts are {"text": ...} or {"image": true}.
TEXT_CASES = [
    ("user-only", [{"role": "user", "parts": [{"text": "A lighthouse keeper's cat at dusk."}]}], False),
    ("system-and-user", [
        {"role": "system", "parts": [{"text": "You describe fictional adult characters for an image generator."}]},
        {"role": "user", "parts": [{"text": "Field: look\nCurrent: silver braid, amber eyes"}]},
    ], False),
    ("unicode-and-newlines", [{"role": "user", "parts": [{"text": "Café crème — naïve résumé 🌙\n\n  indented line\ttab"}]}], False),
    ("multi-turn", [
        {"role": "system", "parts": [{"text": "Answer in one line."}]},
        {"role": "user", "parts": [{"text": "Name a colour."}]},
        {"role": "assistant", "parts": [{"text": "Teal."}]},
        {"role": "user", "parts": [{"text": "Another."}]},
    ], False),
    ("thinking-on", [{"role": "user", "parts": [{"text": "Count to three."}]}], True),
]
IMAGE_CASES = [
    ("one-image-first", [{"role": "user", "parts": [{"image": True}, {"text": "Describe her look in one line."}]}], False),
    ("two-images-after-text", [
        {"role": "system", "parts": [{"text": "You describe fictional adult characters."}]},
        {"role": "user", "parts": [{"text": "Describe the outfit."}, {"image": True}, {"image": True}]},
    ], False),
]
DECODE_REPLY = "Silver braid, amber eyes, a weathered blue coat — calm. 🌙"


def snapshot(hub: Path, repository: str, revision: str) -> Path:
    path = hub / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        sys.exit(f"missing local snapshot {path}")
    return path


def upstream_messages(messages: list[dict], vision: bool) -> list[dict]:
    """Text-only messages pass plain string content (every template's common path); messages with
    images pass typed content parts, as Qwen3-VL's processor template expects."""
    result = []
    for message in messages:
        parts = message["parts"]
        if not vision or not any("image" in part for part in parts):
            result.append({"role": message["role"], "content": "".join(part["text"] for part in parts)})
        else:
            result.append({"role": message["role"], "content": [
                {"type": "image"} if "image" in part else {"type": "text", "text": part["text"]} for part in parts
            ]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists; choose a new path")
    models = []
    for repository, revision, vision in MODELS:
        folder = snapshot(args.hub, repository, revision)
        tokenizer = AutoTokenizer.from_pretrained(folder)
        renderer = AutoProcessor.from_pretrained(folder) if vision else tokenizer
        cases = []
        for name, messages, thinking in TEXT_CASES + (IMAGE_CASES if vision else []):
            text = renderer.apply_chat_template(upstream_messages(messages, vision), tokenize=False,
                                                add_generation_prompt=True, enable_thinking=thinking)
            ids = tokenizer.encode(text, add_special_tokens=False)
            cases.append({"name": name, "messages": messages, "thinking": thinking, "text": text, "tokenIDs": ids})
        reply = tokenizer.encode(DECODE_REPLY, add_special_tokens=False)
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        models.append({
            "repository": repository, "revision": revision, "vision": vision, "cases": cases,
            "decode": {"tokenIDs": reply + [end], "text": tokenizer.decode(reply + [end], skip_special_tokens=True)},
        })
    oracle = {
        "schemaVersion": 1,
        "generator": "Forge/language/reference_chat_tokens.py",
        "environment": {"python": sys.version.split()[0], "transformers": importlib.metadata.version("transformers"),
                        "tokenizers": importlib.metadata.version("tokenizers")},
        "models": models,
    }
    args.output.write_text(json.dumps(oracle, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
