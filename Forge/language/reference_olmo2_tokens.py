#!/usr/bin/env python3
"""Record OLMo2 chat-token parity fixtures using the pinned upstream tokenizer."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoTokenizer
from reference_chat_tokens import TEXT_CASES, DECODE_REPLY, upstream_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    cases = []
    for name, messages, thinking in TEXT_CASES:
        text = tokenizer.apply_chat_template(upstream_messages(messages, False), tokenize=False,
                                             add_generation_prompt=True, enable_thinking=thinking)
        cases.append(dict(name=name, messages=messages, thinking=thinking, text=text,
                          tokenIDs=tokenizer.encode(text, add_special_tokens=False)))
    reply = tokenizer.encode(DECODE_REPLY, add_special_tokens=False) + [tokenizer.eos_token_id]
    record = dict(repository='allenai/OLMo-2-0425-1B-Instruct', revision='48d788eca847d4d7548f375ad03d3c9312f6139e',
                  tokenizerSHA256=hashlib.sha256((args.checkpoint/'tokenizer.json').read_bytes()).hexdigest(),
                  cases=cases, decode=dict(tokenIDs=reply, text=tokenizer.decode(reply, skip_special_tokens=True)))
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
