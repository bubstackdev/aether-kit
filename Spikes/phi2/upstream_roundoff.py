#!/usr/bin/env python3
"""Measure upstream float32 full-sequence versus cached/chunked execution.

Run with Forge's Python: upstream_roundoff.py CHECKPOINT REFERENCE_DIR OUTPUT_JSON.
This baseline executes only Transformers, independently of the port.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    record = json.loads((args.reference / 'reference.json').read_text())
    expected = np.fromfile(args.reference / record['forcedLogits']['file'], dtype='<f4').reshape(
        -1, record['forcedLogits']['vocab']).astype(np.float64)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32).eval()
    results = {}
    with torch.no_grad():
        for width in (7, 8):
            cache = None
            prompt, rows = record['promptTokens'], []
            for start in range(0, len(prompt), width):
                out = model(torch.tensor([prompt[start:start + width]]), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            rows.append(out.logits[0, -1].double().numpy())
            for token in record['forcedTokens'][:-1]:
                out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                rows.append(out.logits[0, -1].double().numpy())
            actual = np.stack(rows)
            cosine = np.sum(actual * expected, axis=1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
            results[str(width)] = {'maxAbs': float(np.abs(actual - expected).max()),
                'minCosine': float(cosine.min()), 'rows': len(rows),
                'argmaxEqualRows': int((actual.argmax(axis=1) == expected.argmax(axis=1)).sum())}
    args.output.write_text(json.dumps({'source': record['source'], 'versions': record['versions'],
        'promptTokens': len(prompt), 'oracle': 'Transformers float32 CPU: full sequence versus cached chunks',
        'chunks': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
