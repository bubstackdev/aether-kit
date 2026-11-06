#!/usr/bin/env python3
"""Eager CPU self-test of the export graphs, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/clip/selftest_clip.py [--reference <reference dir>] [--report <json>]

- With the towers in float32, `ImageGraph` and `TextGraph` equal upstream's own `CLIPModel.forward` embeddings bit
  for bit (so in-graph normalization, mask-free causal pooling at the first EOS and the graphs' L2 normalization change
  nothing), with and without upstream's attention mask.
- The text graph is pad-agnostic: padding with '!' (0) instead of EOS gives the identical embedding.
- The exported precision (float16 towers), run eagerly, is within the gate's bar (check_clip.py).
- The reference preprocessing (`clip_graphs.resize_crop`) equals openai/CLIP's own transform (torchvision Resize(224,
  BICUBIC) + CenterCrop(224) on a PIL image) byte for byte on every reference image.
- The tokenizer facts the Swift `CLIPTokenizer` relies on hold for the pinned vocabulary (BOS 49406, EOS 49407, '!'
  0), and '!' is an ordinary piece for this tokenizer ('hello!' -> 3306, 256), not the pad token.
- `--sample-manifest`'s manifest names the `clip-joint@1` roles (MANIFEST.md §1.3) and matches the graphs' I/O.
With `--reference`, the cases are reference_clip.py's (and upstream's float32 output must reproduce them exactly);
otherwise one seeded synthetic image and the edge texts. Exits nonzero when a check fails. CPU only; a few seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clip_graphs as cg  # noqa: E402
import export_clip  # noqa: E402
import reference_clip  # noqa: E402
from check_clip import COSINE_BAR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=cg.DEFAULT_SNAPSHOT)
    ap.add_argument("--upstream", type=Path, default=cg.DEFAULT_UPSTREAM)
    args = ap.parse_args()
    cg.verify_sources(args.snapshot, args.upstream)
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed)})
        print(f"[selftest] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    tok = cg.tokenizer(args.snapshot)
    images, recorded_text = [], None
    if args.reference:
        index = json.loads((args.reference / "references.json").read_text())
        for entry in index["cases"]:
            folder = args.reference / "images" / entry["name"]
            images.append((entry["name"], cg.read_f32(folder / "input.f32", index["shapes"]["image"]),
                           cg.read_f32(folder / "embedding.f32", index["shapes"]["imageEmbedding"])))
        texts = index["texts"]["strings"]
        ids = cg.read_i32(args.reference / "texts/tokens.i32", index["shapes"]["tokens"])
        again, _ = cg.tokenize(tok, texts)
        record("tokens.reproduced", float(np.array_equal(again, ids)), 1.0, np.array_equal(again, ids))
        recorded_text = cg.read_f32(args.reference / "texts/embedding.f32", index["shapes"]["textEmbedding"])
    else:
        images.append(("synthetic-3", cg.synthetic_tensor(3), None))
        texts = reference_clip.EDGE_TEXTS
        ids, _ = cg.tokenize(tok, texts)
    ids_t = torch.from_numpy(ids)
    mask_t = torch.from_numpy(cg.tokenize(tok, texts)[1])

    upstream = cg.build_model(args.snapshot)
    exact_image = cg.ImageGraph(cg.build_model(args.snapshot), torch.float32).eval()
    exact_text = cg.TextGraph(cg.build_model(args.snapshot), torch.float32).eval()
    mixed_image = cg.ImageGraph(cg.build_model(args.snapshot), torch.float16).eval()
    mixed_text = cg.TextGraph(cg.build_model(args.snapshot), torch.float16).eval()

    # Text: float32 exact (masked and unmasked upstream), pad-agnostic, float16 within the bar.
    want_masked = cg.upstream_embeddings(upstream, None, ids_t, mask_t)["text"]
    want_plain = cg.upstream_embeddings(upstream, None, ids_t, None)["text"]
    got = exact_text(ids_t)
    record("text.float32.exact.masked", float(torch.equal(got, want_masked)), 1.0, torch.equal(got, want_masked))
    record("text.float32.exact.unmasked", float(torch.equal(got, want_plain)), 1.0, torch.equal(got, want_plain))
    if recorded_text is not None:
        same = np.array_equal(want_masked.numpy(), recorded_text)
        record("text.reference.reproduced", float(same), 1.0, same)
    bang = ids.copy()
    for row in bang:
        first = int(np.argmax(row == cg.EOS))
        row[first + 1:] = 0
    same = torch.equal(exact_text(torch.from_numpy(bang)), got)
    record("text.padAgnostic", float(same), 1.0, same)
    cos = cg.cosine(mixed_text(ids_t).numpy(), want_masked.numpy())
    record("text.float16.cosine.min", round(float(cos.min()), 7), COSINE_BAR, bool((cos >= COSINE_BAR).all()))

    # Images: float32 exact against the joint forward, float16 within the bar.
    label_ids, label_mask = cg.tokenize(tok, [reference_clip.PROMPT.format(x) for x in reference_clip.LABELS])
    for name, x, recorded in images:
        image = torch.from_numpy(x)
        joint = cg.upstream_embeddings(upstream, image, torch.from_numpy(label_ids), torch.from_numpy(label_mask))
        got = exact_image(image.half())
        record(f"image.{name}.float32.exact", float(torch.equal(got, joint["image"])), 1.0,
               torch.equal(got, joint["image"]))
        if recorded is not None:
            same = np.array_equal(joint["image"].numpy(), recorded)
            record(f"image.{name}.reference.reproduced", float(same), 1.0, same)
        c = float(cg.cosine(mixed_image(image.half()).numpy(), joint["image"].numpy())[0])
        record(f"image.{name}.float16.cosine", round(c, 7), COSINE_BAR, c >= COSINE_BAR)

    # Reference preprocessing equals openai/CLIP's transform.
    import torchvision.transforms as T
    from PIL import Image
    openai = T.Compose([T.Resize(cg.SIZE, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(cg.SIZE)])
    for relative in cg.PINS["referenceImages"]["files"]:
        path = args.upstream / "images" / relative
        with Image.open(path) as im:
            want = np.asarray(openai(im.convert("RGB")))
        same = np.array_equal(np.asarray(cg.resize_crop(path)), want)
        record(f"preprocess.{Path(relative).stem}.openaiTransform", float(same), 1.0, same)

    # Tokenizer facts for the Swift CLIPTokenizer.
    vocab = json.loads((args.snapshot / "vocab.json").read_text())
    facts = (vocab.get("<|startoftext|>") == cg.BOS and vocab.get("<|endoftext|>") == cg.EOS and vocab.get("!") == 0
             and tok.pad_token_id == cg.EOS)
    record("tokenizer.ids", float(facts), 1.0, facts)
    bang_ids = tok("hello!").input_ids
    record("tokenizer.bangIsOrdinary", float(bang_ids == [cg.BOS, 3306, 256, cg.EOS]), 1.0,
           bang_ids == [cg.BOS, 3306, 256, cg.EOS])

    sample = export_clip.sample_manifest("macos")
    image_spec, text_spec = sample["graphs"]["imageEncoder"], sample["graphs"]["textEncoder"]
    roles = (sample["driver"] == "clip-joint@1" and image_spec["inputs"] == {"image": "image"}
             and image_spec["outputs"] == {"imageEmbedding": "image_embedding"}
             and text_spec["inputs"] == {"tokens": "tokens"} and text_spec["outputs"] == {"textEmbedding": "text_embedding"}
             and export_clip.IMAGE_IO == (["image"], ["image_embedding"])
             and export_clip.TEXT_IO == (["tokens"], ["text_embedding"])
             and sample["postprocess"]["kind"] == "embedding"
             and sample["tokenizer"]["text"]["padTo"] == cg.TEXT_LENGTH
             and abs(sample["postprocess"]["embedding"]["logitScale"] - cg.logit_scale(upstream)) < 1e-4)
    record("manifest.roles", float(roles), 1.0, roles)
    passed = all(c["passed"] for c in checks)
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()
