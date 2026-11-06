#!/usr/bin/env python3
"""Upstream-PyTorch CLIP ViT-B/32 references for the Core AI gate (check_clip.py) and the Swift driver gate.

    Forge/.venv/bin/python -I Forge/clip/reference_clip.py --output <new reference dir>

Runs the unmodified transformers `CLIPModel.forward` (pinned weights) on the CPU in float32, on exactly the tensors the
exported graphs are fed: for images, raw RGB on [0,1], [1,3,224,224], already rounded through float16 (the graph's
input dtype), normalized as upstream CLIP does; for texts, the ids and attention mask of transformers' CLIPTokenizer
(padding='max_length', truncation=True). Preprocessing is not under test here: each input is written out, and gates
feed it as is. Files:

    images/<case>/case.json       source, preprocessing recipe, SHA-256 of each file, expected zero-shot label
    images/<case>/input.f32       the graph input [1,3,224,224], little-endian float32 (every value fp16-representable)
    images/<case>/embedding.f32   upstream unit image embedding [1,512], little-endian float32
    texts/tokens.i32              upstream token ids [N,77], little-endian int32 (BOS 49406, EOS 49407 = pad)
    texts/mask.i32                upstream attention mask [N,77] (1 through the first EOS); the graph does not take it
    texts/embedding.f32           upstream unit text embeddings [N,512], little-endian float32
    references.json               the index: image cases, the N texts, the zero-shot label rows, logit scale and
                                  upstream's logits_per_image and argmax per image

Cases: six scikit-image sample images (pinned commit), shortest-side bicubic + center crop as upstream does, and one
seeded synthetic image; texts: the zero-shot prompts ("a photo of a {label}.") plus tokenizer edge cases (empty,
punctuation, case and whitespace, accents and CJK, digits, a long text that truncates). CPU only; no GPU lock needed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clip_graphs as cg  # noqa: E402

LABELS = ["astronaut", "cat", "cup of coffee", "rocket", "motorcycle", "man holding a camera", "dog", "horse",
          "page of text", "car"]
PROMPT = "a photo of a {}."
EDGE_TEXTS = [
    "",
    "hello!",
    "Wow!!  A  DOG\tin the\nsnow",
    "Café naïve résumé, 東京の夜景",
    "There are 12 apples and 3 pears; isn't that 15?",
    "a photo of a cat",
    " ".join(["the quick brown fox jumps over the lazy dog"] * 10),
]
IMAGE_CASES = [
    {"name": "astronaut", "image": "skimage/data/astronaut.png", "label": "astronaut"},
    {"name": "chelsea", "image": "skimage/data/chelsea.png", "label": "cat"},
    {"name": "coffee", "image": "skimage/data/coffee.png", "label": "cup of coffee"},
    {"name": "rocket", "image": "skimage/data/rocket.jpg", "label": "rocket"},
    {"name": "motorcycle", "image": "skimage/data/motorcycle_right.png", "label": "motorcycle"},
    {"name": "camera", "image": "skimage/data/camera.png", "label": "man holding a camera"},
    {"name": "synthetic-7", "synthetic": 7},
]
SHAPES = {"image": [1, 3, cg.SIZE, cg.SIZE], "imageEmbedding": [1, cg.EMBED_DIM]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=cg.DEFAULT_SNAPSHOT)
    ap.add_argument("--upstream", type=Path, default=cg.DEFAULT_UPSTREAM)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    verified = cg.verify_sources(args.snapshot, args.upstream)
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = cg.build_model(args.snapshot)
    tok = cg.tokenizer(args.snapshot)
    scale = cg.logit_scale(model)

    started = time.monotonic()
    texts = [PROMPT.format(label) for label in LABELS] + EDGE_TEXTS
    ids, mask = cg.tokenize(tok, texts)
    if not ((ids == cg.EOS).any(axis=1).all() and (ids[:, 0] == cg.BOS).all()):
        raise SystemExit("a tokenized text lacks BOS or EOS")
    text_emb = cg.upstream_embeddings(model, None, torch.from_numpy(ids), torch.from_numpy(mask))["text"].numpy()
    (args.output / "texts").mkdir(parents=True)
    cg.write_i32(args.output / "texts/tokens.i32", ids)
    cg.write_i32(args.output / "texts/mask.i32", mask)
    cg.write_f32(args.output / "texts/embedding.f32", text_emb)
    label_rows = list(range(len(LABELS)))
    text_seconds = round(time.monotonic() - started, 2)

    index = {
        "schema": 1, "generator": "Forge/clip/reference_clip.py",
        "scripts": {p.name: cg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": cg.REPO, "revision": cg.REVISION, "verified": verified,
                   "images": {"repository": cg.IMAGES_REPO, "commit": cg.IMAGES_COMMIT}},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "transformers", "numpy",
                                                                "pillow")},
        "host": {"machine": platform.machine(), "python": platform.python_version()},
        "precision": "float32 on the CPU (transformers CLIPModel.forward, unmodified, sdpa attention)",
        "shapes": {**SHAPES, "tokens": [len(texts), cg.TEXT_LENGTH], "textEmbedding": [len(texts), cg.EMBED_DIM]},
        "layout": "row-major, little-endian float32 (.f32) and int32 (.i32)",
        "graphInput": {"image": "input.f32 converted to float16 (exact: every value is fp16-representable)",
                       "tokens": "texts/tokens.i32 rows, in buckets of the text graph's static batch"},
        "texts": {"strings": texts, "files": {n: cg.sha256_file(args.output / "texts" / n)
                                              for n in ("tokens.i32", "mask.i32", "embedding.f32")},
                  "tokenizer": "transformers CLIPTokenizer (pinned snapshot vocab.json/merges.txt), "
                               "padding='max_length', max_length=77, truncation=True",
                  "seconds": text_seconds},
        "zeroShot": {"labels": LABELS, "prompt": PROMPT, "textRows": label_rows, "logitScale": scale,
                     "rule": "softmax over labels of logitScale * cosine(image, text)", "perImage": {}},
        "cases": [],
    }
    label_ids, label_mask = torch.from_numpy(ids[label_rows]), torch.from_numpy(mask[label_rows])
    for case in IMAGE_CASES:
        started = time.monotonic()
        out = args.output / "images" / case["name"]
        out.mkdir(parents=True)
        if "image" in case:
            x = cg.image_tensor(args.upstream / "images" / case["image"])
            source = {"kind": "image", "upstreamFile": case["image"],
                      "sha256": cg.PINS["referenceImages"]["files"][case["image"]],
                      "preprocess": "Pillow convert('RGB'), shortest side to 224 BICUBIC (long side int(224*long/"
                                    "short)), center crop 224 (offsets round((side-224)/2)), /255, NCHW, rounded "
                                    "through float16"}
        else:
            x = cg.synthetic_tensor(case["synthetic"])
            source = {"kind": "synthetic", "seed": case["synthetic"], "recipe": "clip_graphs.synthetic_tensor"}
        # The full joint forward against the label prompts: the image embedding and upstream's own logits.
        joint = cg.upstream_embeddings(model, torch.from_numpy(x), label_ids, label_mask)
        image_emb = joint["image"].numpy()
        alone = cg.upstream_embeddings(model, torch.from_numpy(x), None)["image"].numpy()
        if not np.array_equal(alone, image_emb) or not np.allclose(joint["text"].numpy(), text_emb[label_rows],
                                                                   rtol=0, atol=1e-6):
            raise SystemExit(f"{case['name']}: the joint forward and the single-tower features disagree")
        logits = joint["logitsPerImage"].numpy()[0]
        if not (np.isfinite(image_emb).all() and np.isfinite(logits).all()):
            raise SystemExit(f"{case['name']}: non-finite upstream output")
        argmax = int(np.argmax(logits))
        if "label" in case and LABELS[argmax] != case["label"]:
            raise SystemExit(f"{case['name']}: upstream picks {LABELS[argmax]!r}, not {case['label']!r}")
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        cg.write_f32(out / "input.f32", x)
        cg.write_f32(out / "embedding.f32", image_emb)
        record = {"name": case["name"], "source": source, "shapes": SHAPES,
                  "files": {n: cg.sha256_file(out / n) for n in ("input.f32", "embedding.f32")},
                  "expectedLabel": case.get("label"), "zeroShot": {"argmax": argmax, "label": LABELS[argmax],
                                                                   "logits": logits.tolist(),
                                                                   "probability": float(probs[argmax])},
                  "seconds": round(time.monotonic() - started, 2)}
        (out / "case.json").write_text(json.dumps(record, indent=2) + "\n")
        index["zeroShot"]["perImage"][case["name"]] = {"argmax": argmax, "logits": logits.tolist()}
        index["cases"].append({"name": case["name"], "caseJSONSHA256": cg.sha256_file(out / "case.json")})
        print(f"[reference] {case['name']}: {LABELS[argmax]} ({probs[argmax]:.3f}), {record['seconds']} s",
              file=sys.stderr, flush=True)
    index["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    (args.output / "references.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
