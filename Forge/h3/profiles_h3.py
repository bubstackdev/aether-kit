"""MiniMax H3's profiles and the bundle manifest (MANIFEST.md §5.8, §5.10), standard library only.

The exporter (`export_h3.py`) and the manifest-only tools (`extend_h3.py`, `record_envelopes_h3.py`) share this module:
the constants the manifest states, the frame grid (`Geometry`), the profile table (canvas, frames, the recipes each
profile offers, its memory bar and the physical memory it needs) and `manifest(meta, files)`, which builds
`.aether/aether.json` from `work/build.json` and the bundle's file list. It imports nothing outside the standard
library, so a manifest can be rebuilt (and tested) without torch or the checkpoint.

    python3 -I Forge/h3/profiles_h3.py --sample-manifest > Tests/AetherCoreTests/Fixtures/h3/minimax-h3.json
"""
from __future__ import annotations

import json
import struct
import sys

REPO = "MiniMaxAI/MiniMax-H3"
REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DIFFUSERS_COMMIT = "2a1433913c7044f34fa4767dd84b28bba0bb0ecf"   # tag v0.42.1; see Forge/pyproject.toml
ADAPTER_REPO = "lightx2v/Minimax-h3-Turbo"
ADAPTER_REVISION = "3ec17a324ced54151364f24f8b5fb6bf7e26414f"
ADAPTER_FILE = "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
ADAPTER_SHA256 = "e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f"
LORA_RANK, LORA_ALPHA = 128, 8.0
LORA_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")

BLOCKS, CONDITIONER_LAYERS, TAP = 50, 50, 50     # the conditioner taps hidden_states[50], unnormalised
HIDDEN, TEXT_DIM, TEMB_DIM, HEADS, HEAD_DIM = 5376, 5120, 2688, 56, 128
ROWS = (64, 40960)                               # the body's dynamic row range
TEXT_TOKENS = (2, 4096)
CARDINALITY = (1, 3)                             # distinct timesteps per step (table rows 3k)
VIDEO_TILE, WINDOW_LATENTS, WINDOW_FRAMES, WINDOW_STRIDE = 256, 7, 22, 5
PIXEL_MEAN, PIXEL_STD, ENCODE_SEED = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 42
SAMPLE_RATE, AUDIO_HOP, AUDIO_CHANNELS = 32000, 800, 2
# Schedules: grid points and the video/audio shifts of the 544p Turbo adapter. FL2VA's keyframe rows sit at 0.999.
RECIPES = {"turbo8": (9, 12.0, 3.0), "turbo4": (5, 12.0, 3.0)}
FL2VA_CONDITION_TIMESTEP = 0.999

GIB = 1 << 30
# The memory bars (0.0.x's measured-admission method): a profile inside today's region (at most 960×544's pixels and
# 124 frames) is held to 18 GiB; a larger canvas or a longer clip to 24 GiB, and needs a Mac with 64 GiB of RAM.
BAR, EXTENDED_BAR, EXTENDED_PHYSICAL = 18 * GIB, 24 * GIB, 64 * GIB
REGION_PIXELS, REGION_FRAMES = 960 * 544, 124


class Geometry:
    """H3's frame grid (the prototype's `H3Geometry`): requested frames snap up to 17n+5 at 24 fps; 16× spatial and
    the (1, 2, 2) patch give the packed video rows; 40 audio latents a second."""

    def __init__(self, width: int = 960, height: int = 544, requested_frames: int = 120):
        assert width % 32 == 0 and height % 32 == 0 and width * height <= 1_032_192, (width, height)
        n = (max(0, requested_frames - 5) + 16) // 17
        self.width, self.height, self.frames = width, height, 17 * n + 5
        self.latent_frames, self.latent_height, self.latent_width = 5 * n + 2, height // 16, width // 16
        self.video_rows = self.latent_frames * (height // 32) * (width // 32)
        self.keyframe_rows = (height // 32) * (width // 32)
        self.audio_latents = round(self.frames / 24 * 40)   # Python rounds half to even, as Swift's .toNearestOrEven

    @property
    def key(self) -> str:
        return f"{self.width}x{self.height}x{self.frames}"

    @property
    def canvas(self) -> str:
        return f"{self.width}x{self.height}"

    def dict(self) -> dict:
        return {"width": self.width, "height": self.height, "frames": self.frames, "latentFrames": self.latent_frames,
                "videoRows": self.video_rows, "keyframeRows": self.keyframe_rows, "audioLatents": self.audio_latents}


class Profile(Geometry):
    """A profile the export offers: its geometry, the recipes it is measured for, and its memory class."""

    def __init__(self, width: int, height: int, requested_frames: int, recipes: tuple[str, ...]):
        super().__init__(width, height, requested_frames)
        assert recipes and set(recipes) <= set(RECIPES), recipes
        self.recipes = tuple(recipes)
        self.extended = self.width * self.height > REGION_PIXELS or self.frames > REGION_FRAMES

    @property
    def memory(self) -> dict:
        return {"barBytes": EXTENDED_BAR if self.extended else BAR,
                "minimumPhysicalMemoryBytes": EXTENDED_PHYSICAL if self.extended else 0}


# The profiles, the first the default. 5 s is 124 frames, 10 s 243.
PROFILES = [
    Profile(960, 544, 120, ("turbo8", "turbo4")),
    Profile(544, 960, 120, ("turbo8", "turbo4")),
    Profile(960, 544, 240, ("turbo8", "turbo4")),
    Profile(544, 960, 240, ("turbo8", "turbo4")),
    Profile(704, 704, 120, ("turbo8",)),
    Profile(1280, 704, 120, ("turbo8",)),
    Profile(704, 1280, 120, ("turbo8",)),
]
CANVASES = sorted({p.canvas for p in PROFILES})
AUDIO_LATENTS = sorted({p.audio_latents for p in PROFILES})


def f32(value: float) -> float:
    """`value` rounded to float32 (as numpy's float32 would hold it)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def cardinalities(meta: dict, recipes: tuple[str, ...], keyframes: bool) -> list[int]:
    """The modulation cardinalities a profile's renders bind: each recipe's largest step cardinality (the driver pads
    every step to it), per task."""
    tasks = ("t2va", "fl2va") if keyframes else ("t2va",)
    found = set()
    for recipe in recipes:
        for task in tasks:
            steps = meta["tables"]["recipes"].get(f"{recipe}/{task}", {}).get("steps", [])
            if steps:
                found.add(max(len(s["timesteps"]) for s in steps))
    return sorted(found)


def profile_entry(p: Profile, meta: dict) -> dict | None:
    """The manifest entry of `p`, or None when the export lacks its audio decoder. Its keyframe graphs (the canvas's
    encoder tail and, when the export has the tower, its vision graph) make it an FL2VA profile; without them it is
    text-to-video only. `shapes` is the profile's ShapeProfile: the graph roles it runs, and the shapes of the inputs
    that vary between its renders."""
    if str(p.audio_latents) not in meta["audio-decoder"]["fixed"]:
        return None
    tails = meta.get("video-encoder", {}).get("tails", {})
    keyframes = p.canvas in tails and ("vision" not in meta or p.canvas in meta["vision"].get("functions", {}))
    ks = cardinalities(meta, p.recipes, keyframes)
    shapes = {
        "conditionerLayer": {}, "prepareText": {}, "preparePack": {},
        "body": {"table": [[6, 3 * k, HIDDEN] for k in ks]},
        "heads": {"temb": [[k, TEMB_DIM] for k in ks]},
        "videoDecoder": {},
        f"audioDecoder{p.audio_latents}": {},
    }
    entry = {"key": p.key, **p.dict(), "audioDecoder": f"audioDecoder{p.audio_latents}"}
    if keyframes:
        entry["keyframeEncoder"] = f"videoEncoderTail{p.canvas}"
        shapes["videoEncoderTile"] = {}
        shapes[f"videoEncoderTail{p.canvas}"] = {}
        if "vision" in meta:
            shapes[f"conditionerVision{p.canvas}"] = {}
    entry.update({"recipes": list(p.recipes), "memory": p.memory, "shapes": shapes})
    return entry


# MARK: the manifest (MANIFEST.md §5.8, §5.10)

def manifest(meta: dict, files: dict[str, dict], profiles: list[Profile] | None = None) -> dict:
    """The bundle manifest from `build.json` (`meta`) and the bundle's files. Only graphs the export made are named:
    an encoder tail per canvas in `meta["video-encoder"]["tails"]`, a vision tower per canvas in
    `meta["vision"]["functions"]`, an audio decoder per latent count in `meta["audio-decoder"]["fixed"]`, and only the
    profiles whose audio decoder exists."""
    # No graph takes Core AI's `expectFrequentReshapes` hint: with it the GPU's bf16 numerics drift from the path
    # check_h3.py qualifies against the original modules (the Swift driver's step-0 head velocities moved 5.4% from it,
    # 10.8% at step 1); without it the driver is bit-identical to that path (G4 step 2b). A profile pins the body's
    # shapes, so each specializes once; the conditioner specializes per prompt length, within its phase.
    gpu = {"compute": "gpu", "expectFrequentReshapes": False}
    fixed = {"compute": "gpu", "expectFrequentReshapes": False}
    # `name` is this graph's input tensor; `from` is `<role>.<the producing graph's output tensor>`.
    from_ = lambda name, ref: {"name": name, "source": {"from": ref}}  # noqa: E731

    def weights(table: str, count: int, blocks: int) -> dict:
        return {"table": f"main:weights/{table}.json", "inputs": [f"w{i}" for i in range(count)], "blocks": blocks}

    body_inputs = len(meta["body"]["check"]["bindings"])
    text_inputs = len(meta["prepare"]["textInputs"])
    layer_inputs = len(meta["conditioner"]["inputs"])
    graphs = {
        "conditionerLayer": {"file": "main:models/conditioner-layer.aimodel", "function": "main",
                             "inputs": {"hidden": "hidden", "cos": "cos", "sin": "sin"}, "outputs": {"hidden": "hidden_out"},
                             "weights": weights("conditioner", layer_inputs, CONDITIONER_LAYERS), "residency": "perPrompt", **gpu},
        "prepareText": {"file": "main:models/prepare-text.aimodel", "function": "main", "inputs": {"text": "text"},
                        "outputs": {"refined": "refined"}, "weights": weights("prepare-text", text_inputs, 1),
                        "residency": "perPrompt", **gpu},
        "preparePack": {"file": "main:models/prepare-pack.aimodel", "function": "main",
                        "inputs": {"video": "video", "audio": "audio", "text": from_("text", "prepareText.refined"), "gather": "gather"},
                        "outputs": {"hidden": "hidden"}, "residency": "perStep", **gpu},
        "body": {"file": "main:models/body.aimodel", "function": "main",
                 "inputs": {"hidden": from_("hidden", "preparePack.hidden"), "table": "table", "tags": "tags", "cos": "cos", "sin": "sin"},
                 "outputs": {"hidden": "hidden_out"}, "weights": weights("body", body_inputs, BLOCKS),
                 "residency": "perStep", **gpu},
        "heads": {"file": "main:models/heads.aimodel", "function": "main",
                  "inputs": {"hidden": from_("hidden", "body.hidden_out"), "temb": "temb", "indices": "indices",
                             "videoIndices": "video_indices", "audioIndices": "audio_indices"},
                  "outputs": {"video": "video", "audio": "audio"}, "residency": "perStep", **gpu},
        "videoDecoder": {"file": "main:models/video-decoder-tile256-bf16.aimodel", "function": "main", "inputs": {"rows": "rows"},
                         "outputs": {"pixels": "pixels"}, "residency": "perImage", **fixed},
        "videoEncoderTile": {"file": "main:models/video-encoder-tile256.aimodel", "function": "main", "inputs": {"pixels": "pixels"},
                             "outputs": {"moments": "moments"}, "residency": "perPrompt", **fixed},
    }
    tails = meta.get("video-encoder", {}).get("tails", {})
    towers = meta.get("vision", {}).get("functions", {})
    for key in sorted(set(tails) | set(towers)):
        if key in tails:
            graphs[f"videoEncoderTail{key}"] = {"file": f"main:models/video-encoder-tail-{key}.aimodel", "function": "main",
                                                "inputs": {"tiles": "tiles"}, "outputs": {"condition": "condition"},
                                                "residency": "perPrompt", **fixed}
        if "vision" in meta and key in towers:
            graphs[f"conditionerVision{key}"] = {
                "file": f"main:models/conditioner-vision-{key}.aimodel", "function": "main",
                "inputs": {"pixels": "pixels", "posIndex": "pos_index", "posWeight": "pos_weight", "patchRC": "patch_rc"},
                "outputs": {"imageEmbeds": "image_embeds", "deepstack": "deepstack"},
                "weights": weights("vision", len(meta["vision"]["inputs"]), 1), "residency": "perPrompt", **fixed}
    for na in sorted(meta["audio-decoder"]["fixed"], key=int):
        graphs[f"audioDecoder{na}"] = {"file": f"main:models/audio-decoder-na{na}.aimodel", "function": "main",
                                       "inputs": {"rows": "rows"}, "outputs": {"pcm": "pcm"}, "residency": "perImage", **fixed}
    entries = [e for e in (profile_entry(p, meta) for p in (profiles or PROFILES)) if e is not None]
    assert entries, "no profile has its audio decoder in this export"
    c, v, a = meta["conditioner"], meta["video-decoder"], meta["audio-decoder"]
    h3 = {
        "tasks": ["t2va", "fl2va"],
        "profiles": entries,
        "limits": {"rows": list(ROWS), "cardinality": list(CARDINALITY), "textTokens": list(TEXT_TOKENS)},
        "conditioner": {"layers": CONDITIONER_LAYERS, "tap": TAP, "embedding": c["embedding"],
                        "rope": {k: c["rope"][k] for k in ("theta", "mropeSection", "headDim", "dtype")},
                        "ropeTables": {"cos": "main:tables/conditioner/rope-cos.raw", "sin": "main:tables/conditioner/rope-sin.raw",
                                       "rows": TEXT_TOKENS[1]}},
        "recipes": {k: {"video": r["video"], "audio": r["audio"], "steps": [{"timesteps": s["timesteps"], "temb": s["temb"]["file"],
                                                                            "blocks": s["blocks"]} for s in r["steps"]]}
                    for k, r in meta["tables"]["recipes"].items()},
        "defaultRecipe": "turbo8",
        "adapter": meta["body-weights"]["adapter"],
        "video": {"tile": v["tile"], "windowLatents": v["windowLatents"], "windowFrames": v["windowFrames"],
                  "windowStride": v["windowStride"], "precision": v["precision"], "packing": "patch2x2ChannelMajor",
                  "rowChannels": 96, "noise": {"kind": "splitmix64BoxMuller", "order": "sourceVideoThenAudio", "dtype": "float32"}},
        "keyframe": {"tileSize": meta["video-encoder"]["tileSize"], "seed": meta["video-encoder"]["seed"],
                     "pixelMean": list(PIXEL_MEAN), "pixelStd": list(PIXEL_STD), "conditionTimestep": FL2VA_CONDITION_TIMESTEP},
        "audio": {"sampleRate": a["sampleRate"], "hop": a["hop"], "channels": a["channels"], "rowChannels": 32},
        "platforms": ["macos"],
    }
    if "vision" in meta:
        vm = meta["vision"]
        h3["vision"] = {k: vm[k] for k in ("patchSize", "mergeSize", "temporalPatchSize", "tableSide", "deepstackLevels", "presentation")}
    return {
        "schema": 1,
        "id": "minimax-h3",
        "task": "text-to-video-audio",
        "driver": "minimax-h3@1",
        "variant": {"platform": "macos"},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **m} for p, m in sorted(files.items())]}},
        "graphs": graphs,
        "tokenizer": {"source": {"dir": "main:tokenizer"}, "chatTemplate": "main:tokenizer/chat_template.jinja",
                      "renderer": "minimax-h3-prompt", "addSpecialTokens": False},
        "h3": h3,
        "precision": {"weights": "bf16 (transformer, conditioner, refiner, Turbo LoRA unmerged)", "activations": "bf16",
                      "components": {"heads": "float32", "videoDecoder": "bf16 weights and compute, float32 in/out",
                                     "audioDecoder": "float32", "videoEncoder": "float32"}},
        "license": "LicenseRef-MiniMax-H3-Community-License",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True, "territory": "excludes the EU, UK, Republic of Korea and USA",
                      "commercialRevenueCapUSD": 20_000_000, "uiAttribution": "MiniMax H3",
                      "encoder": "Qwen3-VL-32B (Apache-2.0)"},
        "provenance": {
            "source": REPO, "sourceRevision": REVISION,
            "adapter": f"{ADAPTER_REPO}@{ADAPTER_REVISION}:{ADAPTER_FILE}",
            "producer": "aether-kit Forge/h3/export_h3.py",
            "refs": {"diffusersCommit": DIFFUSERS_COMMIT, "forgeLockSHA256": meta.get("forgeLockSHA256", "0" * 64),
                     "exportReport": "export-report.json"},
        },
    }


def sample_meta() -> dict:
    """`build.json` with placeholder values: the manifest's shape without an export (the Swift decoding fixture)."""
    plans = {("turbo8", "t2va"): 9, ("turbo8", "fl2va"): 9, ("turbo4", "t2va"): 5, ("turbo4", "fl2va"): 5}
    recipes = {}
    for (name, task), points in plans.items():
        # A well-formed grid (σ falling from 1 to 0, timesteps 1 − σ in float32), not the recipe's values.
        sigmas = [f32(1 - i / (points - 1)) for i in range(points)]
        arm = {"shift": 1.0, "timesteps": [f32(1.0 - s) for s in sigmas[:-1]], "sigmas": sigmas}
        recipes[f"{name}/{task}"] = {"video": arm, "audio": arm, "steps": [
            {"timesteps": [0.5], "temb": {"file": f"main:tables/{name}/{task}/step-{i}/temb.raw"},
             "blocks": f"main:tables/{name}/{task}/step-{i}/block-{{block:02d}}.raw"} for i in range(points - 1)]}
    return {
        "body": {"check": {"bindings": [f"b{i}" for i in range(22)]}},
        "body-weights": {"adapter": {"repo": ADAPTER_REPO, "revision": ADAPTER_REVISION, "sha256": ADAPTER_SHA256,
                                     "execution": "dynamicLora", "rank": LORA_RANK, "alpha": LORA_ALPHA}},
        "prepare": {"textInputs": [{"name": f"w{i}"} for i in range(20)]},
        "conditioner": {"inputs": [{"name": f"w{i}"} for i in range(11)],
                        "embedding": {"file": "main:weights/conditioner/embedding.bin", "rows": 151936, "width": 5120, "dtype": "bfloat16"},
                        "rope": {"theta": 5000000, "mropeSection": [24, 20, 20], "headDim": 128, "dtype": "bfloat16"}},
        "tables": {"recipes": recipes},
        "video-decoder": {"tile": VIDEO_TILE, "windowLatents": WINDOW_LATENTS, "windowFrames": WINDOW_FRAMES,
                          "windowStride": WINDOW_STRIDE, "precision": "bfloat16"},
        "video-encoder": {"tileSize": VIDEO_TILE, "seed": ENCODE_SEED, "tails": {key: {} for key in CANVASES}},
        "audio-decoder": {"fixed": {str(na): {} for na in AUDIO_LATENTS}, "sampleRate": SAMPLE_RATE,
                          "hop": AUDIO_HOP, "channels": AUDIO_CHANNELS},
    }


def sample_manifest() -> dict:
    return manifest(sample_meta(), {".aether/placeholder": {"bytes": 1, "sha256": "0" * 64}})


if __name__ == "__main__":
    if sys.argv[1:] != ["--sample-manifest"]:
        raise SystemExit("usage: profiles_h3.py --sample-manifest")
    print(json.dumps(sample_manifest(), indent=2, ensure_ascii=False))
