"""CPU references for a schema-3 H3 profile run, from the original modules (pinned Diffusers 7ecb1e1b, CPU only, lab-free).

Reads the inputs `aether-probe h3-profile-dump` wrote (token ids, layout, seeded latents, a probe hidden state) and
writes raw tensors in the same manifest format, so `--part compare` (or a human) can set them beside a native run.

  text     The conditioner tap and the refined text. The tap runs the ORIGINAL Qwen3-VL model forward
           (`text_encoder.model`, as the Diffusers encoder step calls it) with decoder layers streamed in place and the
           forward stopped after layer 49 (`hidden_states[50]`); then context_embedder + token_refiner + Turbo LoRA.
  graphs   Each native graph on the CPU, on the same inputs the native `--graphs` run used: pack, blocks 0 and 49 at real
           step-0 contexts (the native packed sequence, and the native block-48 output from the run's chain), with live
           AdaLN from the live time embedding, i.e. independent of the precomputed tables; heads on the native block-49
           output; one tile window
           (`vae._decode`, untiled) and the audio decoder. Also checks the dump's layout and row timesteps against the
           pipeline's own `build_packed_sequence` / `build_row_timesteps` and scheduler.
  render   A complete T2VA trajectory: the original transformer forward with its 50 blocks streamed (each built with
           its Turbo LoRA exactly as PEFT loads it), both schedulers' `step`, then the pipeline's video decode
           (`vae.decode`, ImageNet revert, clamp, RGB8 round-half-even) and audio decode. Uses this oracle's own tap.
  compare  Cosine / relative L2 / max |diff| for every tensor both directories hold; PSNR for frames.
  presentation  Upstream's FL2VA conditioner presentation from the request alone (no dump): the pipeline's resize step
           and `MiniMaxH3FL2VATextEncoderStep` run verbatim on --prompt, --keyframe and/or --last-keyframe at --canvas,
           with the real Qwen3-VL processor, the vision tower loaded whole and the decoder layers streamed. Records the
           ids, tags, Qwen modality types, pixels, M-RoPE positions, merger and DeepStack rows, the decoder input, early
           layer outputs and the tap. --layers N < 50 is a cost slice (no tap).
  trajectory  References at a native trajectory's captured contexts (`h3-profile-run --render --capture`): per step,
           packing, blocks 0 and 49 on their captured inputs, heads on the captured block-49 output and both scheduler
           updates, compared in-process on full tensors (no reference tensors written); float32 blocks as a diagnostic.
Usage (Forge's H3 oracle environment, `uv sync --frozen --project Forge/h3-oracle`):
       Forge/h3-oracle/.venv/bin/python -I -B Scripts/prepare-h3-profile-oracle.py --part PART --dump DUMP --output NEW_DIR [--run RUN] [--text TEXT_DIR]
       Forge/h3-oracle/.venv/bin/python -I -B Scripts/prepare-h3-profile-oracle.py --part presentation --prompt FILE --keyframe PNG --canvas WxH --output NEW_DIR
"""
import argparse, json, math, os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import h3_export_lib as h

SNAPSHOT_ROOT = Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / h.SNAPSHOT
# The working folder that holds the pinned converter (`oracle-scripts/vendor/convert_minimax_h3_to_diffusers.py`) and the
# export (`exports/h3-fl2va`): H3_STAGING, or the folder that contains this checkout.
STAGING = Path(os.environ.get("H3_STAGING", Path(__file__).resolve().parents[2]))
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int32": torch.int32, "uint8": torch.uint8}


class Raw:
    """The probe's raw-tensor directory: manifest.json {tensors: {name: {file, dtype, shape, sha256}}, fields}."""

    def __init__(self, path: Path, new=False):
        self.path = Path(path)
        if new:
            self.path.mkdir(parents=True, exist_ok=False)
            self.manifest = dict(tensors={}, fields={}, report={})
        else:
            self.manifest = json.loads((self.path / "manifest.json").read_text())

    def has(self, name):
        return name in self.manifest["tensors"]

    def get(self, name):
        entry = self.manifest["tensors"][name]
        data = (self.path / entry["file"]).read_bytes()
        assert h.sha256_bytes(data) == entry["sha256"], f"{name} changed on disk"
        return torch.frombuffer(bytearray(data), dtype=DTYPES[entry["dtype"]]).view(entry["shape"]).clone()

    def field(self, name):
        return self.manifest["fields"][name]

    def put(self, name, tensor):
        entry = h.raw_fixture(self.path / f"{name}.raw", name, tensor)
        self.manifest["tensors"][name] = dict(file=f"{name}.raw", dtype=entry["dtype"], shape=entry["shape"], sha256=entry["sha256"])
        self.save()

    def save(self):
        (self.path / "manifest.json").write_text(json.dumps(self.manifest, indent=1, sort_keys=True))


def converter():
    return h.load_converter(STAGING / "oracle-scripts/vendor")


def denoiser_model(checkpoint, adapter):
    """The transformer on meta with every non-block part real (Turbo LoRA on the token refiner, as PEFT loads it)."""
    model = h.meta_model()
    model.load_state_dict(checkpoint.converted(lambda n: not n.startswith("blocks.")), strict=False, assign=True)
    h.load_turbo(model, adapter, lambda k: k.startswith("token_refiner."))
    # Buffers built under the meta device carry no values; rebuild the two modules that own one.
    model.rope = type(model.rope)(rope_freq_dim=model.config.rope_freq_dim, rope_theta=model.config.rope_theta)
    model.time_proj = type(model.time_proj)(num_channels=model.config.freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    return model.eval()


def real_block(checkpoint, adapter, b):
    """Block `b` with its base weights and its Turbo LoRA, built exactly as the export's parameter stage builds it."""
    model = h.meta_model()
    model.load_state_dict(checkpoint.converted(lambda n: n.startswith(f"blocks.{b}.")), strict=False, assign=True)
    h.load_turbo(model, adapter, lambda k: k.startswith(f"transformer_blocks.{b}."))
    return model.transformer_blocks[b].eval()


class StreamedBlock(torch.nn.Module):
    def __init__(self, checkpoint, adapter, index, onblock):
        super().__init__()
        self.checkpoint, self.adapter, self.index, self.onblock = checkpoint, adapter, index, onblock

    def forward(self, *args):
        out = real_block(self.checkpoint, self.adapter, self.index)(*args)
        self.onblock(self.index)
        return out


def compare(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    diff = a - b
    return dict(cosine=float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300)), relative_l2=float(diff.norm() / b.norm().clamp_min(1e-300)),
                max_abs=float(diff.abs().max()), reference_max_abs=float(b.abs().max()), finite=bool(torch.isfinite(a).all()))


def text_part(dump, out, prompt):
    from transformers import AutoTokenizer, Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextRotaryEmbedding
    ids = dump.get("ids").tolist()
    report = out.manifest["report"]
    if prompt is not None:
        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_ROOT / "FL2VA/tokenizer")
        reference = tokenizer(prompt.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
        report["token_ids_equal_transformers"] = reference == ids
        assert reference == ids, "Swift token ids differ from the pipeline's tokenizer"
    directory = SNAPSHOT_ROOT / "FL2VA/text_encoder"
    config = Qwen3VLConfig.from_pretrained(directory)
    config.text_config._attn_implementation = "sdpa"
    spans = h.Spans.sharded(directory)
    from safetensors import safe_open

    def tensors(prefix):
        result = {}
        for name in sorted(k for k in spans.entries if k.startswith(prefix)):
            with safe_open(spans.span(name)[0], framework="pt", device="cpu") as f:
                result[name.removeprefix(prefix)] = f.get_tensor(name)
        return result
    with torch.device("meta"):
        model = Qwen3VLModel(config)
    language = model.language_model
    language.embed_tokens.load_state_dict(tensors("model.language_model.embed_tokens."), strict=True, assign=True)
    language.rotary_emb = Qwen3VLTextRotaryEmbedding(config.text_config)
    captured = {}

    class Stop(Exception):
        pass

    def pre(layer, args, kwargs, index):
        layer.load_state_dict(tensors(f"model.language_model.layers.{index}."), strict=True, assign=True)

    def post(layer, args, kwargs, output, index):
        hidden = output[0] if isinstance(output, tuple) else output
        layer.to("meta")
        if index == 49:
            captured["tap"] = hidden.detach().clone()
            raise Stop

    for index, layer in enumerate(language.layers[:50]):
        layer.register_forward_pre_hook(lambda m, a, k, i=index: pre(m, a, k, i), with_kwargs=True)
        layer.register_forward_hook(lambda m, a, k, o, i=index: post(m, a, k, o, i), with_kwargs=True)
    # `processor.create_mm_token_type_ids` (whose video processor needs torchvision): 0 for text, 1 image, 2 video.
    special = {config.image_token_id: 1, config.video_token_id: 2}
    input_ids = torch.tensor([ids], dtype=torch.long)
    types = torch.tensor([[special.get(i, 0) for i in ids]], dtype=torch.long)
    report["vision_tokens"] = int((types != 0).sum())
    started = time.monotonic()
    with torch.inference_mode():
        try:
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), mm_token_type_ids=types, use_cache=False, output_hidden_states=True)
        except Stop:
            pass
    tap = captured["tap"].to(torch.bfloat16)
    report["tap_seconds"] = round(time.monotonic() - started, 1)
    out.put("tap", tap)
    checkpoint = h.Checkpoint(SNAPSHOT_ROOT, converter())
    model = denoiser_model(checkpoint, h.adapter_path())
    with torch.inference_mode():
        refined = model.token_refiner(model.context_embedder(tap.to(model.context_embedder.weight.dtype)))
    out.put("refined", refined)
    out.save()


def presentation_part(out, prompt, first, last, canvas, layers=50, substitute_vision=None):
    """Upstream's FL2VA presentation from the request alone: the pipeline's own resize step and FL2VA text encoder step,
    run verbatim (real `Qwen3VLProcessor`, FL2VA tokenizer, a `Qwen3VLForConditionalGeneration` whose vision tower is
    loaded whole and whose decoder layers stream in place). Nothing comes from a Kit dump. Hooks record what the step
    builds: ids, Qwen modality types, pixels, M-RoPE positions, the merger and DeepStack outputs, the decoder input,
    early layer outputs and the tap. `layers` < 50 runs only that many decoder layers (a cost slice); the rest pass
    their input through, so the step's `hidden_states[50]` is then not the tap and is not recorded as one.
    `substitute_vision` (diagnostic only, one keyframe) replaces the tower's merger and DeepStack outputs with another
    run's (`image_embeds`, `deepstack`, cast to bf16), so the rest of the step runs upstream's code on those rows."""
    from types import SimpleNamespace
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding, Qwen3VLVisionRotaryEmbedding)
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.modular_pipelines import PipelineState
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3ResizeStep
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3FL2VATextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG
    report, fields = out.manifest["report"], out.manifest["fields"]
    width, height = canvas
    text = prompt.read_text(encoding="utf-8")
    fields.update(part="presentation", prompt_sha256=h.sha256_file(prompt), canvas=f"{width}x{height}", layers=layers,
                  first=first and dict(name=first.name, sha256=h.sha256_file(first)),
                  last=last and dict(name=last.name, sha256=h.sha256_file(last)))
    first_image, last_image = (Image.open(p).convert("RGB") if p else None for p in (first, last))
    images = [i for i in (first_image, last_image) if i is not None]
    assert images, "a presentation needs a first or a last keyframe"
    for i, image in enumerate(images):
        out.put(f"keyframe{i}_source", torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(image.size[1], image.size[0], 3))

    # 1. The pipeline's resize step puts the keyframes on the canvas (a canvas-sized keyframe passes through).
    resize = MiniMaxH3ResizeStep()
    components = SimpleNamespace(image_processor=VaeImageProcessor(vae_scale_factor=16), canvas_multiple=32,
                                 config=SimpleNamespace(canvas_short_edge=768, canvas_max_pixels=768 * 1344))
    state = PipelineState()
    state.set("image", first_image)
    state.set("last_image", last_image)
    state.set("height", height)
    state.set("width", width)
    resize(components, state)
    placed = state.get("keyframes")
    report["keyframes_passed_through"] = [a.size == (width, height) for a in images]
    report["keyframe_anchors"] = list(state.get("keyframe_anchors"))
    for i, image in enumerate(placed):
        out.put(f"keyframe{i}", torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(height, width, 3))

    # 2. The conditioner: vision tower and token embedding loaded, decoder layers streamed in place.
    directory = SNAPSHOT_ROOT / "FL2VA/text_encoder"
    config = Qwen3VLConfig.from_pretrained(directory)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    spans = h.Spans.sharded(directory)

    def tensors(prefix):
        result = {}
        for name in sorted(k for k in spans.entries if k.startswith(prefix)):
            with safe_open(spans.span(name)[0], framework="pt", device="cpu") as f:
                result[name.removeprefix(prefix)] = f.get_tensor(name)
        return result
    with torch.device("meta"):
        encoder = Qwen3VLForConditionalGeneration(config)
    visual, language = encoder.model.visual, encoder.model.language_model
    visual.load_state_dict(tensors("model.visual."), strict=True, assign=True)
    visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.vision_config)
    language.embed_tokens.load_state_dict(tensors("model.language_model.embed_tokens."), strict=True, assign=True)
    language.norm.load_state_dict(tensors("model.language_model.norm."), strict=True, assign=True)
    language.rotary_emb = Qwen3VLTextRotaryEmbedding(config.text_config)
    assert not any(t.is_meta for t in [*visual.parameters(), *visual.buffers()]), "vision tower left on meta"

    class PassThrough(Qwen3VLTextDecoderLayer):
        """A decoder layer that returns its input, still recorded as a decoder-layer output."""
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.gradient_checkpointing = False

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states
    recorded = {}
    keep = {0, 1, 2, 3, layers - 1}

    def pre(layer, args, kwargs, index):
        layer.load_state_dict(tensors(f"model.language_model.layers.{index}."), strict=True, assign=True)

    def post(layer, args, kwargs, output, index):
        hidden = output[0] if isinstance(output, tuple) else output
        layer.to("meta")
        if index in keep:
            recorded[f"layer{index:02d}"] = hidden.detach().clone()
    for index in range(len(language.layers)):
        if index < layers:
            language.layers[index].register_forward_pre_hook(lambda m, a, k, i=index: pre(m, a, k, i), with_kwargs=True)
            language.layers[index].register_forward_hook(lambda m, a, k, o, i=index: post(m, a, k, o, i), with_kwargs=True)
        else:
            language.layers[index] = PassThrough()

    def model_inputs(module, args, kwargs):
        recorded["input_ids"] = kwargs["input_ids"].detach().clone()
        recorded["mm_token_type_ids"] = kwargs["mm_token_type_ids"].detach().clone()
        if kwargs.get("pixel_values") is not None:
            recorded["pixel_values"] = kwargs["pixel_values"].detach().clone()
            recorded["image_grid_thw"] = kwargs["image_grid_thw"].detach().clone()

    def vision_outputs(module, args, kwargs, output):
        if substitute_vision is not None:
            embeds, deepstack = (substitute_vision.get(n).to(torch.bfloat16) for n in ("image_embeds", "deepstack"))
            assert embeds.shape == output.pooler_output.shape and deepstack.shape[1:] == output.deepstack_features[0].shape
            output.pooler_output, output.deepstack_features = embeds, list(deepstack.unbind(0))
        recorded["image_embeds"] = output.pooler_output.detach().clone()
        recorded["deepstack"] = torch.stack([f.detach() for f in output.deepstack_features]).clone()
        return output

    def language_inputs(module, args, kwargs):
        recorded["position_ids"] = kwargs["position_ids"].detach().clone()
        recorded["inputs_embeds"] = kwargs["inputs_embeds"].detach().clone()
    encoder.model.register_forward_pre_hook(model_inputs, with_kwargs=True)
    visual.register_forward_hook(vision_outputs, with_kwargs=True)
    language.register_forward_pre_hook(language_inputs, with_kwargs=True)

    # 3. The FL2VA text encoder step, verbatim.
    step = MiniMaxH3FL2VATextEncoderStep()
    components = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT_ROOT / "FL2VA/tokenizer"),
                                 processor=Qwen3VLProcessor.from_pretrained(SNAPSHOT_ROOT / "FL2VA/processor"),
                                 text_encoder=encoder, text_encoder_layer=50, text_tag=MINIMAX_H3_TEXT_TAG,
                                 video_tag=MINIMAX_H3_VIDEO_TAG, _execution_device=torch.device("cpu"))
    state = PipelineState()
    state.set("prompt", text)
    state.set("keyframes", placed)
    started = time.monotonic()
    with torch.inference_mode():
        step(components, state)
    report["step_seconds"] = round(time.monotonic() - started, 1)

    # 4. Everything the step built, as raw tensors.
    ids = recorded["input_ids"][0]
    out.put("token_ids", ids.to(torch.int32))
    out.put("text_token_tags", state.get("text_token_tags").to(torch.int32))
    out.put("mm_token_type_ids", recorded["mm_token_type_ids"][0].to(torch.int32))
    out.put("position_ids", recorded["position_ids"][:, 0].to(torch.int32))    # [3, T]: time, height, width
    if "pixel_values" in recorded:
        out.put("pixel_values", recorded["pixel_values"])                         # bf16, as the step feeds the tower
        out.put("pixel_values_f32", components.processor.image_processor(images=placed, return_tensors="pt")["pixel_values"])
        out.put("image_grid_thw", recorded["image_grid_thw"].to(torch.int32))
        out.put("image_embeds", recorded["image_embeds"])
        out.put("deepstack", recorded["deepstack"])
    out.put("inputs_embeds", recorded["inputs_embeds"])
    for name in sorted(k for k in recorded if k.startswith("layer")):
        out.put(name, recorded[name])
    if layers == 50:
        tap = state.get("prompt_embeds")
        assert torch.equal(tap, recorded["layer49"]), "the step's hidden_states[50] is not layer 49's output"
        out.put("tap", tap)
    report.update(tokens=int(ids.numel()), image_tokens=int((ids == config.image_token_id).sum()),
                  max_position=int(recorded["position_ids"].max()))
    out.save()


def layout(dump, model_config, text_tags=None):
    """The pipeline's own layout and row-timestep plan for the dump's geometry, checked against the dump. `text_tags`
    are the conditioning rows' tags (upstream's FL2VA presentation tags its vision blocks 0); text-only by default."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames, video_latent_num_frames
    from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as mp
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    width, height, frames = int(dump.field("width")), int(dump.field("height")), int(dump.field("frames"))
    tokens = int(dump.field("tokens"))
    # The dump's keyframe anchors in packed order: "first", "last" or "first,last" (absent for T2VA).
    anchors = tuple(a for a in dump.manifest["fields"].get("anchors", "").split(",") if a)
    assert anchors in ((), ("first",), ("last",), ("first", "last")), anchors
    position_ids, token_tags, video_indices, audio_indices, text_indices, ncv, nca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.full((tokens,), 1, dtype=torch.long) if text_tags is None else text_tags.long(), video_latent_num_frames(frames, 17, 5), height // 16, width // 16,
        audio_latent_num_frames(frames), (1, 2, 2), mp.MINIMAX_H3_AUDIO_CHANNELS, mp.MINIMAX_H3_AUDIO_TAG, mp.MINIMAX_H3_VIDEO_TAG, anchors)
    assert nca == 0 and (ncv > 0) == bool(anchors)
    checks = dict(positions=bool(torch.equal(position_ids.float(), dump.get("positions"))), token_tags=bool(torch.equal(token_tags.int(), dump.get("token_tags"))),
                  video_indices=bool(torch.equal(video_indices.int(), dump.get("video_indices"))),
                  audio_indices=bool(torch.equal(audio_indices.int(), dump.get("audio_indices"))),
                  text_indices=bool(torch.equal(text_indices.int(), dump.get("text_indices"))))
    recipe = dump.field("recipe")
    points = dict(turbo8=9, turbo4=5)[recipe]
    schedulers = []
    for shift in (12.0, 3.0):
        s = MiniMaxH3Scheduler(shift=shift); s.set_timesteps(points); schedulers.append(s)
    plan = []
    for i, (tv, ta) in enumerate(zip(schedulers[0].timesteps, schedulers[1].timesteps)):
        unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(video_indices, audio_indices, ncv, 0, tokens, float(tv), float(ta), max(float(tv), 0.999), 1.0)
        checks[f"step{i}"] = bool(torch.equal(unique.float(), dump.get(f"step{i}_timesteps")) and torch.equal(inverse.int(), dump.get(f"step{i}_indices"))
                                  and torch.equal((inverse * 3 + token_tags).int(), dump.get(f"step{i}_adaln")))
        plan.append((unique, inverse))
    assert len(plan) == int(dump.field("steps"))
    return dict(ncv=ncv, position_ids=position_ids, token_tags=token_tags, video_indices=video_indices, audio_indices=audio_indices,
                text_indices=text_indices, plan=plan, schedulers=schedulers, frames=frames, width=width, height=height), checks


KEYFRAMES = ("keyframe", "last_keyframe")   # the dump's first and last frames, in packed order


def condition_latents(dump, name="keyframe"):
    """The pipeline's keyframe conditioning for one of the dump's keyframes (`encode_vae_condition`, float32 VAE, default
    tiling). The first and the last frame use the same single-frame encode."""
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    vae = AutoencoderKLMiniMaxH3.from_pretrained(STAGING / "exports/h3-fl2va/converted/vae", torch_dtype=torch.float32).eval()
    pixels = dump.get(name).permute(2, 0, 1)[None, :, None]
    with torch.inference_mode():
        return encode_vae_condition(vae, pixels, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 42)


def condition_rows(dump, scheduler):
    """The pipeline's FL2VA condition rows (`MiniMaxH3PrepareConditionLatentsStep`): each keyframe's latents noised to
    t = 0.999 by its own draw (`condition_noise[i]`, in packed order) and patchified, first then last."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    names = [n for n in KEYFRAMES if dump.has(n)]
    noise = dump.get("condition_noise")
    assert noise.shape[0] == len(names), (noise.shape, names)
    conditions = [condition_latents(dump, n) for n in names]
    rows = [patchify_video_latents(scheduler.scale_noise(c, 0.999, noise[i:i + 1]), (1, 2, 2)) for i, c in enumerate(conditions)]
    return torch.cat(conditions), torch.cat(rows)


def graphs_part(dump, out, run, text, blocks=(0, 49)):
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    report = out.manifest["report"]
    keyframe = next((n for n in KEYFRAMES if dump.has(n)), None)
    if keyframe:   # FL2VA dumps qualify the keyframe encoder only, on their first keyframe (the probe's --graphs does the same)
        t = time.monotonic()
        out.put("condition", condition_latents(dump, keyframe))
        report["condition_seconds"] = round(time.monotonic() - t, 1)
        out.save()
        return
    checkpoint = h.Checkpoint(SNAPSHOT_ROOT, converter())
    adapter = h.adapter_path()
    model = denoiser_model(checkpoint, adapter)
    shape, checks = layout(dump, model.config)
    report["layout"] = checks
    assert all(checks.values()), checks
    refined = text.get("refined")
    video, audio = dump.get("latents_video"), dump.get("latents_audio")
    started = time.monotonic()
    with torch.inference_mode():
        # The original forward's packing (index_copy into zeros), not the export's gather.
        hidden = refined.new_zeros((1, len(shape["position_ids"]), refined.shape[-1]))
        hidden = hidden.index_copy(1, shape["text_indices"], refined)
        hidden = hidden.index_copy(1, shape["video_indices"], model.proj_in(video.to(model.proj_in.weight.dtype)).to(refined.dtype))
        hidden = hidden.index_copy(1, shape["audio_indices"], model.audio_proj_in(audio.to(model.audio_proj_in.weight.dtype)).to(refined.dtype))
        out.put("packed", hidden)
        unique, inverse = shape["plan"][0]
        temb = model.time_embedder(model.time_proj(unique).to(torch.float32))
        rotary = model.rope(shape["position_ids"])
        adaln = inverse * 3 + shape["token_tags"]
        # Real step-0 contexts, as the native --graphs run evaluated them: block 0 reads the native packed sequence and
        # block b > 0 reads the native output of block b - 1 (`contextNN`); heads read the native block-49 output.
        # A run from before these contexts existed falls back to the dump's synthetic probe hidden state.
        probe = dump.get("probe_hidden")
        report["block_inputs"] = {}
        for b in blocks:
            name = "packed" if b == 0 else f"context{b:02d}"
            source = run.get(name) if run.has(name) else probe
            report["block_inputs"][f"block{b:02d}"] = name if run.has(name) else "probe_hidden"
            t = time.monotonic()
            out.put(f"block{b:02d}", real_block(checkpoint, adapter, b)(source, temb, adaln, rotary))
            report[f"block{b:02d}_seconds"] = round(time.monotonic() - t, 1)
        head_input = run.get("block49") if run.has("block49") and run.has("context49") else probe
        report["heads_input"] = "native block49 output" if head_input is not probe else "probe_hidden"
        states = model.norm_out(head_input, temb, inverse).to(model.proj_out.weight.dtype)
        out.put("heads_video", model.proj_out(states).index_select(1, shape["video_indices"]))
        out.put("heads_audio", model.audio_proj_out(states).index_select(1, shape["audio_indices"]))
        del model
        vae = AutoencoderKLMiniMaxH3.from_pretrained(STAGING / "exports/h3-fl2va/converted/vae", torch_dtype=torch.float32).eval()
        vae.use_tiling = False
        rows = run.get("tile_rows")
        z = rows.view(7, 8, 8, 24, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(1, 24, 7, 16, 16)
        z = z * torch.tensor(vae.config.latents_std).view(1, 24, 1, 1, 1) + torch.tensor(vae.config.latents_mean).view(1, 24, 1, 1, 1)
        out.put("tile", vae._decode(z).contiguous())   # the original single-window control flow keeps the 22 frames itself
        del vae
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(STAGING / "exports/h3-fl2va/converted/audio-vae", torch_dtype=torch.float32).eval()
        na = audio.shape[1] // 2
        rows = audio[0].reshape(2, na, 32).permute(0, 2, 1)
        rows = rows * torch.tensor(audio_vae.config.latents_std).view(1, -1, 1) + torch.tensor(audio_vae.config.latents_mean).view(1, -1, 1)
        out.put("pcm", audio_vae.decode(rows, return_dict=False)[0].float().squeeze(1).contiguous())
    report["graphs_seconds"] = round(time.monotonic() - started, 1)
    out.save()


def render_part(dump, out, text, tap_source=None):
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    report = out.manifest["report"]
    checkpoint = h.Checkpoint(SNAPSHOT_ROOT, converter())
    adapter = h.adapter_path()
    model = denoiser_model(checkpoint, adapter)
    shape, checks = layout(dump, model.config)
    report["layout"] = checks
    assert all(checks.values()), checks
    progress = dict(started=time.monotonic())
    model.transformer_blocks = torch.nn.ModuleList(
        [StreamedBlock(checkpoint, adapter, b, lambda b: b == 49 and print(f"step done {time.monotonic() - progress['started']:.0f} s", flush=True)) for b in range(50)])
    tap = (tap_source or text).get("tap")
    report["tap_from"] = str((tap_source or text).path)
    video, audio = dump.get("latents_video")[0].clone(), dump.get("latents_audio")[0].clone()
    video_scheduler, audio_scheduler = shape["schedulers"]
    ncv = shape["ncv"]
    if ncv:
        # The pipeline's FL2VA conditioning: noise each keyframe's latents to t = 0.999 with its own draw, pack, lead.
        condition, rows = condition_rows(dump, video_scheduler)
        out.put("condition", condition)
        assert rows.shape[0] == ncv
        video = torch.cat([rows, video])
    kwargs = dict(token_tags=shape["token_tags"], position_ids=shape["position_ids"], video_indices=shape["video_indices"],
                  audio_indices=shape["audio_indices"], text_indices=shape["text_indices"])
    with torch.inference_mode():
        for i, t in enumerate(video_scheduler.timesteps):
            unique, inverse = shape["plan"][i]
            v, a = model(hidden_states=video[None], audio_hidden_states=audio[None], encoder_hidden_states=tap, timestep=unique,
                         timestep_indices=inverse, return_dict=False, **kwargs)
            video[ncv:] = video_scheduler.step(v[0, ncv:].float(), t, video[ncv:], return_dict=False)[0]
            audio = audio_scheduler.step(a[0].float(), audio_scheduler.timesteps[i], audio, return_dict=False)[0]
            out.put(f"step{i}_video", video[None].contiguous()); out.put(f"step{i}_audio", audio[None].contiguous())
        report["denoise_seconds"] = round(time.monotonic() - progress["started"], 1)
        out.put("final_video", video[None].contiguous()); out.put("final_audio", audio[None].contiguous())
        del model
        lf, lh, lw = video_latent_frames(shape), shape["height"] // 16, shape["width"] // 16
        rows = video[ncv:].reshape(1, lf, lh // 2, lw // 2, 24, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 24, lf, lh, lw)
        vae = AutoencoderKLMiniMaxH3.from_pretrained(STAGING / "exports/h3-fl2va/converted/vae", torch_dtype=torch.float32).eval()
        latents = rows * torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1) + torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        t = time.monotonic()
        clip = vae.decode(latents, return_dict=False)[0]
        report["video_decode_seconds"] = round(time.monotonic() - t, 1)
        pixels = (clip.float() * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)).clamp(0, 1)
        out.put("frames_float", pixels[0].permute(1, 2, 3, 0).contiguous())
        out.put("frames", (pixels[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).contiguous())
        del vae
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(STAGING / "exports/h3-fl2va/converted/audio-vae", torch_dtype=torch.float32).eval()
        na = audio.shape[0] // 2
        rows = audio.reshape(2, na, 32).permute(0, 2, 1)
        rows = rows * torch.tensor(audio_vae.config.latents_std).view(1, -1, 1) + torch.tensor(audio_vae.config.latents_mean).view(1, -1, 1)
        out.put("pcm", audio_vae.decode(rows, return_dict=False)[0].float().squeeze(1).contiguous())
    out.save()


def trajectory_part(dump, out, run, steps, float32_diagnostic=False, presentation=None):
    """CPU references at a native trajectory's captured contexts (`h3-profile-run --render --capture`), per step:
    the original packing of the captured input latents and refined text; blocks 0 and 49 on their captured inputs
    (packed / native block-48 output) with live AdaLN from the live time embedding; both heads on the captured block-49
    output; both schedulers' `step` on the captured heads and input latents. Original precision (bf16 blocks), full
    tensors. With `float32_diagnostic`, a separate pass instead runs each block upcast to float32 (diagnostic only).
    A dump presenting keyframes to the conditioner needs `presentation`, upstream's own (`--part presentation`, all 50
    layers): its ids and tags must equal the dump's, the layout is built from its tags, and the captured refined text
    is compared with the original refiner on its tap."""
    report = out.manifest["report"]
    capture = Raw(Path(run.path) / "capture")
    hashes = capture.manifest["report"]["hashes"]
    checkpoint = h.Checkpoint(SNAPSHOT_ROOT, converter())
    adapter = h.adapter_path()
    model = denoiser_model(checkpoint, adapter)
    tags = None
    if dump.has("presentation_tags"):
        assert presentation is not None, "this dump presents keyframes to the conditioner: pass upstream's --presentation"
        tags = presentation.get("text_token_tags")
        exact = dict(ids=bool(torch.equal(presentation.get("token_ids").reshape(-1).int(), dump.get("presentation_ids").int())),
                     tags=bool(torch.equal(tags.reshape(-1).int(), dump.get("presentation_tags").int())),
                     positions=bool(torch.equal(presentation.get("position_ids").reshape(3, -1).int(), dump.get("presentation_positions").int())))
        # The render's own conditioning ids and tags (h3-profile-run records what the session's conditioner read).
        exact.update(render_ids=run.has("render_ids") and bool(torch.equal(run.get("render_ids").int(), presentation.get("token_ids").reshape(-1).int())),
                     render_tags=run.has("render_tags") and bool(torch.equal(run.get("render_tags").int(), tags.reshape(-1).int())))
        report["presentation"] = dict(exact, oracle=str(presentation.path), tokens=int(tags.numel()))
        assert all(exact.values()), exact
        tags = tags.reshape(-1)
    else:
        assert presentation is None, "this dump conditions on the prompt alone; a presentation does not apply"
    shape, checks = layout(dump, model.config, tags)
    report["layout"] = checks
    assert all(checks.values()), checks
    rotary = model.rope(shape["position_ids"])
    refined = capture.get("refined")
    assert h.sha256_bytes(refined.contiguous().view(torch.uint8).numpy().tobytes()) == hashes["refined"]
    # One block resident at a time, rebuilt per use (bounded resident memory). The float32 diagnostic is a separate
    # pass (`float32_diagnostic=True`): it upcasts the same module in place and records only native-vs-float32.
    rotary32 = tuple(r.float() for r in rotary)
    comparisons, diagnostics = report.setdefault("comparisons", {}), report.setdefault("float32_diagnostics", {})
    sha = lambda t: h.sha256_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
    if presentation is not None and not float32_diagnostic:
        # The conditioning end to end: Kit's refined text (native presentation, vision tower and conditioner) against
        # the original refiner on upstream's own tap.
        with torch.inference_mode():
            reference = model.token_refiner(model.context_embedder(presentation.get("tap").to(model.context_embedder.weight.dtype)))
        assert reference.shape == refined.shape and reference.dtype == refined.dtype, (reference.shape, refined.shape)
        comparisons["refined"] = dict(compare(refined.float(), reference.float()), byte_identical=bool(torch.equal(refined, reference)),
                                      reference_sha256=sha(reference), native_sha256=hashes["refined"])
        del reference
        out.save()

    def check(name, reference, native_name):
        native = capture.get(native_name)
        assert sha(native) == hashes[native_name], f"{native_name} differs from the capture hash"
        assert native.shape == reference.shape and native.dtype == reference.dtype, (native_name, native.shape, reference.shape, native.dtype, reference.dtype)
        row = compare(native.float(), reference.float())
        row.update(byte_identical=bool(torch.equal(native, reference)), reference_sha256=sha(reference), native_sha256=hashes[native_name])
        comparisons[name] = row
        return native

    if shape["ncv"] and 0 in steps and not float32_diagnostic:
        # The step-0 input's condition rows against the pipeline's own: each keyframe through the original VAE encode,
        # noised by the dump's draws and packed in anchor order.
        name = "step0_latents_in_video"
        native = capture.get(name)
        assert sha(native) == hashes[name], f"{name} differs from the capture hash"
        _, reference = condition_rows(dump, shape["schedulers"][0])
        row = compare(native[0, :shape["ncv"]].float(), reference.float())
        row.update(anchors=dump.manifest["fields"]["anchors"], rows=int(shape["ncv"]))
        comparisons["step0/conditions"] = row
        out.save()
    with torch.inference_mode():
        for s in steps:
            started = time.monotonic()
            unique, inverse = shape["plan"][s]
            temb = model.time_embedder(model.time_proj(unique).to(torch.float32))
            adaln = inverse * 3 + shape["token_tags"]
            if float32_diagnostic:
                for b, context in ((0, f"step{s}_packed"), (49, f"step{s}_context49")):
                    x, native = capture.get(context), capture.get(f"step{s}_block{b:02d}")
                    assert sha(x) == hashes[context if context != f"step{s}_context49" else f"step{s}_block48"]
                    assert sha(native) == hashes[f"step{s}_block{b:02d}"]
                    block = real_block(checkpoint, adapter, b).float()
                    t = time.monotonic()
                    ref32 = block(x.float(), temb.float(), adaln, rotary32)
                    del block
                    diagnostics[f"step{s}/block{b:02d}"] = dict(
                        native_vs_float32=float((native.float() - ref32).double().norm() / ref32.double().norm()), seconds=round(time.monotonic() - t, 1))
                    del x, native, ref32
                out.save()
                print(f"step {s} float32 diagnostic done", flush=True)
                continue
            video_in, audio_in = capture.get(f"step{s}_latents_in_video"), capture.get(f"step{s}_latents_in_audio")
            for n, t in ((f"step{s}_latents_in_video", video_in), (f"step{s}_latents_in_audio", audio_in)):
                assert sha(t) == hashes[n], f"{n} differs from the capture hash"
            hidden = refined.new_zeros((1, len(shape["position_ids"]), refined.shape[-1]))
            hidden = hidden.index_copy(1, shape["text_indices"], refined)
            hidden = hidden.index_copy(1, shape["video_indices"], model.proj_in(video_in.to(model.proj_in.weight.dtype)).to(refined.dtype))
            hidden = hidden.index_copy(1, shape["audio_indices"], model.audio_proj_in(audio_in.to(model.audio_proj_in.weight.dtype)).to(refined.dtype))
            check(f"step{s}/packed", hidden, f"step{s}_packed")
            del hidden
            for b, context in ((0, f"step{s}_packed"), (49, f"step{s}_context49")):
                x = capture.get(context)
                assert sha(x) == hashes[context if context != f"step{s}_context49" else f"step{s}_block48"], f"{context} differs from the capture hash"
                block = real_block(checkpoint, adapter, b)
                t = time.monotonic()
                reference = block(x, temb, adaln, rotary)
                report[f"step{s}_block{b:02d}_seconds"] = round(time.monotonic() - t, 1)
                del block
                check(f"step{s}/block{b:02d}", reference, f"step{s}_block{b:02d}")
                del x, reference
            final = capture.get(f"step{s}_block49")
            states = model.norm_out(final, temb, inverse).to(model.proj_out.weight.dtype)
            heads_video = check(f"step{s}/heads_video", model.proj_out(states).index_select(1, shape["video_indices"]), f"step{s}_heads_video")
            heads_audio = check(f"step{s}/heads_audio", model.audio_proj_out(states).index_select(1, shape["audio_indices"]), f"step{s}_heads_audio")
            del states, final
            schedulers = []
            for shift in (12.0, 3.0):
                scheduler = type(shape["schedulers"][0])(shift=shift)
                scheduler.set_timesteps(dict(turbo8=9, turbo4=5)[dump.field("recipe")])
                schedulers.append(scheduler)
            ncv = shape["ncv"]
            video = video_in[0].clone()
            video[ncv:] = schedulers[0].step(heads_video[0, ncv:].float(), schedulers[0].timesteps[s], video[ncv:], return_dict=False)[0]
            audio = schedulers[1].step(heads_audio[0].float(), schedulers[1].timesteps[s], audio_in[0].clone(), return_dict=False)[0]
            assert schedulers[0].step_index == s + 1 and schedulers[1].step_index == s + 1
            check(f"step{s}/latents_out_video", video[None].contiguous(), f"step{s}_latents_out_video")
            check(f"step{s}/latents_out_audio", audio[None].contiguous(), f"step{s}_latents_out_audio")
            report[f"step{s}_seconds"] = round(time.monotonic() - started, 1)
            out.save()
            print(f"step {s} references done {report[f'step{s}_seconds']} s", flush=True)
    out.save()


def video_latent_frames(shape):
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
    return video_latent_num_frames(shape["frames"], 17, 5)


def compare_part(reference, run, out):
    rows = {}
    for name in sorted(reference.manifest["tensors"]):
        if not run.has(name):
            continue
        a, b = run.get(name), reference.get(name)
        if a.shape != b.shape:
            rows[name] = dict(error=f"shape {list(a.shape)} vs {list(b.shape)}"); continue
        rows[name] = dict(compare(a.float(), b.float()), exact=bool(torch.equal(a, b)))
        if name == "frames":
            mse = float(((a.double() - b.double()) ** 2).mean())
            rows[name].update(psnr_db=round(10 * math.log10(255 ** 2 / mse), 2) if mse else math.inf,
                              mean_abs=float((a.double() - b.double()).abs().mean()), exact_fraction=float((a == b).double().mean()))
    out.manifest["report"]["compare"] = rows
    out.manifest["fields"].update(reference=str(reference.path), run=str(run.path))
    out.save()
    print(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", required=True, choices=["text", "graphs", "render", "compare", "trajectory", "presentation"])
    ap.add_argument("--keyframe", type=Path, help="presentation: the first keyframe image")
    ap.add_argument("--last-keyframe", type=Path, help="presentation: the last keyframe image (alone, or after --keyframe)")
    ap.add_argument("--canvas", help="presentation: WxH of the generated video")
    ap.add_argument("--layers", type=int, default=50, help="presentation: decoder layers to run (50 for the tap; fewer is a cost slice)")
    ap.add_argument("--substitute-vision", type=Path, help="presentation, diagnostic only: take the tower's outputs from this run")
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--text", type=Path, help="a --part text output (graphs, render) or the reference (compare)")
    ap.add_argument("--prompt", type=Path)
    ap.add_argument("--tap", type=Path, help="render: take the conditioner tap from this directory instead (e.g. a native run)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--blocks", default="0,49", help="graphs: blocks to check ('' for none; CPU attention at 960x544 takes hours)")
    ap.add_argument("--steps", default="all", help="trajectory: captured steps to reference ('all' or S,S,...)")
    ap.add_argument("--float32-diagnostic", action="store_true", help="trajectory: instead of references, the float32 block diagnostic only")
    ap.add_argument("--presentation", type=Path, help="trajectory: upstream's presentation (a --part presentation output) for a dump that presents keyframes")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    out = Raw(args.output, new=True)
    if args.part == "compare":
        return compare_part(Raw(args.text), Raw(args.run), out)
    if args.part == "presentation":
        width, height = (int(v) for v in args.canvas.split("x"))
        out.manifest["fields"] = dict(oracle_sha256=h.sha256_file(Path(__file__)), diffusers_commit=h.COMMIT, snapshot=h.SNAPSHOT,
                                      torch=torch.__version__)
        if args.substitute_vision:
            out.manifest["fields"].update(substitute_vision=str(args.substitute_vision.resolve()), diagnostic=True)
        presentation_part(out, args.prompt, args.keyframe, args.last_keyframe, (width, height), args.layers,
                          Raw(args.substitute_vision) if args.substitute_vision else None)
        print(out.path)
        return
    dump = Raw(args.dump)
    out.manifest["fields"] = dict(dump=str(args.dump.resolve()), part=args.part, profile_sha256=dump.field("profile_sha256"),
                                  oracle_sha256=h.sha256_file(Path(__file__)), diffusers_commit=h.COMMIT, snapshot=h.SNAPSHOT, torch=torch.__version__)
    out.save()
    if args.part == "text":
        text_part(dump, out, args.prompt)
    elif args.part == "graphs":
        graphs_part(dump, out, Raw(args.run), Raw(args.text), [int(b) for b in args.blocks.split(",") if b])
    elif args.part == "trajectory":
        steps = list(range(int(dump.field("steps")))) if args.steps == "all" else [int(s) for s in args.steps.split(",")]
        out.manifest["fields"].update(run=str(args.run.resolve()), steps=",".join(map(str, steps)))
        if args.presentation:
            out.manifest["fields"].update(presentation=str(args.presentation.resolve()))
        trajectory_part(dump, out, Raw(args.run), steps, float32_diagnostic=args.float32_diagnostic,
                        presentation=Raw(args.presentation) if args.presentation else None)
    else:
        render_part(dump, out, Raw(args.text), Raw(args.tap) if args.tap else None)
    print(out.path)


if __name__ == "__main__":
    main()
