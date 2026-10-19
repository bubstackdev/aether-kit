"""Lab-free helpers for exporting MiniMax H3 profiles with pinned Diffusers 7ecb1e1b (CPU only).

Everything here reads the original FL2VA checkpoint and the Turbo adapter read-only. Conversion uses the
pinned Diffusers converter (`convert_minimax_h3_to_diffusers.py`, loaded from a supplied directory) so the
per-head QKV normalisation is the upstream one, not a transcription. No Core AI runtime or GPU is used:
graphs are traced with torch.export and converted with coreai-torch, which writes assets on the CPU.
"""
import hashlib, importlib.util, json
from pathlib import Path
import torch
from safetensors import safe_open

SNAPSHOT = "42ed227ee7df40d41602854ae760620d6eb651fe"
COMMIT = "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941"
CANONICAL_QKV_LAYOUT = "raw_per_head_qkv_to_contiguous_qkv_v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_converter(directory: Path):
    spec = importlib.util.spec_from_file_location("h3_converter", Path(directory) / "convert_minimax_h3_to_diffusers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Checkpoint:
    """Streams converted transformer weights from the raw FL2VA shards."""

    def __init__(self, snapshot: Path, converter):
        self.directory = Path(snapshot) / "FL2VA/transformer"
        self.config = json.loads((self.directory / "config.json").read_text())
        self.index = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.converter = converter

    def converted(self, predicate) -> dict:
        weights = {}
        for name in sorted(n for n in self.index if predicate(n)):
            with safe_open(self.directory / self.index[name], framework="pt", device="cpu") as f:
                value = f.get_tensor(name)
            if name.endswith(".attn.qkv_proj.weight"):
                value = self.converter.reorder_interleaved_qkv(
                    value, self.config["num_attention_heads"], self.config["attention_head_dim"])
            for key, tensor in self.converter.convert_transformer_key(name, value, self.config):
                weights[key] = tensor
        return weights


def meta_model():
    from diffusers import MiniMaxH3Transformer3DModel
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel()
    return model


class CachedModulation(torch.nn.Module):
    """Stands in for a block's `adaln_proj`: the six modulation tensors come precomputed as one `[6, 3k, D]` table."""

    def forward(self, table):
        return table.unbind(0)


class BlockCall(torch.nn.Module):
    """One transformer block with its modulation table and rotary tables as explicit inputs."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, table, tags, cos, sin):
        return self.block(hidden, table, tags, (cos, sin))


class Lifted(torch.nn.Module):
    """`module` with every parameter lifted to a positional input, in `named_parameters()` order (w0, w1, ...)."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.names = [name for name, _ in module.named_parameters()]

    def forward(self, *args):
        count = len(args) - len(self.names)
        inputs, weights = args[:count], args[count:]
        return torch.func.functional_call(self.module, dict(zip(self.names, weights)), inputs)


def lift(module):
    """Returns the lifted graph module and its parameters (name -> tensor), in input order."""
    lifted = Lifted(module).eval()
    return lifted, dict(module.named_parameters())


def export_asset(path: Path, module, example, input_names, output_names, dynamic_shapes=None) -> dict:
    """Traces `module` and writes a Core AI asset on the CPU. Never loads or specialises the asset."""
    import coreai_torch as ct
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        program = torch.export.export(module.eval(), example, dynamic_shapes=dynamic_shapes).run_decompositions(ct.get_decomp_table())
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(program, input_names=input_names, output_names=output_names, state_names=[], entrypoint_name="main")
    asset = converter.to_coreai()
    asset.optimize()
    asset.save_asset(path)
    checksum = sha256_file(path / "main.mlirb")
    assert (path / "main.hash").read_bytes().hex() == checksum, "Core AI asset hash file differs from its stream"
    return dict(path=str(path), function="main", sha256=checksum)


def raw_fixture(path: Path, name: str, tensor) -> dict:
    tensor = tensor.detach().contiguous().cpu()
    data = tensor.view(torch.uint8).numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return dict(name=name, path=str(path), dtype=str(tensor.dtype).removeprefix("torch."), shape=list(tensor.shape), sha256=sha256_bytes(data))


ADAPTER_REPOSITORY = "lightx2v/Minimax-h3-Turbo"
ADAPTER_REVISION = "3ec17a324ced54151364f24f8b5fb6bf7e26414f"
ADAPTER_FILE = "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
ADAPTER_SHA256 = "e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f"


class Spans:
    """Byte ranges of tensors inside safetensors files, so weights can be referenced in place (never copied).

    `span(key)` -> (resolved file, absolute offset, length, dtype, shape). Paths are resolved through the HF cache
    symlinks to the content-addressed blobs, which stay immutable while the snapshot exists.
    """

    def __init__(self, files):
        import struct
        self.entries = {}
        for file in files:
            file = Path(file).resolve()
            with open(file, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                start, end = value["data_offsets"]
                self.entries[key] = (file, 8 + size + start, end - start, value["dtype"], value["shape"])

    @classmethod
    def sharded(cls, directory: Path):
        index = json.loads((Path(directory) / "model.safetensors.index.json").read_text())["weight_map"]
        return cls(sorted({Path(directory) / name for name in index.values()}))

    def span(self, key):
        return self.entries[key]

    def segments(self, key, rows=None):
        """Segments covering `key`, or only the listed `(first_row, row_count)` runs of its leading axis, in order."""
        file, offset, length, dtype, shape = self.entries[key]
        if rows is None:
            return [dict(path=str(file), offset=offset, length=length)]
        stride = length // shape[0]
        return [dict(path=str(file), offset=offset + first * stride, length=count * stride) for first, count in rows]


def read_segments(segments) -> bytes:
    parts = []
    for s in segments:
        with open(s["path"], "rb") as f:
            f.seek(s["offset"]); data = f.read(s["length"])
        assert len(data) == s["length"], s
        parts.append(data)
    return b"".join(parts)


def tensor_bytes(tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def transformer_source_segments(spans: Spans, name: str, heads: int = 56, head_dim: int = 128):
    """In-place segments for one converted (Diffusers) transformer / token-refiner weight, mirroring the pinned
    converter: per-head interleaved QKV rows -> contiguous to_q/to_k/to_v, the SwiGLU fc1 halves swapped, renames."""
    original = name
    for new, old in ((".refiner_blocks.", ".blocks."), ("transformer_blocks.", "blocks."), (".attn.norm_q.", ".attn.q_norm."),
                     (".attn.norm_k.", ".attn.k_norm."), (".attn.to_out.0.", ".attn.out_proj."), (".ff.net.2.", ".mlp.fc2."),
                     ("context_embedder.", "condition_proj.")):
        original = original.replace(new, old, 1)
    for part, which in ((".attn.to_q.weight", 0), (".attn.to_k.weight", 1), (".attn.to_v.weight", 2)):
        if original.endswith(part):
            key = original.removesuffix(part) + ".attn.qkv_proj.weight"
            return spans.segments(key, [(h * 3 * head_dim + which * head_dim, head_dim) for h in range(heads)])
    if original.endswith(".ff.net.0.proj.weight"):
        key = original.removesuffix(".ff.net.0.proj.weight") + ".mlp.fc1.weight"
        half = spans.span(key)[4][0] // 2
        return spans.segments(key, [(half, half), (0, half)])   # Diffusers SwiGLU wants [value; gate]
    return spans.segments(original)


def bound_tensor(name, segments, reference, expected_dtype=None) -> dict:
    """A manifest tensor entry for `segments`, after proving its bytes equal `reference` (the converter/PEFT tensor)."""
    data = read_segments(segments)
    assert data == tensor_bytes(reference), f"in-place bytes differ from the converted tensor for {name}"
    dtype = str(reference.dtype).removeprefix("torch.")
    assert expected_dtype is None or dtype == expected_dtype, (name, dtype)
    return dict(name=name, segments=segments, dtype=dtype, shape=list(reference.shape), sha256=sha256_bytes(data))


def adapter_path():
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(ADAPTER_REPOSITORY, ADAPTER_FILE, revision=ADAPTER_REVISION, local_files_only=True)).resolve()
    assert sha256_file(path) == ADAPTER_SHA256, "Turbo adapter hash differs; refusing"
    return path


def load_turbo(model, adapter: Path, predicate):
    """Loads the Turbo LoRA for the targets `predicate` accepts, exactly as the AetherKit oracles do."""
    weights, alphas = {}, {}
    with safe_open(adapter, framework="pt", device="cpu") as source:
        for key in source.keys():
            if predicate(key):
                weights["transformer." + key.replace(".default.weight", ".weight")] = source.get_tensor(key)
                if ".lora_A." in key:
                    alphas["transformer." + key.split(".lora_A.")[0] + ".alpha"] = 8.0
    model.load_lora_adapter(weights, prefix="transformer", adapter_name="turbo", network_alphas=alphas)


def lora_source_key(binding_prefix: str, name: str) -> str:
    """`block.attn.to_q.lora_A.turbo.weight` under `transformer_blocks.3` -> the adapter file's PEFT key."""
    return f"{binding_prefix}.{name.split('.', 1)[1]}".replace(".turbo.weight", ".default.weight")
