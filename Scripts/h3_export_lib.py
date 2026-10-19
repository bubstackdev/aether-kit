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
