"""Temporal-window stitching oracle from the pinned Diffusers H3 VAE `_decode` control flow.

Runs the original `AutoencoderKLMiniMaxH3._decode` with `_decode_clip` replaced by a deterministic
stub (no weights, no model inference) and writes, for each decoder call, the 22 frames the exported
window decoder emits (`clip[3:20] ++ clip[23:28]`), plus the original stitched output.
Usage: python -I -B temporal_stitch_oracle.py OUTPUT_DIR
"""
import hashlib, json, math, os, sys, types
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3 as VAE

out = sys.argv[1]
os.makedirs(out, exist_ok=False)
H = W = 16; T = 12  # 2 windows -> 39 frames
vae = VAE.__new__(VAE)
torch.nn.Module.__init__(vae)
vae.temporal_compression_ratio = 4
vae.frame_pre_padding = (-17) % 4
vae.tokens_chunk_size = math.ceil(17 / 4)
vae.token_overlap = (-3) % vae.tokens_chunk_size
vae.frame_overlap = max(vae.token_overlap * 4 - vae.frame_pre_padding, 0)
object.__setattr__(vae, "_internal_dict", {"token_drop": 3, "clip_length": 17})
type(vae).config = property(lambda self: types.SimpleNamespace(**self._internal_dict))
calls = []
def stub(self, z):
    g = torch.Generator().manual_seed(1000 + len(calls))
    clip = torch.randn((1, 3, z.shape[2] * 4, H, W), generator=g, dtype=torch.float32)
    calls.append(clip)
    return clip
vae._decode_clip = types.MethodType(stub, vae)
z = torch.zeros((1, 24, T, H // 16, W // 16))
dec = vae._decode(z)
assert dec.shape == (1, 3, 39, H, W), dec.shape
def save(name, tensor):
    data = tensor.contiguous().numpy().astype("<f4").tobytes()
    open(os.path.join(out, name), "wb").write(data)
    return {"path": name, "shape": list(tensor.shape), "sha256": hashlib.sha256(data).hexdigest()}
windows = []
for i, clip in enumerate(calls):
    window = torch.cat([clip[:, :, 3:20], clip[:, :, 23:28]], dim=2)[0]  # [3,22,H,W]
    windows.append(save(f"window-{i}.f32", window))
# Frame-major [frames,3,H,W], as the Swift stitcher emits.
reference = save("reference.f32", dec[0].permute(1, 0, 2, 3))
json.dump({"schema": 1, "width": W, "height": H, "frames": 39, "windows": windows, "reference": reference,
           "scope": "pinned Diffusers 7ecb1e1b AutoencoderKLMiniMaxH3._decode with a deterministic stub _decode_clip; no weights",
           "torch": torch.__version__}, open(os.path.join(out, "temporal-oracle.json"), "w"), indent=1)
print(len(calls), "calls", reference["sha256"])
