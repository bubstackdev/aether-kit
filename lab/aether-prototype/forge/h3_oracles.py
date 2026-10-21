"""Save small independent CPU oracles by calling pinned Diffusers unchanged."""
from pathlib import Path
import argparse, hashlib, json, sys

ap=argparse.ArgumentParser()
ap.add_argument("--lab",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
args=ap.parse_args()
sys.dont_write_bytecode=True
sys.path[:0]=[str(args.lab/"forge"),str(args.lab/"forge/vendor/python"),str(args.lab/"forge/vendor/diffusers/src")]
import qualify_block as q
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
torch.manual_seed(8271)
video=torch.randn(32); audio=torch.randn(32)
report=dict(schema=1,diffusers_commit="7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941",video=video.tolist(),audio=audio.tolist(),steps=[])
sha=lambda t:hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
schedulers=[q.MiniMaxH3Scheduler(shift=s) for s in (12.,3.)]
for scheduler in schedulers:scheduler.set_timesteps(51)
for i in range(50):
    vv,av=torch.randn(32),torch.randn(32)
    video=schedulers[0].step(vv,schedulers[0].timesteps[i],video,return_dict=False)[0]
    audio=schedulers[1].step(av,schedulers[1].timesteps[i],audio,return_dict=False)[0]
    report["steps"].append(dict(video_velocity=vv.tolist(),audio_velocity=av.tolist(),video_sha256=sha(video),audio_sha256=sha(audio)))
latent=torch.arange(24*7*4*4,dtype=torch.float32).reshape(1,24,7,4,4)
report["patchify_sha256"]=sha(patchify_video_latents(latent,(1,2,2)))
from diffusers import AutoencoderKLMiniMaxH3
import types
with torch.device("meta"):vae=AutoencoderKLMiniMaxH3.from_config(q.SNAPSHOT/"vae/config.json")
def stub_clip(self,z):
    value=z.repeat_interleave(4,dim=2).repeat(1,3,1,2,2)
    offset=torch.arange(value.numel(),dtype=torch.float32).reshape(value.shape)*0.00813
    return value*1.713+offset
vae._decode_clip=types.MethodType(stub_clip,vae)
z=torch.randn(1,1,37,1,1)
reference=vae._decode(z)[0].permute(1,0,2,3).contiguous()
windows=[vae._decode(z[:,:,i*5:i*5+7]).flatten().tolist() for i in range(7)]
report["decode_stitching"]=dict(frames=124,width=2,height=2,windows=windows,reference_sha256=sha(reference))
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+"\n")
print(args.output)
