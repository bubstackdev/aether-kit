"""Aether's offline H3 recipe. Calls pinned, unmodified upstream modules.

Python is export/oracle tooling only; the generation runtime is Rust/Corsairs.
All derived files stay in --output. The lab and checkpoint are read-only.
"""
from pathlib import Path
import argparse, copy, gc, hashlib, json, os, sys, time
from h3_weights import CANONICAL_QKV_LAYOUT, canonical_body_parameters, convert_raw_transformer_key

SNAPSHOT = "42ed227ee7df40d41602854ae760620d6eb651fe"
COMMIT = "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while data := f.read(8 << 20): h.update(data)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", choices=["front", "modulations", "conditioner", "video", "audio", "check-video", "check-audio", "decoders", "finalize", "all"], default="all")
    ap.add_argument("--generated",type=Path,help="Clip directory with retained raw latents and decoder outputs")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--text-rows", type=int, default=49)
    ap.add_argument("--reuse",type=Path,help="Reuse geometry-independent modulation exports and a matching conditioner")
    args = ap.parse_args()
    lab, out = args.lab.resolve(), args.output.resolve()
    if out.is_relative_to(lab): raise ValueError("outputs must stay outside the read-only lab")
    out.mkdir(parents=True, exist_ok=True)
    environment = json.loads((lab / "records/environment.json").read_text())
    assert environment["unmodifiedDependencies"] and environment["diffusers"]["commit"] == COMMIT
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(lab / "forge"), str(lab / "forge/vendor/python"), str(lab / "forge/vendor/diffusers/src")]
    import qualify_block as q
    from export_functional import parameter_input_graph
    import torch
    import coreai_torch as ct
    from safetensors import safe_open
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    torch.set_num_threads(4)
    assert q.SNAPSHOT.name == SNAPSHOT
    if not (args.width > 0 and args.height > 0 and args.width % 32 == args.height % 32 == 0): raise ValueError("canvas must be positive multiples of 32")
    if args.frames < 5 or (args.frames - 5) % 17: raise ValueError("frame count must be 17*n+5")
    lf, lh, lw = (args.frames - 5) // 17 * 5 + 2, args.height // 16, args.width // 16
    na, nt = round(args.frames / 24 * 40) * 2, args.text_rows
    nv, rows = lf * (lh // 2) * (lw // 2), lf * (lh // 2) * (lw // 2) + na + nt
    meta_path = out / "build.json"
    identity = dict(snapshot=SNAPSHOT, diffusers_commit=COMMIT, width=args.width, height=args.height, frames=args.frames, text_rows=nt)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(schema=1, **identity)
    assert all(meta[k] == v for k, v in identity.items()), "output belongs to another profile"
    meta["recipe_sha256"] = digest(__file__)
    if args.reuse:
        previous=json.loads(args.reuse.read_text())
        assert previous["snapshot"]==SNAPSHOT and previous["diffusers_commit"]==COMMIT
        if "modulations" in previous:meta["modulations"]=previous["modulations"]
        if previous["text_rows"]==nt and "conditioner" in previous:meta["conditioner"]=previous["conditioner"]
    def update(**fields):
        meta.update(fields); save(meta_path, meta)
    def fixture(name, tensor):
        tensor = tensor.detach().contiguous().cpu()
        path = out / "fixtures" / (name + ".raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = tensor.view(torch.uint8).numpy().tobytes()
        path.write_bytes(data)
        return dict(name=name, path=str(path), dtype=str(tensor.dtype).removeprefix("torch."), shape=list(tensor.shape), sha256=hashlib.sha256(data).hexdigest())
    def export(name, module, example, names, outputs, dynamic_shapes=None):
        path = out / "models" / (name + ".aimodel")
        path.parent.mkdir(parents=True, exist_ok=True)
        print("export", name, flush=True)
        started = time.monotonic()
        with torch.inference_mode():
            ep = torch.export.export(module.eval(), example, dynamic_shapes=dynamic_shapes).run_decompositions(ct.get_decomp_table())
        cv = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        cv.add_exported_program(ep, input_names=names, output_names=outputs, state_names=[], entrypoint_name="main")
        program = cv.to_coreai(); program.optimize(); program.save_asset(path)
        source_sha256 = digest(path / "main.mlirb")
        assert (path / "main.hash").read_bytes().hex() == source_sha256
        print("exported", name, round(time.monotonic() - started, 2), flush=True)
        return dict(path=str(path), function="main", sha256=source_sha256,recipe_sha256=digest(__file__))
    def check_decode(kind,module,shape,filename):
        if not args.generated:raise ValueError("--generated is required")
        folder=args.generated.resolve()
        rows=torch.frombuffer(bytearray((folder/f"{kind}-latents.f32").read_bytes()),dtype=torch.float32).reshape(shape)
        with torch.inference_mode(): reference=module.eval()(rows).contiguous()
        if kind=="video" and json.loads((folder/"decoded.json").read_text())["video"].get("layout")=="FCHW":reference=reference[0].permute(1,0,2,3).contiguous()
        actual=torch.frombuffer(bytearray((folder/filename).read_bytes()),dtype=torch.float32).reshape(reference.shape)
        finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
        error=(actual.double()-reference.double())
        relative_l2=float(error.norm()/reference.double().norm().clamp_min(1e-30))
        result=dict(scope="full raw original CPU decoder comparison; no lossy media comparison",passed=finite and relative_l2<=0.02,
            predefined_relative_l2_limit=0.02,relative_l2=relative_l2,max_absolute_error=float(error.abs().max()),finite=finite,
            shape=list(reference.shape),actual_sha256=digest(folder/filename),reference_sha256=hashlib.sha256(reference.view(torch.uint8).numpy().tobytes()).hexdigest(),
            byte_identical=bool(torch.equal(reference,actual)),snapshot=SNAPSHOT,diffusers_commit=COMMIT)
        save(folder/f"check-{kind}.json",result)
        print(json.dumps(result),flush=True)
        assert result["passed"], result
    with torch.device("meta"): model = q.MiniMaxH3Transformer3DModel()
    cfg, converter = dict(model.config), q.converter()
    index = json.loads((q.SNAPSHOT / "FL2VA/transformer/model.safetensors.index.json").read_text())["weight_map"]
    def transformer_weights(predicate):
        weights = {}
        for name, shard in sorted(index.items()):
            if not predicate(name): continue
            with safe_open(q.SNAPSHOT / "FL2VA/transformer" / shard, framework="pt", device="cpu") as f: value = f.get_tensor(name)
            for key, tensor in convert_raw_transformer_key(converter, name, value, cfg): weights[key] = tensor
        return weights
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(torch.ones(nt, dtype=torch.int64), lf, lh, lw, na // 2, (1,2,2), 2, 2, 0)
    pos, tags, vi, ai, ti, _, _ = layout
    rope = type(model.rope)(rope_freq_dim=cfg["rope_freq_dim"], rope_theta=cfg["rope_theta"])
    with torch.no_grad(): cos, sin = rope(pos)
    if "rotary_cos" not in meta: update(rotary_cos=fixture("cos", cos), rotary_sin=fixture("sin", sin))
    stages = ["front", "modulations", "conditioner", "audio", "video", "finalize"] if args.stage == "all" else ["audio","video","finalize"] if args.stage=="decoders" else [args.stage]
    for stage in stages:
        if stage == "front":
            weights = transformer_weights(lambda n: not n.startswith("blocks.") or n.startswith("blocks.0."))
            model.load_state_dict(weights, strict=False, assign=True)
            class Prepare(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.video, self.audio = model.proj_in, model.audio_proj_in
                    self.text, self.refine = model.context_embedder, model.token_refiner
                def forward(self, video, audio, text):
                    text = self.refine(self.text(text))
                    # T2VA's index_copy destinations are disjoint contiguous
                    # [text|audio|video] ranges. Concatenation avoids the SDK's
                    # incorrect BF16 full/zeros dtype inference.
                    return torch.cat((text,self.audio(audio).to(text.dtype),self.video(video).to(text.dtype)),dim=1)
            class Time(torch.nn.Module):
                def __init__(self): super().__init__(); self.proj, self.embed = model.time_proj, model.time_embedder
                def forward(self, timestep): return self.embed(self.proj(timestep))
            class Heads(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.norm, self.video, self.audio = model.norm_out, model.proj_out, model.audio_proj_out
                def forward(self, hidden, temb, indices):
                    hidden = self.norm(hidden, temb, indices).float()
                    return self.video(hidden).index_select(1,vi), self.audio(hidden).index_select(1,ai)
            k = torch.export.Dim("timestep_count", min=1, max=3)
            prepare_inputs = (torch.zeros(1,nv,96), torch.zeros(1,na,32), torch.zeros(1,nt,5120,dtype=torch.bfloat16))
            prepare = export("prepare", Prepare(), prepare_inputs, ["video","audio","text"], ["hidden"])
            time_model = export("time", Time(), (torch.tensor([0.1,0.2]),), ["timestep"], ["temb"], ({0:k},))
            hidden = torch.zeros(1,rows,5376,dtype=torch.bfloat16)
            temb = torch.zeros(2,2688)
            indices = (tags * 0).to(torch.int64)
            heads = export("heads", Heads(), (hidden,temb,indices), ["hidden","temb","indices"], ["video","audio"], (None,{0:k},None))
            cached = q.BlockCall(model.transformer_blocks[0]).eval()
            cached.block.adaln_proj = q.CachedModulation()
            bodies=[]
            # The mechanical parameter-lifting FX graph retains its table-row
            # specialization. Preserve each actual cardinality explicitly.
            for cardinality in (1,2):
                table = torch.zeros(6,3*cardinality,5376,dtype=torch.bfloat16)
                inputs = (hidden,table,tags,cos,sin)
                graph, params = parameter_input_graph(cached, inputs)
                bodies.append(export(f"body-{cardinality}", graph, (*inputs,*params.values()), ["hidden","temb","tags","cos","sin"]+[f"w{i}" for i in range(len(params))], ["hidden_out"]))
            parameter_record = json.loads((lab / "records/functional-backbone-parameters.json").read_text())
            assert parameter_record["snapshot"] == SNAPSHOT and parameter_record["completed"]
            template = json.loads((lab / "records/functional-export-19350.json").read_text())["parameters"]
            assert list(params) == [p["parameter"] for p in template]
            blocks=[]
            for block in parameter_record["blocks"]:
                assert block["block"] == len(blocks)
                blocks.append([dict(p, path=str(lab/p["file"])) for p in block["parameters"]])
            assert len(blocks)==50
            blocks=canonical_body_parameters(blocks,q.SNAPSHOT,converter,cfg,out)
            # Independent first-input oracle: real saved features, original model calls.
            case=json.loads((lab/"records/conditioner-prefix.json").read_text())["cases"][0]
            if nt==case["tokenCount"]:
                text=torch.frombuffer(bytearray((lab/"forge/fixtures/conditioner/case-0.f32").read_bytes()),dtype=torch.float32).reshape(1,nt,5120).to(torch.bfloat16)
                torch.manual_seed(1043)
                latent=torch.randn(1,24,lf,lh,lw)
                from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
                video=patchify_video_latents(latent,(1,2,2))[None]
                audio=torch.randn(1,na,32)
                model.rope=rope
                captured={}
                class Stop(Exception):pass
                def before(_module,inputs):captured["hidden"]=inputs[0].detach();raise Stop()
                hook=model.transformer_blocks[0].register_forward_pre_hook(before)
                with torch.no_grad():
                    try:model(video,audio,text,torch.zeros(1),tags*0,tags,pos,vi,ai,ti,return_dict=False)
                    except Stop:pass
                    expected=captured["hidden"]
                    assert torch.equal(expected,Prepare()(video,audio,text))
                hook.remove()
                oracle=dict(text=fixture("oracle/text",text),video=fixture("oracle/video",video),audio=fixture("oracle/audio",audio),hidden=fixture("oracle/hidden",expected),prompt=case["prompt"])
                update(oracle=oracle)
            schedule = out/"schedule.json"
            schedule.write_bytes((lab/"records/scheduler-50-forward.json").read_bytes())
            update(prepare=prepare,time=time_model,heads=heads,bodies=bodies,parameters=blocks,schedule=str(schedule),tokenizer=str(q.SNAPSHOT/"FL2VA/text_encoder/tokenizer.json"),transformer_weight_layout=CANONICAL_QKV_LAYOUT,qkv_recipe_sha256=digest(Path(__file__).with_name("h3_weights.py")))
            del weights,params,graph,cached; gc.collect()
        elif stage == "modulations":
            result=[]
            # One model acquired at a time at runtime; all actual schedule pairs
            # are evaluated before the body parameters become resident.
            for i in range(50):
                with torch.device("meta"): block=q.MiniMaxH3Transformer3DModel().transformer_blocks[i]
                weights=transformer_weights(lambda n: n.startswith(f"blocks.{i}.adaln_proj."))
                prefix=f"transformer_blocks.{i}.adaln_proj."
                block.adaln_proj.load_state_dict({n.removeprefix(prefix):v for n,v in weights.items()},strict=True,assign=True)
                result.append(export(f"modulation-{i}",q.ModulationCall(block.adaln_proj),(torch.zeros(2,2688),),["temb"],["modulations"],({0:torch.export.Dim("timestep_count",min=1,max=3)},)))
                update(modulations=result)
                del block,weights; gc.collect()
        elif stage == "conditioner":
            from transformers import Qwen3VLConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextDecoderLayer
            directory=q.SNAPSHOT/"FL2VA/text_encoder"
            text_index=json.loads((directory/"model.safetensors.index.json").read_text())["weight_map"]
            text_config=Qwen3VLConfig.from_dict(json.loads((directory/"config.json").read_text())).text_config
            text_config._attn_implementation="sdpa"
            def load_text(module,prefix):
                weights={}
                for name,shard in text_index.items():
                    if name.startswith(prefix):
                        with safe_open(directory/shard,framework="pt",device="cpu") as f: weights[name.removeprefix(prefix)]=f.get_tensor(name)
                module.load_state_dict(weights,strict=True,assign=True)
            with torch.device("meta"): text_model=Qwen3VLTextModel(text_config)
            text_model.rotary_emb=type(text_model.rotary_emb)(config=text_config)
            load_text(text_model.embed_tokens,"model.language_model.embed_tokens.")
            class Embed(torch.nn.Module):
                def __init__(self): super().__init__();self.embed,self.rope=text_model.embed_tokens,text_model.rotary_emb
                def forward(self,ids):
                    hidden=self.embed(ids)
                    cos,sin=self.rope(hidden,torch.arange(nt,dtype=torch.int64)[None])
                    return hidden,cos,sin
            embedding=export("conditioner-embedding",Embed(),(torch.ones(1,nt,dtype=torch.int64),),["ids"],["hidden","cos","sin"])
            del text_model; gc.collect()
            class Layer(torch.nn.Module):
                def __init__(self,layer):super().__init__();self.layer=layer
                def forward(self,hidden,cos,sin): return self.layer(hidden,(cos,sin),attention_mask=None,use_cache=False)
            layers=[]
            for i in range(50):
                with torch.device("meta"): layer=Qwen3VLTextDecoderLayer(text_config,i)
                load_text(layer,f"model.language_model.layers.{i}.")
                inputs=(torch.zeros(1,nt,5120,dtype=torch.bfloat16),torch.zeros(1,nt,128,dtype=torch.bfloat16),torch.zeros(1,nt,128,dtype=torch.bfloat16))
                layers.append(export(f"conditioner-{i}",Layer(layer),inputs,["hidden","cos","sin"],["hidden_out"]))
                update(conditioner=dict(embedding=embedding,layers=layers))
                del layer; gc.collect()
        elif stage in ("audio","check-audio"):
            with torch.device("meta"): vae=AutoencoderKLMiniMaxH3Audio.from_config(q.SNAPSHOT/"audio_vae/config.json")
            with safe_open(q.SNAPSHOT/"FL2VA/audio_vae/model.safetensors",framework="pt",device="cpu") as f: weights={k:f.get_tensor(k) for k in f.keys()}
            vae.load_state_dict(weights,strict=True,assign=True);vae.eval()
            class Audio(torch.nn.Module):
                def __init__(self):super().__init__();self.vae=vae
                def forward(self,rows):
                    z=rows.reshape(2,na//2,32).permute(0,2,1)
                    z=z*torch.tensor(vae.config.latents_std).view(1,-1,1)+torch.tensor(vae.config.latents_mean).view(1,-1,1)
                    return self.vae.decode(z,return_dict=False)[0][:,0,:]
            if stage=="check-audio":check_decode("audio",Audio(),(na,32),"audio.f32")
            else:update(audio_decoder=export("audio-decoder",Audio(),(torch.zeros(na,32),),["rows"],["pcm"]))
            del vae,weights;gc.collect()
        elif stage in ("video","check-video"):
            with torch.device("meta"): vae=AutoencoderKLMiniMaxH3.from_config(q.SNAPSHOT/"vae/config.json")
            source=q.SNAPSHOT/"FL2VA/video_vae/source/model.safetensors"
            weights={}
            with safe_open(source,framework="pt",device="cpu") as f:
                for name in f.keys():
                    for key,value in converter.convert_video_vae_key(name,f.get_tensor(name),dict(vae.config)): weights[key]=value
            vae.load_state_dict(weights,strict=True,assign=True);vae.eval();vae.enable_tiling()
            # This derived, nonpersistent buffer is not in the checkpoint. Its
            # unmodified constructor must run on CPU after meta initialization.
            vae.decoder.rope=type(vae.decoder.rope)(dim=int(vae.config.decoder_rope_dim_ratio*vae.config.decoder_attention_head_dim),theta=vae.config.decoder_rope_theta)
            class Video(torch.nn.Module):
                def __init__(self):super().__init__();self.vae=vae
                def forward(self,rows):
                    window_lf=lf if stage=="check-video" else 7
                    z=rows.reshape(1,window_lf,lh//2,lw//2,24,1,2,2).permute(0,4,1,5,2,6,3,7).reshape(1,24,window_lf,lh,lw)
                    z=z*torch.tensor(vae.config.latents_std).view(1,-1,1,1,1)+torch.tensor(vae.config.latents_mean).view(1,-1,1,1,1)
                    return self.vae.decode(z,return_dict=False)[0]
            if stage=="check-video":check_decode("video",Video(),(nv,96),"video.f32")
            else:update(video_decoder=export("video-decoder",Video(),(torch.zeros(7*(lh//2)*(lw//2),96),),["rows"],["pixels"]),video_window_latents=7,video_window_frames=22,video_window_stride=5)
            del vae,weights;gc.collect()
        elif stage == "finalize":
            required=["prepare","time","heads","bodies","parameters","modulations","rotary_cos","rotary_sin","video_decoder","audio_decoder","tokenizer","schedule"]
            missing=[name for name in required if name not in meta]
            assert not missing, missing
            assert len(meta["modulations"])==len(meta["parameters"])==50
            assert meta.get("transformer_weight_layout")==CANONICAL_QKV_LAYOUT, "front must be regenerated with canonical raw QKV conversion"
            if "conditioner" in meta: assert len(meta["conditioner"]["layers"])==50
            save(out/"bundle.json",dict(meta,complete=True))
            print("bundle",out/"bundle.json",flush=True)


if __name__ == "__main__": main()
