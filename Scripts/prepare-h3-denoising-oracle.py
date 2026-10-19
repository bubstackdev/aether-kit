#!/usr/bin/env python3
"""Original-CPU references at captured native eight-step contexts (CPU only).

Full time/modulation/prepare/head/scheduler checks and sampled blocks 0/49.
This is not an independently computed CPU trajectory. No GPU API, export,
checkpoint mutation, or model-weight copy; one CPU backbone block at a time.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['lab', 'prototype', 'bundle', 'phases', 'cpu-record', 'capture', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    lab, prototype, out = args.lab.resolve(), args.prototype.resolve(), args.output.resolve()
    if out.is_relative_to(lab) or out.is_relative_to(prototype):
        raise ValueError('source projects remain read-only')
    out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(prototype/'forge'), str(lab/'forge'), str(lab/'forge/vendor/python'), str(lab/'forge/vendor/diffusers/src')]
    import qualify_block as q
    import torch
    from safetensors import safe_open
    from h3_weights import CANONICAL_QKV_LAYOUT, convert_raw_transformer_key
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep
    import diffusers.models.transformers.transformer_minimax_h3 as implementation
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    bundle = json.loads(args.bundle.read_text())
    phases = json.loads(args.phases.read_text())
    cpu = json.loads(args.cpu_record.read_text())
    capture_path = args.capture/'report.json'
    capture = json.loads(capture_path.read_text())
    assert capture['captureComplete'] and not capture['passed'] and capture['mode'] == '--capture'
    assert capture['bundleSHA256'] == sha(args.bundle) == cpu['bundle_sha256']
    assert capture['phaseSHA256'] == sha(args.phases) and capture['cpuRecordSHA256'] == sha(args.cpu_record)
    assert bundle['complete'] and bundle['recipe'] == 'turbo8' and bundle['transformer_weight_layout'] == CANONICAL_QKV_LAYOUT
    assert bundle['snapshot'] == q.SNAPSHOT.name == '42ed227ee7df40d41602854ae760620d6eb651fe'
    assert bundle['diffusers_commit'] == '7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941'
    assert (bundle['width'],bundle['height'],bundle['frames'],bundle['text_rows']) == (960,544,124,49)
    adapter_path = lab/'forge/vendor/weights/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors'
    assert sha(adapter_path) == bundle['adapter']['sha256'] == 'e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f'
    assert bundle['adapter']['execution'] == 'dynamic_lora' and bundle['adapter']['rank'] == 128 and bundle['adapter']['alpha'] == 8
    assert cpu['passed'] and not cpu['gpu_work']
    assert sha(prototype/'forge/h3_diagnose.py') == cpu['source_recipe_sha256']
    records = {x['key']: x for x in capture['outputs']}
    assert len(records) == len(capture['outputs']) == 864
    expected = {k:x['sha256'] for k,x in records.items()}
    manifest = dict(schema=1, bundle_sha256=sha(args.bundle), phase_sha256=sha(args.phases),
        cpu_record_sha256=sha(args.cpu_record), chain_sha256=capture['chainSHA256'],
        modulation_model_sha256=capture['modulationModelSHA256'], capture_report_path=str(capture_path.resolve()),
        capture_report_sha256=sha(capture_path), oracle_recipe_sha256=sha(__file__), expected=expected, references=[])
    report = dict(schema=1,passed=False,gpu_work=False,scope=__doc__,source_pins={},references=0,selected_query_validation=[],stages=[],native_prechecks=[])
    for path in [Path(__file__).resolve(),args.bundle,args.phases,args.cpu_record,capture_path,adapter_path,
                 prototype/'forge/h3_weights.py',prototype/'forge/h3_diagnose.py',lab/'forge/qualify_block.py',
                 lab/'forge/vendor/diffusers/scripts/convert_minimax_h3_to_diffusers.py',Path(implementation.__file__)]:
        report['source_pins'][str(path.resolve())] = sha(path)
    save(out/'cpu-checks.json',report)

    def raw(value):
        assert value.device.type == 'cpu'
        return value.detach().contiguous().view(torch.uint8).numpy().tobytes()

    def read(item):
        data = Path(item['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        value = torch.frombuffer(bytearray(data),dtype=getattr(torch,item['dtype'])).reshape(item['shape'])
        assert torch.isfinite(value).all() and value.device.type == 'cpu'
        return value

    def key(stage, step, block=-1, name='hidden'):
        return f'{stage}.{step}.{block}.{name}'

    def native(stage,step,block=-1,name='hidden'):
        return read(records[key(stage,step,block,name)])

    schedule = json.loads(Path(bundle['schedule']).read_text())
    roots = {x['name']:x for x in next(c for c in phases['cases'] if c['name'] == 'body-0')['inputs']}
    text_item = next(x for x in next(c for c in phases['cases'] if c['name'] == 'prepare')['inputs'] if x['name'] == 'text')
    text = read(text_item)
    _,tags,vi,ai,_,_,_ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(torch.ones(49,dtype=torch.int64),37,34,60,207,(1,2,2),2,2,0)
    plans=[];sources={'conditioning':text_item['sha256'],'rotary.cos':roots['cos']['sha256'],'rotary.sin':roots['sin']['sha256']}
    for step in range(8):
        vt,at = (arm['timesteps'][step] for arm in schedule['schedules'])
        values,indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(vi,ai,0,0,49,vt,at,0.999,1.0)
        plans.append((values,indices))
        for name,value in [('timestep',values),('indices',indices.to(torch.int32)),('tags',(indices*3+tags).to(torch.int32))]:
            sources[f'{name}.{step}']=hashlib.sha256(raw(value)).hexdigest()

    def context(k):
        stage,s,b,_ = k.split('.');s,b=int(s),int(b)
        if stage=='time':keys=[f'timestep.{s}']
        elif stage=='modulation':keys=[key('time',s,name='temb')]
        elif stage=='prepare':keys=[key('input',s,name='video'),key('input',s,name='audio'),'conditioning']
        elif stage=='body':keys=[key('prepare' if b==0 else 'body',s,-1 if b==0 else b-1),key('time',s,name='temb'),key('modulation',s,b,'modulations'),f'tags.{s}','rotary.cos','rotary.sin']
        elif stage=='heads':keys=[key('body',s,49),key('time',s,name='temb'),f'indices.{s}']
        elif stage=='latents':keys=[key('input',s,name='video'),key('input',s,name='audio'),key('heads',s,name='video'),key('heads',s,name='audio')]
        else:raise ValueError(stage)
        return {k: expected[k] if k in expected else sources[k] for k in keys}

    def reference(k,value,rows=None):
        value=value.detach().contiguous().cpu();assert torch.isfinite(value).all()
        data=raw(value);path=out/(k+'.raw');path.write_bytes(data)
        tensor=dict(path=str(path),dtype=str(value.dtype).split('.')[-1],shape=list(value.shape),sha256=hashlib.sha256(data).hexdigest())
        if rows is not None:tensor['row_indices']=rows
        manifest['references'].append(dict(key=k,context=context(k),tensor=tensor))
        report['references']+=1
        if records[k].get('path'):
            captured=read(records[k])
            if rows is not None:captured=captured.index_select(1,torch.tensor(rows,dtype=torch.int64))
            assert captured.shape == value.shape and captured.dtype == value.dtype
            difference=(captured.double()-value.double()).norm().item();norm=value.double().norm().item()
            relative=difference/norm if norm>0 else (0. if difference==0 else float('inf'))
            report['native_prechecks'].append(dict(key=k,relative_l2=relative,passed=relative<=0.02))
            save(out/'cpu-checks.json',report)
            assert relative<=0.02, (k,relative)
        save(out/'cpu-checks.json',report)
        return tensor

    with torch.device('meta'):
        model=q.MiniMaxH3Transformer3DModel()
    config=dict(model.config);converter=q.converter()
    index_path=q.SNAPSHOT/'FL2VA/transformer/model.safetensors.index.json'
    report['source_pins'][str(index_path)]=sha(index_path)
    index=json.loads(index_path.read_text())['weight_map']

    def weights_for(predicate):
        weights={}
        for name,shard in sorted(index.items()):
            if predicate(name):
                with safe_open(q.SNAPSHOT/'FL2VA/transformer'/shard,framework='pt',device='cpu') as source:
                    weights.update(convert_raw_transformer_key(converter,name,source.get_tensor(name),config))
        return weights

    print('CPU load outer modules; native trajectory remains unqualified',flush=True)
    model.load_state_dict(weights_for(lambda name:not name.startswith('blocks.')),strict=False,assign=True)
    adapter={};alphas={}
    with safe_open(adapter_path,framework='pt',device='cpu') as source:
        for name in source.keys():
            if name.startswith('token_refiner.'):
                adapter['transformer.'+name.replace('.default.weight','.weight')]=source.get_tensor(name)
                if '.lora_A.' in name:alphas['transformer.'+name.split('.lora_A.')[0]+'.alpha']=8.0
    model.load_lora_adapter(adapter,prefix='transformer',adapter_name='turbo',network_alphas=alphas)
    model.eval();del adapter
    for step,(values,indices) in enumerate(plans):
        cpu_time=model.time_embedder(model.time_proj(values))
        item=reference(key('time',step,name='temb'),cpu_time)
        original=next(x for x in cpu['cache_comparisons'] if x['name']==f'time-{step}')
        assert item['sha256']==original['expected_sha256']
        print('CPU prepare/heads/update',step,flush=True)
        video=native('input',step,name='video');audio=native('input',step,name='audio')
        prepared=torch.cat((model.token_refiner(model.context_embedder(text)),model.audio_proj_in(audio).to(text.dtype),model.proj_in(video).to(text.dtype)),dim=1)
        item=reference(key('prepare',step),prepared)
        if step==0:assert item['sha256']==next(c for c in phases['cases'] if c['name']=='prepare')['outputs'][0]['sha256']
        del prepared
        hidden=native('body',step,49);embedding=native('time',step,name='temb')
        head_hidden=model.norm_out(hidden,embedding,indices).float()
        reference(key('heads',step,name='video'),model.proj_out(head_hidden).index_select(1,vi))
        reference(key('heads',step,name='audio'),model.audio_proj_out(head_hidden).index_select(1,ai))
        del hidden,head_hidden
        for modality,sample in [('video',video),('audio',audio)]:
            scheduler=q.MiniMaxH3Scheduler(shift=12. if modality=='video' else 3.);scheduler.set_timesteps(9)
            velocity=native('heads',step,name=modality)
            updated=scheduler.step(velocity,scheduler.timesteps[step],sample,return_dict=False)[0]
            item=reference(key('latents',step,name=modality),updated)
            assert item['sha256']==expected[key('latents',step,name=modality)]
        del video,audio,embedding,cpu_time,velocity,updated;gc.collect()
    model.to_empty(device='meta');del model;gc.collect()

    # Load only each block's modulation projection, comparing native embeddings
    # at all eight steps. Do not treat original CPU embeddings as native inputs.
    for block in range(50):
        with torch.device('meta'):model=q.MiniMaxH3Transformer3DModel()
        prefix=f'transformer_blocks.{block}.adaln_proj.'
        weights=weights_for(lambda name:name.startswith(f'blocks.{block}.adaln_proj.'))
        model.transformer_blocks[block].adaln_proj.load_state_dict({k.removeprefix(prefix):v for k,v in weights.items()},strict=True,assign=True)
        for step in range(8):
            embedding=native('time',step,name='temb')
            reference(key('modulation',step,block,'modulations'),torch.stack(model.transformer_blocks[block].adaln_proj(embedding)))
        model.to_empty(device='meta');del model,weights,embedding;gc.collect()
        print('CPU modulation block',block,flush=True)

    def selected_reference(block,hidden,embedding,adaln,rotary,selected):
        dispatch=implementation.dispatch_attention_fn
        def attention(query,key,value,*args,**kwargs):
            value=dispatch(query.index_select(1,selected),key,value,*args,**kwargs)
            return torch.zeros_like(query).index_copy(1,selected,value)
        def before(_module,inputs):return (inputs[0].index_select(1,selected),*inputs[1:])
        def after(_module,_inputs,value):
            shape=list(value.shape);shape[1]=hidden.shape[1]
            return value.new_zeros(shape).index_copy(1,selected,value)
        hooks=[]
        try:
            for module in (block.attn.to_out[0],block.ff):hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
            implementation.dispatch_attention_fn=attention
            return block(hidden,embedding,adaln,rotary).index_select(1,selected)
        finally:
            implementation.dispatch_attention_fn=dispatch
            for hook in hooks:hook.remove()

    selected=next(c for c in phases['cases'] if c['name']=='body-0')['outputs'][0]['row_indices']
    selected_tensor=torch.tensor(selected,dtype=torch.int64)
    rotary=(read(roots['cos']),read(roots['sin']))
    for block_index in [0,49]:
        with torch.device('meta'):model=q.MiniMaxH3Transformer3DModel()
        model.load_state_dict(weights_for(lambda name:name.startswith(f'blocks.{block_index}.')),strict=False,assign=True)
        adapter={};alphas={}
        with safe_open(adapter_path,framework='pt',device='cpu') as source:
            for name in source.keys():
                if name.startswith(f'transformer_blocks.{block_index}.'):
                    adapter['transformer.'+name.replace('.default.weight','.weight')]=source.get_tensor(name)
                    if '.lora_A.' in name:alphas['transformer.'+name.split('.lora_A.')[0]+'.alpha']=8.0
        model.load_lora_adapter(adapter,prefix='transformer',adapter_name='turbo',network_alphas=alphas)
        model.eval();block=model.transformer_blocks[block_index];del adapter
        for step,(_,indices) in enumerate(plans):
            hidden=native('prepare',step) if block_index==0 else native('body',step,block_index-1)
            embedding=native('time',step,name='temb');adaln=indices*3+tags
            small=hidden[:,:128].contiguous();chosen=torch.arange(0,128,4)
            full=block(small,embedding,adaln[:128],(rotary[0][:128],rotary[1][:128]))
            sampled=selected_reference(block,small,embedding,adaln[:128],(rotary[0][:128],rotary[1][:128]),chosen)
            assert torch.equal(full.index_select(1,chosen),sampled)
            report['selected_query_validation'].append(dict(step=step,block=block_index,full_rows=128,selected_rows=32,byte_identical=True))
            del small,full,sampled
            print('CPU body reference',step,block_index,flush=True);started=time.monotonic()
            value=selected_reference(block,hidden,embedding,adaln,rotary,selected_tensor)
            reference(key('body',step,block_index),value,selected)
            report['stages'].append(dict(step=step,block=block_index,seconds=time.monotonic()-started))
            save(out/'cpu-checks.json',report)
            del hidden,embedding,value;gc.collect()
        model.to_empty(device='meta');del block,model;gc.collect()
    assert report['references']==464 and len(report['selected_query_validation'])==16
    for path,digest in report['source_pins'].items():assert sha(path)==digest,path
    save(out/'references.json',manifest);report.update(passed=True,reference_manifest_sha256=sha(out/'references.json'))
    save(out/'cpu-checks.json',report)
    print('CPU references complete; no GPU work',flush=True)


if __name__=='__main__':main()
