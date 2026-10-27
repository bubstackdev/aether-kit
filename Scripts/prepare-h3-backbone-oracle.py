#!/usr/bin/env python3
"""CPU preparation/checkpoints for fifty corrected H3 blocks with bounded RAM.

Verify every base/adapter binding one block at a time. Reuse retained original
CPU modulation tables and the qualified three-block references. Add original
CPU checks at blocks 24 and 49 using complete captured native input contexts.
No GPU API, graph export, checkpoint mutation or full CPU trajectory claim.

Legacy, prototype-only (2026-10-18): this script puts the superseded aether-prototype's forge and the lab's vendored
Diffusers on sys.path, so it runs in neither Forge environment, and its /private/tmp inputs are gone. The 0.0.2 H3
qualification uses prepare-h3-profile-oracle.py instead.
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
    with open(path, 'rb') as file: return hashlib.file_digest(file, 'sha256').hexdigest()


def save(path, value): path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['lab','prototype','bundle','phases','phase-cpu','prior-chain','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--references',type=Path)
    args=ap.parse_args();lab=args.lab.resolve();prototype=args.prototype.resolve();out=args.output.resolve()
    if out.is_relative_to(lab) or out.is_relative_to(prototype):raise ValueError('source projects are read-only')
    out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(prototype/'forge'),str(lab/'forge'),str(lab/'forge/vendor/python'),str(lab/'forge/vendor/diffusers/src')]
    import qualify_block as q
    from h3_weights import CANONICAL_QKV_LAYOUT,convert_raw_transformer_key
    import torch
    from safetensors import safe_open
    import diffusers.models.transformers.transformer_minimax_h3 as implementation
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    base=json.loads(args.bundle.read_text());phase=json.loads(args.phases.read_text())
    old=json.loads(args.prior_chain.read_text());mod=json.loads(args.phase_cpu.read_text())
    assert base['complete'] and base['recipe']=='turbo8' and base['transformer_weight_layout']==CANONICAL_QKV_LAYOUT
    assert base['snapshot']==q.SNAPSHOT.name=='42ed227ee7df40d41602854ae760620d6eb651fe'
    assert base['diffusers_commit']=='7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941'
    assert (base['width'],base['height'],base['frames'],base['text_rows'])==(960,544,124,49)
    assert old['bundle_sha256']==sha(args.bundle) and old['phase_manifest_sha256']==sha(args.phases)
    assert old['block_indices']==[0,1,2] and mod['passed'] and len(base['parameters'])==50
    assert phase['predefined_relative_l2_limit']==old['predefined_relative_l2_limit']==0.02
    adapter_path=lab/'forge/vendor/weights/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors'
    assert sha(adapter_path)==base['adapter']['sha256']=='e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f'
    assert base['adapter']['execution']=='dynamic_lora' and base['adapter']['rank']==128 and base['adapter']['alpha']==8
    def raw(value):return value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    def read(item):
        data=Path(item['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256']
        value=torch.frombuffer(bytearray(data),dtype=getattr(torch,item['dtype'])).reshape(item['shape'])
        assert torch.isfinite(value).all() and value.device.type=='cpu';return value
    def fixture(name,value):
        data=raw(value);path=out/(name+'.raw');path.write_bytes(data)
        return dict(name='hidden_out',path=str(path),shape=list(value.shape),dtype='bfloat16',sha256=hashlib.sha256(data).hexdigest())
    manifest=dict(schema=2,block_indices=list(range(50)),reference_blocks=[0,1,2,24,49],weight_residency='one_block',
                  bundle_sha256=sha(args.bundle),phase_manifest_sha256=sha(args.phases),predefined_relative_l2_limit=0.02,
                  oracle_recipe_sha256=sha(__file__),cases=[])
    report=dict(schema=1,passed=False,gpu_work=False,scope='All 1100 binding identities; CPU numerical checkpoints 0,1,2,24,49 at complete captured native contexts, not CPU trajectory parity',
                source_pins={},bindings_verified=0,tables_verified=0,selected_query_validation=[],cases=[])
    for path in [Path(__file__).resolve(),args.bundle,args.phases,args.prior_chain,args.phase_cpu,
                 prototype/'forge/h3_weights.py',lab/'forge/qualify_block.py',
                 lab/'forge/vendor/diffusers/scripts/convert_minimax_h3_to_diffusers.py',Path(implementation.__file__)]:
        report['source_pins'][str(path)]=sha(path)
    for ordinal in range(2):
        name=f'body-{ordinal}';roots=next(x for x in phase['cases'] if x['name']==name)
        prior=next(x for x in old['cases'] if x['name']==name)
        steps=[]
        for b in range(50):
            comparison=next(x for x in mod['cache_comparisons'] if x['name']==f'block-{b}-step-{ordinal}')
            assert comparison['passed']
            path=args.phase_cpu.parent/f'fixtures/cache-reference/block-{b}-step-{ordinal}.raw'
            table=dict(name='temb',path=str(path.resolve()),dtype='bfloat16',shape=[6,(ordinal+1)*3,5376],sha256=comparison['expected_sha256'])
            assert sha(path)==table['sha256'];read(table);report['tables_verified']+=1
            ref=prior['steps'][b]['reference'] if b<3 else None
            expected=prior['steps'][b]['expected_output_sha256'] if b<3 else None
            if b<3:assert prior['steps'][b]['temb']['sha256']==table['sha256'] and ref is not None and expected is not None
            steps.append(dict(block=b,temb=table,reference=ref,expected_output_sha256=expected))
        manifest['cases'].append(dict(name=name,steps=steps))
    save(out/'cpu-checks.json',report)
    with torch.device('meta'):blueprint=q.MiniMaxH3Transformer3DModel()
    config=dict(blueprint.config);del blueprint
    converter=q.converter();index=json.loads((q.SNAPSHOT/'FL2VA/transformer/model.safetensors.index.json').read_text())['weight_map']
    def original_weights(b):
        converted={}
        for name,shard in sorted(index.items()):
            if name.startswith(f'blocks.{b}.'):
                with safe_open(q.SNAPSHOT/'FL2VA/transformer'/shard,framework='pt',device='cpu') as source:
                    converted.update(convert_raw_transformer_key(converter,name,source.get_tensor(name),config))
        return converted
    if args.prepare:
        # No fifty-block model/weight list is resident on CPU or GPU.
        with safe_open(adapter_path,framework='pt',device='cpu') as adapter:
            for b in range(50):
                converted=original_weights(b)
                for item in base['parameters'][b]:
                    name=f'transformer_blocks.{b}.'+item['parameter'].removeprefix('block.')
                    if '.lora_' in name:
                        tensor=adapter.get_tensor(name.replace('.turbo.weight','.default.weight'))
                    else:tensor=converted[name.replace('.base_layer.','.')]
                    assert tensor.device.type=='cpu' and list(tensor.shape)==item['shape'] and str(tensor.dtype).split('.')[-1]==item['dtype']
                    assert hashlib.sha256(raw(tensor)).hexdigest()==item['sha256']==sha(item['path'])
                    report['bindings_verified']+=1
                del converted,tensor;gc.collect()
                print('CPU verified block',b,'bindings',report['bindings_verified'],flush=True)
                save(out/'cpu-checks.json',report)
        assert report['bindings_verified']==1100
    else:
        capture=json.loads((args.references/'report.json').read_text())
        assert capture['captureComplete'] and not capture['passed'] and capture['mode']=='--capture'
        assert capture['bundleSHA256']==sha(args.bundle) and capture['phaseManifestSHA256']==sha(args.phases)
        prepared_path=Path(capture['chainManifestPath']);assert sha(prepared_path)==capture['chainManifestSHA256']
        prepared=json.loads(prepared_path.read_text());assert prepared['oracle_recipe_sha256']==sha(__file__)
        report['capture_report_sha256']=sha(args.references/'report.json')
        outputs={entry['name']:entry['outputs'] for entry in capture['runs']}
        for case in manifest['cases']:
            native=outputs[case['name']];assert [x['block'] for x in native]==list(range(50))
            for b,step in enumerate(case['steps']):
                if b<3:assert step['expected_output_sha256']==native[b]['sha256']
                step['expected_output_sha256']=native[b]['sha256']
        def selected_reference(block,hidden,embedding,tags,rotary,selected):
            dispatch=implementation.dispatch_attention_fn
            def attention(query,key,value,*a,**kw):
                result=dispatch(query.index_select(1,selected),key,value,*a,**kw)
                return torch.zeros_like(query).index_copy(1,selected,result)
            def before(_module,inputs):return (inputs[0].index_select(1,selected),*inputs[1:])
            def after(_module,_inputs,result):
                shape=list(result.shape);shape[1]=hidden.shape[1]
                return result.new_zeros(shape).index_copy(1,selected,result)
            hooks=[]
            try:
                for module in (block.attn.to_out[0],block.ff):
                    hooks += [module.register_forward_pre_hook(before),module.register_forward_hook(after)]
                implementation.dispatch_attention_fn=attention
                return block(hidden,embedding,tags,rotary).index_select(1,selected)
            finally:
                implementation.dispatch_attention_fn=dispatch
                for hook in hooks:hook.remove()
        for b in [24,49]:
            with torch.device('meta'):model=q.MiniMaxH3Transformer3DModel()
            converted=original_weights(b);model.load_state_dict(converted,strict=False,assign=True);del converted
            adapter_weights={};alphas={}
            with safe_open(adapter_path,framework='pt',device='cpu') as adapter:
                for key in adapter.keys():
                    if key.startswith(f'transformer_blocks.{b}.'):
                        adapter_weights['transformer.'+key.replace('.default.weight','.weight')]=adapter.get_tensor(key)
                        if '.lora_A.' in key:alphas['transformer.'+key.split('.lora_A.')[0]+'.alpha']=8.0
            model.load_lora_adapter(adapter_weights,prefix='transformer',adapter_name='turbo',network_alphas=alphas)
            model.eval();block=model.transformer_blocks[b];del adapter_weights
            for ordinal in range(2):
                name=f'body-{ordinal}';source=next(x for x in phase['cases'] if x['name']==name)
                roots={x['name']:x for x in source['inputs'] if not x['name'].startswith('w')}
                embedding=read(next(x for x in phase['cases'] if x['name']==f'time-{ordinal}')['outputs'][0])
                tags=read(roots['tags']).long();rotary=(read(roots['cos']),read(roots['sin']))
                hidden=read(outputs[name][b-1]);selected=torch.tensor(source['outputs'][0]['row_indices'],dtype=torch.int64)
                step=manifest['cases'][ordinal]['steps'][b]
                assert hashlib.sha256(raw(torch.stack(block.adaln_proj(embedding)))).hexdigest()==step['temb']['sha256']
                small=hidden[:,:128].contiguous();chosen=torch.arange(0,128,4)
                full=block(small,embedding,tags[:128],(rotary[0][:128],rotary[1][:128]))
                sampled=selected_reference(block,small,embedding,tags[:128],(rotary[0][:128],rotary[1][:128]),chosen)
                assert torch.equal(full.index_select(1,chosen),sampled)
                report['selected_query_validation'].append(dict(name=name,block=b,full_rows=128,selected_rows=32,byte_identical=True))
                del small,full,sampled
                print('CPU reference',name,'block',b,flush=True);started=time.monotonic()
                value=selected_reference(block,hidden,embedding,tags,rotary,selected)
                ref=fixture(name+f'-block{b}-reference',value)
                ref.update(row_indices=selected.tolist(),native_shape=[1,19333,5376]);step['reference']=ref
                report['cases'].append(dict(name=name,block=b,seconds=time.monotonic()-started,context_sha256=outputs[name][b-1]['sha256'],reference=ref))
                save(out/'cpu-checks.json',report)
                del hidden,embedding,tags,rotary,value;gc.collect()
            model.to_empty(device='meta');del block,model;gc.collect()
    for path,digest in report['source_pins'].items():assert sha(path)==digest,path
    save(out/'chain.json',manifest);report.update(passed=True,chain_manifest_sha256=sha(out/'chain.json'))
    save(out/'cpu-checks.json',report);print('CPU backbone preparation complete; no GPU work',flush=True)

if __name__=='__main__':main()
