#!/usr/bin/env python3
"""CPU-only original-module references for a bounded corrected H3 chain.

Prepare block-specific tables first, then compute sampled CPU references using
complete captured native contexts. This does not claim an independent CPU
trajectory. Checkpoints/lab/prototype stay read-only; no graph export or GPU API.

Legacy, prototype-only (2026-10-18): this script puts the superseded aether-prototype's forge and the lab's vendored
Diffusers on sys.path, so it runs in neither Forge environment, and its /private/tmp inputs are gone. The 0.0.2 H3
qualification uses prepare-h3-profile-oracle.py instead.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--prototype', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--phases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--references', type=Path, metavar='CAPTURE_DIRECTORY')
    args = parser.parse_args()
    lab, prototype, out = args.lab.resolve(), args.prototype.resolve(), args.output.resolve()
    if out.is_relative_to(lab) or out.is_relative_to(prototype):
        raise ValueError('output must be outside read-only source projects')
    out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(prototype / 'forge'), str(lab / 'forge'),
                   str(lab / 'forge/vendor/python'), str(lab / 'forge/vendor/diffusers/src')]
    import qualify_block as q
    from h3_weights import CANONICAL_QKV_LAYOUT, convert_raw_transformer_key
    import torch
    from safetensors import safe_open
    import diffusers.models.transformers.transformer_minimax_h3 as implementation
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    base = json.loads(args.bundle.read_text())
    phase = json.loads(args.phases.read_text())
    environment = json.loads((lab / 'records/environment.json').read_text())
    assert environment['unmodifiedDependencies']
    assert base['snapshot'] == q.SNAPSHOT.name == '42ed227ee7df40d41602854ae760620d6eb651fe'
    assert base['diffusers_commit'] == environment['diffusers']['commit'] == '7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941'
    assert base['transformer_weight_layout'] == CANONICAL_QKV_LAYOUT
    assert base['recipe'] == 'turbo8' and base['complete']
    assert (base['width'], base['height'], base['frames'], base['text_rows']) == (960,544,124,49)
    assert base['adapter']['execution'] == 'dynamic_lora' and base['adapter']['rank'] == 128 and base['adapter']['alpha'] == 8
    assert phase['predefined_relative_l2_limit'] == 0.02
    adapter_path = lab / 'forge/vendor/weights/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors'
    assert sha(adapter_path) == base['adapter']['sha256'] == 'e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f'
    def raw(value):
        return value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    def read(item):
        data = Path(item['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        value = torch.frombuffer(bytearray(data), dtype=getattr(torch, item['dtype'])).reshape(item['shape'])
        assert value.device.type == 'cpu' and torch.isfinite(value).all()
        return value
    def fixture(name, value, tensor_name):
        data = raw(value); path = out / (name + '.raw'); path.write_bytes(data)
        return dict(name=tensor_name, path=str(path), dtype=str(value.dtype).split('.')[-1],
                    shape=list(value.shape), sha256=hashlib.sha256(data).hexdigest())
    manifest = dict(schema=1, block_indices=[0,1,2], bundle_sha256=sha(args.bundle),
                    phase_manifest_sha256=sha(args.phases), predefined_relative_l2_limit=0.02,
                    oracle_recipe_sha256=sha(__file__), cases=[])
    report = dict(schema=1, passed=False, gpu_work=False,
                  scope='Independent CPU original blocks at complete captured native contexts; not independent CPU chain trajectory',
                  source_recipe_sha256=sha(__file__), source_pins={}, selected_query_validation=[], cases=[])
    for path in [Path(__file__).resolve(), prototype/'forge/h3_weights.py',
                 lab/'forge/qualify_block.py', lab/'forge/vendor/diffusers/scripts/convert_minimax_h3_to_diffusers.py',
                 Path(implementation.__file__), lab/'records/environment.json', args.bundle, args.phases]:
        report['source_pins'][str(path)] = sha(path)
    capture = None
    capture_outputs = {}
    if args.references:
        capture_path = args.references / 'report.json'
        capture = json.loads(capture_path.read_text())
        assert capture['captureComplete'] and capture['mode'] == '--capture' and not capture['passed']
        assert capture['bundleSHA256'] == manifest['bundle_sha256']
        assert capture['phaseManifestSHA256'] == manifest['phase_manifest_sha256']
        assert len(capture['runs']) == 2 and all(x['role'] == 'capture' for x in capture['runs'])
        prepared_path = Path(capture['chainManifestPath'])
        prepared = json.loads(prepared_path.read_text())
        assert sha(prepared_path) == capture['chainManifestSHA256']
        assert prepared['oracle_recipe_sha256'] == sha(__file__)
        assert prepared['bundle_sha256'] == manifest['bundle_sha256']
        assert prepared['phase_manifest_sha256'] == manifest['phase_manifest_sha256']
        for entry in capture['runs']:
            assert [x['block'] for x in entry['outputs']] == [0,1,2]
            for tensor in entry['outputs']:
                assert tensor['dtype'] == 'bfloat16' and tensor['shape'] == [1,19333,5376]
            capture_outputs[entry['name']] = entry['outputs']
        report['captureReportSHA256'] = sha(capture_path)
    save(out/'cpu-checks.json', report)
    with torch.device('meta'):
        model = q.MiniMaxH3Transformer3DModel()
    config, converter = dict(model.config), q.converter()
    index = json.loads((q.SNAPSHOT/'FL2VA/transformer/model.safetensors.index.json').read_text())['weight_map']
    weights = {}
    print('CPU load original corrected blocks 0, 1, 2', flush=True)
    for name, shard in sorted(index.items()):
        if not any(name.startswith(f'blocks.{b}.') for b in range(3)):
            continue
        with safe_open(q.SNAPSHOT/'FL2VA/transformer'/shard, framework='pt', device='cpu') as source:
            weights.update(convert_raw_transformer_key(converter, name, source.get_tensor(name), config))
    model.load_state_dict(weights, strict=False, assign=True)
    del weights
    adapter, alphas = {}, {}
    with safe_open(adapter_path, framework='pt', device='cpu') as source:
        assert float(source.metadata()['alpha']) == 8
        for key in source.keys():
            if any(key.startswith(f'transformer_blocks.{b}.') for b in range(3)):
                adapter['transformer.'+key.replace('.default.weight','.weight')] = source.get_tensor(key)
                if '.lora_A.' in key:
                    alphas['transformer.'+key.split('.lora_A.')[0]+'.alpha'] = 8.0
    model.load_lora_adapter(adapter, prefix='transformer', adapter_name='turbo', network_alphas=alphas)
    model.eval(); del adapter
    # Independently loaded checkpoint/converter/PEFT parameters must match all
    # exported bindings before those files are admitted to the native chain.
    for b in range(3):
        block = model.transformer_blocks[b]
        for item in base['parameters'][b]:
            parameter = block.get_parameter(item['parameter'].removeprefix('block.'))
            assert parameter.device.type == 'cpu' and str(parameter.dtype).split('.')[-1] == item['dtype']
            assert list(parameter.shape) == item['shape']
            assert hashlib.sha256(raw(parameter)).hexdigest() == item['sha256'] == sha(item['path'])
        print('CPU verified all 22 exported bindings for block', b, flush=True)
    def selected_reference(block, hidden, embedding, tags, rotary, selected):
        # Original CPU attention retains every key/value; only query rows and
        # independent row-wise tails are sampled. Validate against full source
        # execution on 128 rows before using this instrumentation at 19333 rows.
        dispatch = implementation.dispatch_attention_fn
        def attention(query, key, value, *args, **kwargs):
            result = dispatch(query.index_select(1, selected), key, value, *args, **kwargs)
            return torch.zeros_like(query).index_copy(1, selected, result)
        def before(_module, inputs): return (inputs[0].index_select(1, selected), *inputs[1:])
        def after(_module, _inputs, result):
            shape = list(result.shape); shape[1] = hidden.shape[1]
            return result.new_zeros(shape).index_copy(1, selected, result)
        hooks = []
        try:
            for module in (block.attn.to_out[0], block.ff):
                hooks += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
            implementation.dispatch_attention_fn = attention
            return block(hidden, embedding, tags, rotary).index_select(1, selected)
        finally:
            implementation.dispatch_attention_fn = dispatch
            for hook in hooks: hook.remove()
    for ordinal in range(2):
        name = f'body-{ordinal}'
        case = next(x for x in phase['cases'] if x['name'] == name)
        roots = {x['name']:x for x in case['inputs'] if not x['name'].startswith('w')}
        embedding = read(next(x for x in phase['cases'] if x['name'] == f'time-{ordinal}')['outputs'][0])
        tags = read(roots['tags']).long(); rotary = (read(roots['cos']),read(roots['sin']))
        initial = read(roots['hidden'])
        selected = torch.tensor(case['outputs'][0]['row_indices'],dtype=torch.int64)
        steps = []
        for b in range(3):
            print('CPU',name,'block',b,'tables' if args.prepare else 'production-context reference',flush=True)
            started = time.monotonic(); block = model.transformer_blocks[b]
            table = torch.stack(block.adaln_proj(embedding))
            table_file = fixture(name+f'-block{b}-temb',table,'temb')
            if b == 0: assert table_file['sha256'] == roots['temb']['sha256']
            reference = case['outputs'][0] if b == 0 else None
            if args.prepare:
                small = initial[:,:128].contiguous(); small_tags=tags[:128]
                small_rotary=(rotary[0][:128],rotary[1][:128]); chosen=torch.arange(0,128,4)
                full=block(small,embedding,small_tags,small_rotary)
                sampled=selected_reference(block,small,embedding,small_tags,small_rotary,chosen)
                assert torch.equal(full.index_select(1,chosen),sampled)
                report['selected_query_validation'].append(dict(name=name,block=b,full_rows=128,selected_rows=32,byte_identical=True))
                del small,full,sampled
            else:
                previous = next(x for x in prepared['cases'] if x['name']==name)['steps'][b]
                assert previous['temb']['sha256'] == table_file['sha256']
                hidden = initial if b == 0 else read(capture_outputs[name][b-1])
                sampled = selected_reference(block,hidden,embedding,tags,rotary,selected)
                generated = fixture(name+f'-block{b}-reference',sampled,'hidden_out')
                generated.update(row_indices=selected.tolist(),native_shape=list(initial.shape))
                if b == 0:
                    assert generated['sha256'] == reference['sha256']
                reference = generated
                del hidden,sampled
            steps.append(dict(block=b,temb=table_file,reference=reference,
                              expected_output_sha256=None if args.prepare else capture_outputs[name][b]['sha256']))
            report['cases'].append(dict(name=name,block=b,seconds=time.monotonic()-started,reference=reference,
                                        contextSHA256=roots['hidden']['sha256'] if args.prepare or b==0 else capture_outputs[name][b-1]['sha256']))
            save(out/'cpu-checks.json',report);gc.collect()
        manifest['cases'].append(dict(name=name,steps=steps))
        del initial,embedding,tags,rotary;gc.collect()
    for path,digest in report['source_pins'].items(): assert sha(path)==digest,path
    save(out/'chain.json',manifest)
    report.update(passed=True,chain_manifest_sha256=sha(out/'chain.json'))
    save(out/'cpu-checks.json',report)
    print('CPU preparation complete; no GPU work',flush=True)

if __name__ == '__main__': main()
