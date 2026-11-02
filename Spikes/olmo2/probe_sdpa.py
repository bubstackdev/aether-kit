#!/usr/bin/env python3
"""Weight-free CPU attention precision RCA; run with Forge's pinned Python."""
import argparse
import asyncio
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import coreai_torch as ct
import coreai.runtime as rt

class SDPA(torch.nn.Module):
    def __init__(self, precise):
        super().__init__(); self.precise=precise
    def forward(self,q,k,v,mask):
        if self.precise:
            return F.scaled_dot_product_attention(q.float(),k.float(),v.float(),attn_mask=mask,enable_gqa=True).to(q.dtype)
        return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True)

async def run(output):
    torch.manual_seed(0)
    q=torch.randn(1,4,3,64).half(); k=torch.randn(1,2,9,64).half(); v=torch.randn(1,2,9,64).half()
    mask=torch.arange(9)[None,:] <= torch.arange(6,9)[:,None]
    expected=SDPA(True)(q,k,v,mask).numpy().astype(np.float32)
    record={}
    for precise in (False,True):
        module=SDPA(precise)
        ep=torch.export.export(module,(q,k,v,mask)).run_decompositions(ct.get_decomp_table())
        converter=ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        converter.add_exported_program(ep,input_names=['q','k','v','mask'],output_names=['out'],state_names=[],entrypoint_name='main')
        program=converter.to_coreai(); program.optimize()
        path=output/f'sdpa-{precise}.aimodel'; program.save_asset(path)
        model=await rt.AIModel.load(path,rt.SpecializationOptions.cpu_only()); fn=model.load_function('main')
        out=await fn({n:rt.NDArray(t.numpy()) for n,t in zip(('q','k','v','mask'),(q,k,v,mask))},{})
        got=out['out'].numpy().astype(np.float32)
        record['float32' if precise else 'float16']=dict(maxAbs=float(np.abs(got-expected).max()),rmse=float(np.sqrt(np.mean((got-expected)**2))))
    print(json.dumps(record,indent=2))
    (output/'sdpa-rca.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Isolate Core AI CPU SDPA precision against float32 accumulation')
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    asyncio.run(run(args.output))
