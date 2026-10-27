"""Does coreai-torch convert F.grid_sample (bilinear, zeros padding, align_corners=False)?"""
import torch
import torch.nn.functional as F

import coreai_torch as ct


class M(torch.nn.Module):
    def forward(self, planes, grid):
        return F.grid_sample(planes, grid, align_corners=False, mode="bilinear")


ex = (torch.randn(3, 40, 64, 64), torch.rand(3, 1, 4096, 2) * 2 - 1)
ep = torch.export.export(M(), ex).run_decompositions(ct.get_decomp_table())
ops = sorted({str(n.target) for n in ep.graph.nodes if n.op == "call_function"})
print("ops after decomposition:", ops)
c = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
c.add_exported_program(ep, input_names=["planes", "grid"], output_names=["out"])
p = c.to_coreai()
p.optimize()
print("converted OK")
