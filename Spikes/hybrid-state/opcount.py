"""Op distribution per exported asset (CPU file parse, no model load)."""
import json
import sys
from pathlib import Path

import coreai.runtime as rt

root = Path.home() / ".cache/huggingface/aether-kit/spikes" / (sys.argv[2] if len(sys.argv) > 2 else "granite")
res = {}
for p in sorted(root.glob("*.aimodel")):
    s = rt._AIModelAsset(p).summary(True)
    dist = dict(s.operation_distribution)
    res[p.stem] = {"functions": s.function_names, "total": sum(dist.values()), "compute": s.compute_types,
                   "storage": s.storage_types, "ops": dist}
    print(p.stem, s.function_names, "total ops", sum(dist.values()), s.compute_types)
json.dump(res, open(sys.argv[1], "w"), indent=1)
