"""Per-step durations of the Core AI runtime profiler intervals (ms), median over steps 2..N."""
import json
import statistics
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    ev = [json.loads(line) for line in open(path)]
    begins, per_step, step = {}, defaultdict(list), 0
    for i, e in enumerate(ev):
        if e["kind"] == "begin":
            begins[i + 1] = e
        elif e["kind"] == "end":
            b = begins.get(e["interval"])
            if b is not None and b["phase"] != "load":
                per_step[(step, e["id"])].append((e["t"] - b["t"]) / 1e6)
        elif e["kind"] == "step":
            step += 1
    names = sorted({k[1] for k in per_step})
    walls = [e["wallNs"] / 1e6 for e in ev if e["kind"] == "step" and "wallNs" in e][1:]
    print(path.rsplit("/", 1)[-1], "wall per call median %.3f ms" % statistics.median(walls) if walls else "")
    for n in names:
        vals = [sum(per_step[(s, n)]) for s in range(1, step) if (s, n) in per_step]   # skip step 0 (first call)
        if vals:
            print("   %-28s median %.3f ms  (n=%d)" % (n, statistics.median(vals), len(vals)))
