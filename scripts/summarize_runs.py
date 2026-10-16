"""Table of test micro-F1 / macro-F1 per config and train size, mean +/- sd over seeds."""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
cells = defaultdict(list)
for m in root.glob("*/*/metrics.json"):
    cfg = json.loads((m.parent / "config.json").read_text())
    t = json.loads(m.read_text())["test"]
    cells[(cfg["name"], cfg["train_limit"] or "all")].append((t["micro_f1"], t["macro_f1"]))
fmt = lambda xs: f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"
print(f"{'config':20} {'n_train':>8} {'seeds':>5}  {'micro-F1':>15}  {'macro-F1':>15}")
for (name, n), v in sorted(cells.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
    print(f"{name:20} {str(n):>8} {len(v):>5}  {fmt([a for a, _ in v]):>15}  {fmt([b for _, b in v]):>15}")
