"""Exploratory error analysis of a set of runs (not part of the pre-set comparisons).

Per relation: F1 per config averaged over seeds. Per test item: how often a
config is right where the baseline is wrong (and the reverse), summed over
common seeds, with the most frequent flips listed.

    python scripts/error_analysis.py runs/clean_v1 --test data/processed/clean_v1/test.jsonl --n 500
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trf.metrics import evaluate
from trf.schema import NO_RELATION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--test", required=True)
    ap.add_argument("--n", default="500", help="train size, or 'all'")
    ap.add_argument("--baseline", default="e0_baseline")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()

    test = [json.loads(l) for l in open(a.test, encoding="utf-8")]
    gold = [d["relation"]["label"] for d in test]
    want = None if a.n == "all" else int(a.n)
    runs = defaultdict(dict)
    for m in Path(a.root).glob("*/*/test_predictions.jsonl"):
        cfg = json.loads((m.parent / "config.json").read_text())
        if cfg["train_limit"] == want:
            runs[cfg["name"]][cfg["seed"]] = [json.loads(l)["pred"] for l in open(m, encoding="utf-8")]
    if a.baseline not in runs:
        sys.exit("no baseline runs at this size")
    labels = sorted(set(gold) - {NO_RELATION})
    configs = sorted(runs)
    shared = set.intersection(*(set(v) for v in runs.values()))     # same seeds for every config
    runs = {c: {s: p for s, p in v.items() if s in shared} for c, v in runs.items()}

    print(f"Per-relation F1, n_train={a.n}, mean over seeds (seeds: "
          f"{', '.join(f'{c}={len(runs[c])}' for c in configs)})")
    per = {c: defaultdict(float) for c in configs}
    for c in configs:
        for preds in runs[c].values():
            r = evaluate(gold, preds)["per_relation"]
            for l in labels:
                per[c][l] += r[l]["f1"] / len(runs[c])
    support = Counter(gold)
    print(f"{'relation':18} {'n':>4} " + " ".join(f"{c[:10]:>10}" for c in configs))
    for l in sorted(labels, key=lambda l: -support[l]):
        print(f"{l:18} {support[l]:4} " + " ".join(f"{per[c][l]:10.3f}" for c in configs))

    base = runs[a.baseline]
    for c in configs:
        if c == a.baseline:
            continue
        common = sorted(set(runs[c]) & set(base))
        win, loss = Counter(), Counter()
        for s in common:
            for i, g in enumerate(gold):
                ok_c, ok_b = runs[c][s][i] == g, base[s][i] == g
                if ok_c and not ok_b:
                    win[i] += 1
                elif ok_b and not ok_c:
                    loss[i] += 1
        print(f"\n== {c} vs {a.baseline} over {len(common)} seeds: "
              f"{sum(win.values())} item-runs fixed, {sum(loss.values())} broken")
        by_rel = Counter(); by_infl = Counter()
        for i, k in win.items():
            by_rel[gold[i]] += k; by_infl[test[i]["meta"]["match"][1]] += k
        for i, k in loss.items():
            by_rel[gold[i]] -= k; by_infl[test[i]["meta"]["match"][1]] -= k
        print("  net fixed by gold label:", dict(by_rel.most_common()))
        print("  net fixed by object match:", dict(by_infl))
        for name, cnt in (("fixed", win), ("broken", loss)):
            print(f"  most often {name}:")
            for i, k in cnt.most_common(a.show):
                d = test[i]; s, o = d["entities"]
                print(f"    {k}/{len(common)}  gold={gold[i]:16} E1={s['text']}  E2={o['text']}")


if __name__ == "__main__":
    main()
