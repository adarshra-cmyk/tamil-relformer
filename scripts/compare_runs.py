"""Paired comparison of configs against the baseline on one test set.

For every (config, train size), F1 is averaged over seeds. The difference to
the baseline is paired by seed (same seed = same training subset and init).
Its 95% interval comes from a two-level bootstrap: resample seeds, then test
items, so it covers both seed-to-seed and test-sample variation. With fewer
than 3 common seeds the seed level is too small to resample, and the interval
covers test items only (marked "items only").

Subsets: objects that appear inflected vs exactly as in Wikidata; items whose
gold label is not LOC:LOCATED_IN (robustness check); and (with
--train) "obj form unseen": the object entity occurs in that run's training
subset, but never in this surface form. The subset is computed per seed,
because the n-item training subset depends on the seed.

    python scripts/compare_runs.py runs/clean_v1 --test data/processed/clean_v1/test.jsonl \
        --train data/processed/clean_v1/train.jsonl [--csv out.csv]
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trf.schema import NO_RELATION, load_jsonl
from trf.train import subsample


def micro_f1(gold, pred, idx):
    tp = sum(gold[i] == pred[i] != NO_RELATION for i in idx)
    npred = sum(pred[i] != NO_RELATION for i in idx)
    ngold = sum(gold[i] != NO_RELATION for i in idx)
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", help="training file; adds the per-seed 'obj form unseen' subset")
    ap.add_argument("--baseline", default="e0_baseline")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", help="also write the table here")
    ap.add_argument("--type-constraint", action="store_true",
                    help="replace predictions whose relation type signature does not match the "
                         "entity types with NO_RELATION (decoding constraint, all configs alike)")
    ap.add_argument("--name-from-dir", action="store_true",
                    help="key runs by their config directory name instead of config.json name "
                         "(compare the same config trained on different data via a symlinked root)")
    ap.add_argument("--relations", default="configs/relations.json")
    a = ap.parse_args()

    test = [json.loads(l) for l in open(a.test, encoding="utf-8")]
    gold = [d["relation"]["label"] for d in test]
    fixed = {"all": list(range(len(test))),
             "obj inflected": [i for i, d in enumerate(test) if d["meta"]["match"][1] == "inflected"],
             "obj exact": [i for i, d in enumerate(test) if d["meta"]["match"][1] == "exact"],
             # robustness: drop the formulaic, most frequent relation
             "no LOCATED_IN": [i for i, g in enumerate(gold) if g != "LOC:LOCATED_IN"]}
    train_full = load_jsonl(a.train) if a.train else None
    unseen_cache = {}

    def unseen(n, seed):
        key = (n, seed)
        if key not in unseen_cache:
            tr = train_full if n == "all" else subsample(list(train_full), n, seed)
            seen_q, seen_form = set(), set()
            for ex in tr:
                for e, q in zip(ex.entities, (ex.meta["wd_subject"], ex.meta["wd_object"])):
                    seen_q.add(q); seen_form.add((q, e.text))
            unseen_cache[key] = [i for i, d in enumerate(test) if d["meta"]["wd_object"] in seen_q
                                 and (d["meta"]["wd_object"], d["entities"][1]["text"]) not in seen_form]
        return unseen_cache[key]

    def subset(name, n, seed):
        return unseen(n, seed) if name == "obj form unseen" else fixed[name]

    names = list(fixed) + (["obj form unseen"] if train_full else [])

    sig = {r["label"]: (r["subj"], r["obj"]) for r in json.load(open(a.relations, encoding="utf-8"))["relations"]}
    types = [(d["entities"][0]["type"], d["entities"][1]["type"]) for d in test]
    constrain = (lambda ps: [p if p == NO_RELATION or sig.get(p) == t else NO_RELATION for p, t in zip(ps, types)]) \
        if a.type_constraint else (lambda ps: ps)
    if a.type_constraint:
        print("type-constrained decoding: predictions with the wrong type signature -> NO_RELATION")

    runs = defaultdict(dict)                  # (config, n) -> seed -> preds
    for m in Path(a.root).glob("*/*/test_predictions.jsonl"):
        cfg = json.loads((m.parent / "config.json").read_text())
        preds = constrain([json.loads(l)["pred"] for l in open(m, encoding="utf-8")])
        if len(preds) != len(test):
            sys.exit(f"{m}: {len(preds)} predictions for {len(test)} test items")
        name = m.parent.parent.name if a.name_from_dir else cfg["name"]
        runs[(name, cfg["train_limit"] or "all")][cfg["seed"]] = preds

    rng = random.Random(a.seed)
    rows = []
    for n in sorted({n for _, n in runs}, key=lambda x: (x == "all", x if x != "all" else 0)):
        base = runs.get((a.baseline, n))
        if not base:
            continue
        print(f"\n== train size {n}  (seeds: "
              f"{', '.join(f'{c}={len(s)}' for (c, m), s in sorted(runs.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))) if m == n)})")
        print(f"{'config':16} {'subset':16} {'items':>5} {'F1':>7} {'vs base':>8}  {'95% CI':>17}")
        for (c, m), seeds in sorted(runs.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            if m != n:
                continue
            common = sorted(set(seeds) & set(base))
            for name in names:
                f = sum(micro_f1(gold, seeds[s], subset(name, n, s)) for s in seeds) / len(seeds)
                size = round(sum(len(subset(name, n, s)) for s in seeds) / len(seeds))
                row = {"n_train": n, "config": c, "subset": name, "items": size, "seeds": len(seeds),
                       "f1": round(f, 4), "diff": "", "lo": "", "hi": "", "ci": ""}
                if c != a.baseline and common:
                    def diff(ss, resample):
                        tot = 0.0
                        for s in ss:
                            idx = subset(name, n, s)
                            if resample:
                                idx = [rng.choice(idx) for _ in idx]
                            tot += micro_f1(gold, seeds[s], idx) - micro_f1(gold, base[s], idx)
                        return tot / len(ss)
                    d = diff(common, False)
                    two_level = len(common) >= 3
                    bs = sorted(diff([rng.choice(common) for _ in common] if two_level else common, True)
                                for _ in range(a.n_boot))
                    lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]
                    row.update(diff=round(d, 4), lo=round(lo, 4), hi=round(hi, 4),
                               ci="seeds+items" if two_level else "items only")
                    star = " *" if lo > 0 or hi < 0 else ""
                    print(f"{c:16} {name:16} {size:5} {f:7.3f} {d:+8.3f}  [{lo:+.3f}, {hi:+.3f}]{star}"
                          f"{'' if two_level else '  (items only)'}")
                else:
                    print(f"{c:16} {name:16} {size:5} {f:7.3f}")
                rows.append(row)
        # per-seed differences on the full test set: how consistent is each gap
        print(f"  per-seed difference vs {a.baseline} (all items):")
        for (c, m), seeds in sorted(runs.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            if m != n or c == a.baseline:
                continue
            common = sorted(set(seeds) & set(base))
            ds = [micro_f1(gold, seeds[s], fixed["all"]) - micro_f1(gold, base[s], fixed["all"]) for s in common]
            if ds:
                pos = sum(d > 0 for d in ds)
                print(f"    {c:22} " + "  ".join(f"s{s}:{d:+.3f}" for s, d in zip(common, ds))
                      + f"   ({pos}/{len(ds)} seeds positive)")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
