"""Plan-named secondary analyses on existing runs (exploratory; see plan sections 8 and 10).

Plan 10 ("Morphology does not improve F1"): examine low-data settings, case-signalled
relations and analyzer quality. Plan 8: macro-F1 and per-relation F1. Everything here
is computed on runs that already exist; every comparison is printed, not only the
ones that clear zero.

  case-signalled relation: defined from TRAIN gold morphology only. For each relation,
      the share of training objects carrying its most common non-nominative case;
      "case-signalled" if that share >= 0.6 (threshold fixed before looking at test).
  analyzer quality: test items where Stanza's object case equals the gold object case
      ("stanza case right") vs not ("stanza case wrong").

    python scripts/probe_leads.py runs/clean_v1 --n 500
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trf.encode import morph_for
from trf.metrics import evaluate
from trf.schema import NO_RELATION, load_jsonl

D = "data/processed/clean_v1"
CONFIGS = ["e1_lemma", "e1dup_surface_control", "e2a_case_tag", "e2b_case_emb", "e2c_case_tag_emb"]


def micro(gold, pred, idx):
    tp = sum(gold[i] == pred[i] != NO_RELATION for i in idx)
    npred = sum(pred[i] != NO_RELATION for i in idx)
    ngold = sum(gold[i] != NO_RELATION for i in idx)
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def rel_f1(rel):
    def f(gold, pred, idx):
        tp = sum(gold[i] == pred[i] == rel for i in idx)
        npred = sum(pred[i] == rel for i in idx)
        ngold = sum(gold[i] == rel for i in idx)
        p = tp / npred if npred else 0.0
        r = tp / ngold if ngold else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0
    return f


def macro(gold, pred, idx):
    return evaluate([gold[i] for i in idx], [pred[i] for i in idx])["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n", default="500", help="train size tag (500, 100, all)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--out", help="write all rows as JSON here")
    a = ap.parse_args()
    rng = random.Random(0)

    # case-signalled relations from TRAIN gold morphology
    train = load_jsonl(f"{D}/train_goldmorph.jsonl")
    by_rel = defaultdict(Counter)
    for ex in train:
        by_rel[ex.label][morph_for(ex, "obj").get("case") or "?"] += 1
    print("TRAIN object case by relation (gold morphology):")
    signalled = []
    for rel, c in sorted(by_rel.items()):
        n = sum(c.values())
        non_nom = [(k, v) for k, v in c.most_common() if k not in ("Nom", "?")]
        top, share = (non_nom[0][0], non_nom[0][1] / n) if non_nom else ("-", 0.0)
        flag = rel != NO_RELATION and share >= 0.6
        if flag:
            signalled.append(rel)
        print(f"  {rel:18} n={n:4}  top non-Nom {top:5} {share:.2f}  {'CASE-SIGNALLED' if flag else ''}"
              f"   {dict(c.most_common(4))}")

    test = load_jsonl(f"{D}/test.jsonl")
    gtest = load_jsonl(f"{D}/test_goldmorph.jsonl")
    gold = [ex.label for ex in test]
    allidx = list(range(len(test)))
    right = [i for i in allidx if morph_for(test[i], "obj").get("case") == morph_for(gtest[i], "obj").get("case")]
    wrong = [i for i in allidx if i not in set(right)]
    csig = [i for i in allidx if gold[i] in signalled]
    print(f"\ntest: {len(test)} items; stanza obj case right {len(right)}, wrong {len(wrong)}; "
          f"gold in case-signalled relations {len(csig)}")

    def load(cfg):
        out = {}
        for p in Path(a.root, cfg).glob("seed*/test_predictions.jsonl"):
            c = json.loads((p.parent / "config.json").read_text())
            if str(c["train_limit"] or "all") == a.n:
                out[c["seed"]] = [json.loads(l)["pred"] for l in open(p, encoding="utf-8")]
        return out

    base = load("e0_baseline")
    analyses = [("micro, all", micro, allidx), ("macro, all", macro, allidx),
                ("micro, case-signalled gold", micro, csig),
                ("micro, stanza case right", micro, right), ("micro, stanza case wrong", micro, wrong)]
    rels = sorted({g for g in gold if g != NO_RELATION})
    analyses += [(f"F1 {r} ({sum(g == r for g in gold)})", rel_f1(r), allidx) for r in rels]

    rows = []
    print(f"\nn={a.n}, baseline seeds {sorted(base)}; diff vs E0, 95% CI seeds+items, {a.n_boot} resamples")
    for cfg in a.configs:
        runs = load(cfg)
        common = sorted(set(runs) & set(base))
        if not common:
            continue
        print(f"\n== {cfg} ({len(common)} seeds)")
        for name, fn, idx in analyses:
            def diff(ss, resample):
                tot = 0.0
                for s in ss:
                    ii = [rng.choice(idx) for _ in idx] if resample else idx
                    tot += fn(gold, runs[s], ii) - fn(gold, base[s], ii)
                return tot / len(ss)
            d = diff(common, False)
            bs = sorted(diff([rng.choice(common) for _ in common], True) for _ in range(a.n_boot))
            lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]
            pos = sum(fn(gold, runs[s], idx) > fn(gold, base[s], idx) for s in common)
            star = " *" if lo > 0 or hi < 0 else ""
            print(f"  {name:38} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {pos}/{len(common)} seeds +{star}")
            rows.append({"config": cfg, "analysis": name, "diff": d, "lo": lo, "hi": hi,
                         "seeds_pos": pos, "seeds": len(common), "n": a.n})
    if a.out:
        Path(a.out).write_text(json.dumps({"case_signalled": signalled, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
