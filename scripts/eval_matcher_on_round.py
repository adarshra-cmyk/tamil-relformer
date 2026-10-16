"""Score a candidate file against an annotated round (span/type judgements).

Like eval_matcher_on_pilot.py, but for a round made by make_clean_round.py:
the key gives the original candidates and their split; pass-1 outputs (and
reconciliations) give span_ok/type_ok. An annotated item is kept_same,
kept_changed (spans/types differ), relinked (same sentence and subject, an
overlapping object span) or dropped in the new candidates.

    python scripts/eval_matcher_on_round.py NEW_CANDS --round data/annotation/clean_v1 [--split train]
"""
import argparse
import glob
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cands")
    ap.add_argument("--round", default="data/annotation/clean_v1")
    ap.add_argument("--split", choices=["all", "train", "dev", "test"], default="all")
    a = ap.parse_args()

    key = [k for k in map(json.loads, open(f"{a.round}/key.jsonl", encoding="utf-8"))
           if a.split == "all" or k["split"] == a.split]
    ann = {}
    for f in sorted(glob.glob(f"{a.round}/llm/out_pass1_b*.jsonl")) + sorted(glob.glob(f"{a.round}/llm/out_recon_b*.jsonl")):
        for d in map(json.loads, open(f, encoding="utf-8")):
            ann[d["item_id"]] = d
    K = lambda d: (d["source_id"], d["meta"]["wd_subject"], d["meta"]["wd_object"])
    wanted = {K(k["candidate"]): k for k in key}
    found, by_subj = {}, {}
    for l in open(a.cands, encoding="utf-8"):
        d = json.loads(l)
        if K(d) in wanted:
            found[K(d)] = d
        if (d["source_id"], d["meta"]["wd_subject"]) in {kk[:2] for kk in wanted}:
            by_subj.setdefault(K(d)[:2], []).append(d)
    res = Counter()
    for kk, k in wanted.items():
        r = ann[k["item_id"]]
        good = r["span_ok"] == "yes" and r["type_ok"] == "yes"
        old, new = k["candidate"], found.get(kk)
        if new is None:
            o = old["entities"][1]
            out = "relinked" if any(n["entities"][1]["start"] < o["end"] and o["start"] < n["entities"][1]["end"]
                                    for n in by_subj.get(kk[:2], [])) else "dropped"
        else:
            same = [(e["start"], e["end"], e["type"]) for e in old["entities"]] == \
                   [(e["start"], e["end"], e["type"]) for e in new["entities"]]
            out = "kept_same" if same else "kept_changed"
        res[(good, out)] += 1
    n_good = sum(v for (g, _), v in res.items() if g); n_bad = sum(v for (g, _), v in res.items() if not g)
    print(f"split: {a.split}")
    print(f"{'':16}{'usable':>8}{'error':>8}")
    for out in ("kept_same", "kept_changed", "relinked", "dropped"):
        print(f"{out:16}{res[(True, out)]:8}{res[(False, out)]:8}")
    print(f"usable kept: {(n_good - res[(True, 'dropped')]) / n_good:.3f}; "
          f"errors removed/changed/relinked: {(n_bad - res[(False, 'kept_same')]) / n_bad:.3f}; "
          f"error rate among kept-same: {res[(False, 'kept_same')] / max(1, res[(False, 'kept_same')] + res[(True, 'kept_same')]):.3f}")


if __name__ == "__main__":
    main()
