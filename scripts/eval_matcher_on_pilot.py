"""Score a candidate file against pilot judgements (span/type OK or not).

For each pilot item, look for the same (article sentence, subject, object) in
the new candidates and report whether it was dropped, kept unchanged, or kept
with changed spans/types. A good matcher drops the span/type-error items and
keeps the usable ones.

An item counts as "relinked" when it is gone but the same sentence and subject
now pair with an object whose span overlaps the old one (கோயம்புத்தூர் ->
கோயம்புத்தூர் மாவட்டம்).

--split tune|held scores half the pilot, split by article, so rules can be
tuned on one half and checked on the other.

    python scripts/eval_matcher_on_pilot.py data/processed/ds_candidates_v1.jsonl [--split tune]
"""
import argparse
import csv
import hashlib
import json
from collections import Counter

base = "data/annotation/pilot_v0/"
ap = argparse.ArgumentParser()
ap.add_argument("cands")
ap.add_argument("--adjudicated", default=base + "pilot_adjudicated_llm.csv")
ap.add_argument("--split", choices=["all", "tune", "held"], default="all")
ap.add_argument("--list", action="store_true", help="print every usable item dropped and error item kept")
a = ap.parse_args()
cand_path, adj_path = a.cands, a.adjudicated


def half(doc):
    return "tune" if hashlib.md5(doc.encode()).digest()[0] % 2 == 0 else "held"

adj = {r["item_id"]: r for r in csv.DictReader(open(adj_path, encoding="utf-8-sig"))}
key = [json.loads(l) for l in open(base + "pilot_key.jsonl", encoding="utf-8")]
wanted = {}
for k in key:
    c = k["candidate"]
    if a.split != "all" and half(c["meta"]["doc"]) != a.split:
        continue
    wanted[(c["source_id"], c["meta"]["wd_subject"], c["meta"]["wd_object"])] = k

found = {}
by_subj = {}
for l in open(cand_path, encoding="utf-8"):
    d = json.loads(l)
    kk = (d["source_id"], d["meta"]["wd_subject"], d["meta"]["wd_object"])
    if kk in wanted:
        found[kk] = d
    by_subj.setdefault(kk[:2], []).append(d)

res = Counter()
examples = {"good_dropped": [], "bad_kept": []}
for kk, k in wanted.items():
    r = adj[k["item_id"]]
    good = bool(r["label"].strip())
    old = k["candidate"]
    new = found.get(kk)
    if new is None:
        o = old["entities"][1]
        out = "relinked" if any(n["entities"][1]["start"] < o["end"] and o["start"] < n["entities"][1]["end"]
                                for n in by_subj.get(kk[:2], [])) else "dropped"
    else:
        same = [(e["start"], e["end"], e["type"]) for e in old["entities"]] == \
               [(e["start"], e["end"], e["type"]) for e in new["entities"]]
        out = "kept_same" if same else "kept_changed"
    res[(good, out)] += 1
    if good and out == "dropped":
        examples["good_dropped"].append(k["item_id"])
    if not good and out == "kept_same":
        examples["bad_kept"].append(k["item_id"])
    if a.list and ((good and out == "dropped") or (not good and out == "kept_same")):
        print("USABLE DROPPED" if good else "ERROR KEPT    ", k["item_id"], r["sentence_marked"][:120])

n_good = sum(v for (g, _), v in res.items() if g); n_bad = sum(v for (g, _), v in res.items() if not g)
print(f"{'':28}{'usable':>10}{'span/type err':>15}")
print(f"split: {a.split}")
for out in ("kept_same", "kept_changed", "relinked", "dropped"):
    print(f"{out:28}{res[(True, out)]:>10}{res[(False, out)]:>15}")
print(f"{'total':28}{n_good:>10}{n_bad:>15}")
kept = (n_good - res[(True, "dropped")]) / n_good
fixed = (n_bad - res[(False, "kept_same")]) / n_bad
print(f"usable kept (incl. changed/relinked): {kept:.3f}; errors removed, changed or relinked: {fixed:.3f}")
print("usable items dropped:", examples["good_dropped"][:15])
print("error items kept unchanged:", examples["bad_kept"][:15])
