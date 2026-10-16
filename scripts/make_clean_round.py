"""Sample a labelled round with fixed train/dev/test splits (clean_v1).

Articles are assigned to one split by hash, so no article is in two splits, and
pilot articles are left out. Within each split, items are spread over DS labels
(at most one item per article per label). Writes, under --out-dir:
    key.jsonl                       item_id, split, double, ds_label, candidate
    llm/batches/pass1_bNN.jsonl     annotator view (DS label hidden), --batch-size items each
    llm/batches/pass2_bNN.jsonl     second independent pass on the double items

    python scripts/make_clean_round.py --candidates runs/matcher_cmp/cands_v2.jsonl
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_io import mark

NO_RELATION = "NO_RELATION"


def split_of(doc: str, ratios: dict[str, float]) -> str:
    x = int(hashlib.md5(("clean_v1:" + doc).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, r in ratios.items():
        acc += r
        if x < acc:
            return name
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="runs/matcher_cmp/cands_v2.jsonl")
    ap.add_argument("--pilot-key", default="data/annotation/pilot_v0/pilot_key.jsonl")
    ap.add_argument("--out-dir", default="data/annotation/clean_v1")
    ap.add_argument("--sizes", nargs=3, type=int, default=[2000, 400, 1200], metavar=("TRAIN", "DEV", "TEST"))
    ap.add_argument("--neg-frac", type=float, default=0.35, help="share of each split with DS label NO_RELATION")
    ap.add_argument("--double-frac", type=float, default=0.15, help="of dev and test items")
    ap.add_argument("--batch-size", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--exclude-key", nargs="*", default=[],
                    help="key files of earlier rounds; their items are not sampled again")
    ap.add_argument("--span-filter-round", help="annotated round to train scripts/span_filter.py on; "
                    "candidates it rates p > --span-filter-p are skipped (use for training rounds only)")
    ap.add_argument("--span-filter-p", type=float, default=0.5)
    ap.add_argument("--prefix", default="", help="item id prefix, e.g. 'v2' gives v2tr0001")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    sizes = dict(zip(["train", "dev", "test"], a.sizes))
    ratios = {"train": 0.6, "dev": 0.1, "test": 0.3}

    pilot_docs = {json.loads(l)["candidate"]["meta"]["doc"] for l in open(a.pilot_key, encoding="utf-8")}
    K = lambda d: (d["source_id"], d["meta"]["wd_subject"], d["meta"]["wd_object"])
    seen = {K(json.loads(l)["candidate"]) for p in a.exclude_key for l in open(p, encoding="utf-8")}
    flt = None
    if a.span_filter_round:
        import glob as _glob
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from span_filter import feats
        rk = list(map(json.loads, open(f"{a.span_filter_round}/key.jsonl", encoding="utf-8")))
        ann = {}
        for f in _glob.glob(f"{a.span_filter_round}/llm/out_pass1_b*.jsonl"):
            for d in map(json.loads, open(f, encoding="utf-8")):
                ann[d["item_id"]] = d
        vec = DictVectorizer()
        X = vec.fit_transform([feats(k["candidate"]) for k in rk])
        y = [not (ann[k["item_id"]]["span_ok"] == "yes" and ann[k["item_id"]]["type_ok"] == "yes") for k in rk]
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(X, y)
        flt = lambda d: clf.predict_proba(vec.transform([feats(d)]))[0, 1] > a.span_filter_p
    skipped = Counter()
    pools = defaultdict(lambda: defaultdict(list))        # split -> label -> candidates
    for l in open(a.candidates, encoding="utf-8"):
        d = json.loads(l)
        doc = d["meta"]["doc"]
        if doc in pilot_docs:
            continue
        sp = split_of(doc, ratios)
        if sizes.get(sp, 0) == 0:
            continue
        if K(d) in seen:
            skipped["in earlier round"] += 1; continue
        if flt is not None and flt(d):
            skipped["span filter"] += 1; continue
        pools[sp][d["relation"]["label"]].append(d)

    key = []
    if skipped:
        print("skipped:", dict(skipped))
    for split, n in sizes.items():
        if n == 0:
            continue
        by_label = pools[split]
        pos_labels = sorted(l for l in by_label if l != NO_RELATION)
        quota = {NO_RELATION: round(n * a.neg_frac)}
        per_pos = (n - quota[NO_RELATION]) // len(pos_labels)
        # small labels give their unused quota to the others
        short = {l: len(by_label[l]) for l in pos_labels if len(by_label[l]) < per_pos}
        rest = [l for l in pos_labels if l not in short]
        extra = (n - quota[NO_RELATION] - sum(short.values())) // max(1, len(rest))
        quota.update(short); quota.update({l: extra for l in rest})
        picked = []
        for label, k in sorted(quota.items()):
            items = by_label[label][:]
            rng.shuffle(items)
            seen, take = set(), []
            for d in items:
                if d["meta"]["doc"] in seen:
                    continue
                seen.add(d["meta"]["doc"]); take.append(d)
                if len(take) == k:
                    break
            picked += take
        rng.shuffle(picked)
        n_double = round(a.double_frac * len(picked)) if split != "train" else 0
        for i, d in enumerate(picked):
            key.append({"item_id": f"{a.prefix}{split[:2]}{i:04d}", "split": split, "double": i < n_double,
                        "ds_label": d["relation"]["label"], "candidate": d})
        print(f"{split}: {len(picked)} items, {n_double} double; "
              f"{dict(Counter(d['relation']['label'] for d in picked).most_common())}")

    out = Path(a.out_dir); (out / "llm/batches").mkdir(parents=True, exist_ok=True)
    with open(out / "key.jsonl", "w", encoding="utf-8") as f:
        for k in key:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    def view(k):
        d = k["candidate"]
        subj = next(e for e in d["entities"] if e["id"] == d["relation"]["subject"])
        obj = next(e for e in d["entities"] if e["id"] == d["relation"]["object"])
        return {"item_id": k["item_id"], "sentence_marked": mark(d),
                "subj_type": subj["type"], "obj_type": obj["type"]}

    order = key[:]
    rng.shuffle(order)                                    # mix splits so batches are alike
    for name, items in (("pass1", order), ("pass2", [k for k in order if k["double"]])):
        for b in range(0, len(items), a.batch_size):
            with open(out / f"llm/batches/{name}_b{b // a.batch_size:02d}.jsonl", "w", encoding="utf-8") as f:
                for k in items[b:b + a.batch_size]:
                    f.write(json.dumps(view(k), ensure_ascii=False) + "\n")
    print(f"{len(key)} items -> {out}; batches: "
          f"{len(list((out / 'llm/batches').glob('pass1_*')))} pass1, {len(list((out / 'llm/batches').glob('pass2_*')))} pass2")


if __name__ == "__main__":
    main()
