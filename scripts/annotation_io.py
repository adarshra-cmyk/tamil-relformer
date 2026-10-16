"""Annotation round trip for the pilot.

  export: sample candidates into an annotator CSV (DS label hidden) plus a key file
      python scripts/annotation_io.py export --n-per-label 40 --n-neg 120 --double-frac 0.2
  import: read completed CSV(s) into schema JSONL with verified labels/morphology
      python scripts/annotation_io.py import ann_A.csv [ann_B.csv ...] --out data/processed/annotated.jsonl
  agree: Cohen's kappa on items both annotators labelled
      python scripts/annotation_io.py agree ann_A.csv ann_B.csv

The distant-supervision label is kept out of the annotator sheet to avoid
anchoring; it lives in the key file for later DS-precision analysis.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trf.schema import NO_RELATION

COLS = ["item_id", "sentence_marked", "subj_type", "obj_type", "label", "span_ok", "type_ok",
        "variety", "pred_subj_lemma", "pred_subj_case", "pred_obj_lemma", "pred_obj_case",
        "subj_lemma", "subj_case", "obj_lemma", "obj_case", "notes"]


def mark(d: dict) -> str:
    s = d["sentence"]
    ents = sorted(d["entities"], key=lambda e: e["start"])
    out, pos = [], 0
    for e in ents:
        tag = "E1" if e["id"] == d["relation"]["subject"] else "E2"
        out += [s[pos:e["start"]], f"[{tag}: {e['text']}]"]
        pos = e["end"]
    return "".join(out + [s[pos:]])


def export(a):
    rng = random.Random(a.seed)
    cands = [json.loads(l) for l in open(a.candidates, encoding="utf-8")]
    by_label = defaultdict(list)
    for d in cands:
        by_label[d["relation"]["label"]].append(d)
    # at most one item per document per label, to spread the pilot across articles
    picked = []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        seen_docs, take = set(), []
        k = a.n_neg if label == NO_RELATION else a.n_per_label
        for d in items:
            if d["meta"]["doc"] in seen_docs:
                continue
            seen_docs.add(d["meta"]["doc"]); take.append(d)
            if len(take) == k:
                break
        picked += take
    rng.shuffle(picked)

    analyzer = None
    if not a.no_morph:
        from trf.morph import StanzaAnalyzer, entity_features
        from trf.schema import Entity
        analyzer = StanzaAnalyzer()

    n_double = int(round(a.double_frac * len(picked)))
    double_ids = set(range(n_double))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sheets = {"A": [], "B": []}
    key = []
    for i, d in enumerate(picked):
        iid = f"p{i:04d}"
        row = {c: "" for c in COLS}
        subj = next(e for e in d["entities"] if e["id"] == d["relation"]["subject"])
        obj = next(e for e in d["entities"] if e["id"] == d["relation"]["object"])
        row.update(item_id=iid, sentence_marked=mark(d), subj_type=subj["type"], obj_type=obj["type"])
        if analyzer:
            toks = analyzer.analyze(d["sentence"])
            for role, e in (("subj", subj), ("obj", obj)):
                f = entity_features(toks, Entity(e["id"], e["text"], e["type"], e["start"], e["end"]))
                row[f"pred_{role}_lemma"], row[f"pred_{role}_case"] = f.lemma or "", f.case
        # split singles between A and B; doubles go to both
        if i in double_ids:
            sheets["A"].append(row); sheets["B"].append(dict(row))
        else:
            sheets["A" if i % 2 else "B"].append(row)
        key.append({"item_id": iid, "double": i in double_ids, "ds_label": d["relation"]["label"],
                    "candidate": d})
    for name, rows in sheets.items():
        rows.sort(key=lambda r: r["item_id"])
        with open(out / f"pilot_sheet_{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    with open(out / "pilot_key.jsonl", "w", encoding="utf-8") as f:
        for k in key:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"{len(picked)} items ({n_double} double-annotated); "
          f"sheet A {len(sheets['A'])} rows, sheet B {len(sheets['B'])} rows -> {out}")
    print("by DS label:", dict(Counter(k["ds_label"] for k in key)))


def read_sheet(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def cohen_kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agree(a):
    A, B = read_sheet(a.sheet_a), read_sheet(a.sheet_b)
    both = sorted(i for i in A if i in B and A[i]["label"] and B[i]["label"])
    if not both:
        sys.exit("no item labelled in both sheets")
    x = [A[i]["label"].strip() for i in both]; y = [B[i]["label"].strip() for i in both]
    print(f"items labelled by both: {len(both)}")
    print(f"raw agreement {sum(p == q for p, q in zip(x, y)) / len(both):.3f}; Cohen's kappa {cohen_kappa(x, y):.3f}")
    conf = Counter((p, q) for p, q in zip(x, y) if p != q)
    print("most common disagreements:")
    for (p, q), c in conf.most_common(10):
        print(f"  {c:3}  A={p}  B={q}")


def do_import(a):
    key = {k["item_id"]: k for k in map(json.loads, open(a.key, encoding="utf-8"))}
    adjud = read_sheet(a.adjudicated) if a.adjudicated else {}
    rows = {}
    for p in a.sheets:
        for iid, r in read_sheet(p).items():
            rows.setdefault(iid, []).append(r)
    out, skipped = [], Counter()
    for iid, rs in sorted(rows.items()):
        r = adjud.get(iid) or rs[0]
        if len(rs) > 1 and iid not in adjud and rs[0]["label"].strip() != rs[1]["label"].strip():
            skipped["disagreement_not_adjudicated"] += 1; continue
        if not r["label"].strip():
            skipped["unlabelled"] += 1; continue
        if r["span_ok"].strip().lower() == "no" or r["type_ok"].strip().lower() == "no":
            skipped["span_or_type_error"] += 1; continue
        d = json.loads(json.dumps(key[iid]["candidate"]))
        d["relation"]["label"] = r["label"].strip()
        d["variety"] = r["variety"].strip() or None
        morph = {}
        for role, eid in (("subj", d["relation"]["subject"]), ("obj", d["relation"]["object"])):
            lem = r[f"{role}_lemma"].strip() or r[f"pred_{role}_lemma"].strip()
            case = r[f"{role}_case"].strip() or r[f"pred_{role}_case"].strip()
            morph[eid] = {"lemma": lem, "case": case}
        if any(r[f"{x}_{y}"].strip() for x in ("subj", "obj") for y in ("lemma", "case")):
            d["meta"]["morph"] = morph          # only rows where the annotator checked morphology
        d["meta"].update(verified=True, item_id=iid, ds_label=key[iid]["ds_label"],
                         notes=r["notes"].strip(), annotator=a.annotator)
        out.append(d)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    ds_prec = sum(d["meta"]["ds_label"] == d["relation"]["label"] for d in out) / max(1, len(out))
    print(f"imported {len(out)}; skipped {dict(skipped)}; DS label agrees with annotator on {ds_prec:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--candidates", default="data/processed/ds_candidates.jsonl")
    e.add_argument("--out-dir", default="data/annotation/pilot_v0")
    e.add_argument("--n-per-label", type=int, default=40)
    e.add_argument("--n-neg", type=int, default=120)
    e.add_argument("--double-frac", type=float, default=0.2)
    e.add_argument("--no-morph", action="store_true")
    e.add_argument("--seed", type=int, default=13)
    g = sub.add_parser("agree"); g.add_argument("sheet_a"); g.add_argument("sheet_b")
    i = sub.add_parser("import"); i.add_argument("sheets", nargs="+")
    i.add_argument("--key", default="data/annotation/pilot_v0/pilot_key.jsonl")
    i.add_argument("--adjudicated", default=None)
    i.add_argument("--out", default="data/processed/annotated.jsonl")
    i.add_argument("--annotator", default="human", help="recorded in meta.annotator (e.g. human, llm)")
    a = ap.parse_args()
    {"export": export, "agree": agree, "import": do_import}[a.cmd](a)


if __name__ == "__main__":
    main()
