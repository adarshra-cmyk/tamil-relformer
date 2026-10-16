"""Merge LLM annotations of a clean round into train/dev/test (provisional, not human gold).

  agree:  pass1 vs pass2 on the double items (label kappa, span/type flags), and
          write llm/batches/recon_bNN.jsonl for every double item where they differ
  import: final judgement per item = recon if present, else pass1 (double items
          that differ and have no recon are left out); writes
          <out-dir>/{train,dev,test}.jsonl with Stanza morphology in meta.morph_pred

    python scripts/import_clean_round.py agree
    python scripts/import_clean_round.py import
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_io import cohen_kappa, mark

FLAGS = ("span_ok", "type_ok")


def read(glob, base):
    out = {}
    for f in sorted((base / "llm").glob(glob)):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); out[d["item_id"]] = d
    return out


def norm(d):
    ok = all(d[f].strip().lower() == "yes" for f in FLAGS)
    return (d["label"].strip() if ok else "", ok)


def agree(a, base, key):
    p1, p2 = read("out_pass1_b*.jsonl", base), read("out_pass2_b*.jsonl", base)
    both = [i for i in p2 if i in p1]
    print(f"pass1 {len(p1)}/{len(key)} items, pass2 {len(p2)}, both {len(both)}")
    flag_agree = sum(norm(p1[i])[1] == norm(p2[i])[1] for i in both)
    lab = [i for i in both if norm(p1[i])[1] and norm(p2[i])[1]]
    x = [p1[i]["label"].strip() for i in lab]; y = [p2[i]["label"].strip() for i in lab]
    print(f"span/type flag agree {flag_agree}/{len(both)}")
    if lab:
        print(f"labels on {len(lab)} items both found usable: raw {sum(p == q for p, q in zip(x, y)) / len(lab):.3f}, "
              f"kappa {cohen_kappa(x, y):.3f}")
    for (p, q), c in Counter((p, q) for p, q in zip(x, y) if p != q).most_common(8):
        print(f"  {c:3}  pass1={p}  pass2={q}")
    diff = [i for i in both if norm(p1[i]) != norm(p2[i])]
    out = base / "llm/batches"
    for f in out.glob("recon_b*.jsonl"):
        f.unlink()
    for b in range(0, len(diff), a.batch_size):
        with open(out / f"recon_b{b // a.batch_size:02d}.jsonl", "w", encoding="utf-8") as f:
            for i in diff[b:b + a.batch_size]:
                d = key[i]["candidate"]
                f.write(json.dumps({"item_id": i, "sentence_marked": mark(d),
                                    "subj_type": d["entities"][0]["type"], "obj_type": d["entities"][1]["type"],
                                    "annotator_1": {k: p1[i][k] for k in ("label", *FLAGS, "notes")},
                                    "annotator_2": {k: p2[i][k] for k in ("label", *FLAGS, "notes")}},
                                   ensure_ascii=False) + "\n")
    print(f"{len(diff)} items differ -> recon batches")


def do_import(a, base, key):
    from trf.morph import StanzaAnalyzer, example_features
    from trf.schema import parse_example
    p1, p2, rc = read("out_pass1_b*.jsonl", base), read("out_pass2_b*.jsonl", base), read("out_recon_b*.jsonl", base)
    an = StanzaAnalyzer()
    splits, skipped = {"train": [], "dev": [], "test": []}, Counter()
    for iid, k in key.items():
        if iid in rc:
            r = rc[iid]
        elif iid in p1:
            if iid in p2 and norm(p1[iid]) != norm(p2[iid]):
                skipped["double_differs_no_recon"] += 1; continue
            r = p1[iid]
        else:
            skipped["not_annotated"] += 1; continue
        label, ok = norm(r)
        if not ok:
            skipped["span_or_type_error"] += 1; continue
        if not label:
            skipped["unlabelled"] += 1; continue
        d = json.loads(json.dumps(k["candidate"]))
        d["relation"]["label"] = label
        d["variety"] = (r.get("variety") or "").strip() or None
        d["meta"].update(verified=True, item_id=iid, ds_label=k["ds_label"], notes=r.get("notes", ""),
                         annotator="llm", round=base.name, decided_by="recon" if iid in rc else "pass1")
        d["meta"]["morph_pred"] = {e: f.to_dict() for e, f in example_features(parse_example(d), an).items()}
        d["meta"]["morph_analyzer"] = an.name
        splits[k["split"]].append(d)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, xs in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in xs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        ds = sum(d["meta"]["ds_label"] == d["relation"]["label"] for d in xs) / max(1, len(xs))
        print(f"{name:5} {len(xs):5} items; DS label right {ds:.3f}; "
              f"{dict(Counter(d['relation']['label'] for d in xs).most_common(6))}")
    print("skipped:", dict(skipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["agree", "import"])
    ap.add_argument("--round-dir", default="data/annotation/clean_v1")
    ap.add_argument("--out-dir", default="data/processed/clean_v1")
    ap.add_argument("--batch-size", type=int, default=100)
    a = ap.parse_args()
    base = Path(a.round_dir)
    key = {k["item_id"]: k for k in map(json.loads, open(base / "key.jsonl", encoding="utf-8"))}
    {"agree": agree, "import": do_import}[a.cmd](a, base, key)


if __name__ == "__main__":
    main()
