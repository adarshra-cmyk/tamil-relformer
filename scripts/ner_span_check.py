"""Check matched entity spans against IndicNER (ai4bharat/IndicNER) entity boundaries.

For every candidate in a round's key, runs NER on the sentence (CPU) and
records, per entity span, whether an NER entity overlaps it and whether that
entity extends beyond the span by at least one word on the left or right (a
sign that the span cuts a longer name). Writes <out> as JSONL keyed by
item_id; `--eval` then compares the signals with the annotators' span_ok.

    python scripts/ner_span_check.py data/annotation/clean_v1/key.jsonl --out runs/ner_spans_clean_v1.jsonl
    python scripts/ner_span_check.py data/annotation/clean_v1/key.jsonl --out runs/ner_spans_clean_v1.jsonl --eval
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[^\s.,;:!?\"'()\[\]{}<>“”‘’–—-]+")


def run_ner(key_path, out_path, batch=32, threads=4):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
    model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")
    ner = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple", device=-1)
    key = [json.loads(l) for l in open(key_path, encoding="utf-8")]
    done = set()
    if Path(out_path).exists():
        done = {json.loads(l)["item_id"] for l in open(out_path, encoding="utf-8")}
    todo = [k for k in key if k["item_id"] not in done]
    with open(out_path, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            res = ner([k["candidate"]["sentence"] for k in chunk])
            for k, ents in zip(chunk, res):
                s = k["candidate"]["sentence"]
                out = {"item_id": k["item_id"], "spans": {}}
                for e in k["candidate"]["entities"]:
                    a, z = e["start"], e["end"]
                    hits = [x for x in ents if x["start"] < z and a < x["end"]]
                    info = {"overlap": bool(hits), "ner_types": sorted({x["entity_group"] for x in hits}),
                            "extends_left": any(WORD.search(s[x["start"]:a] or "") for x in hits if x["start"] < a),
                            "extends_right": any(WORD.search(s[z:x["end"]] or "") for x in hits if x["end"] > z)}
                    out["spans"][e["id"]] = info
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"  {b + len(chunk)}/{len(todo)}", flush=True)


def evaluate(key_path, out_path, ann_glob):
    key = {k["item_id"]: k for k in map(json.loads, open(key_path, encoding="utf-8"))}
    ner = {d["item_id"]: d for d in map(json.loads, open(out_path, encoding="utf-8"))}
    ann = {}
    for f in glob.glob(ann_glob):
        for d in map(json.loads, open(f, encoding="utf-8")):
            ann[d["item_id"]] = d
    rules = {
        "any span extends (NER entity longer)": lambda sp: any(v["extends_left"] or v["extends_right"] for v in sp.values()),
        "any span has no NER entity": lambda sp: any(not v["overlap"] for v in sp.values()),
        "either signal": lambda sp: any(v["extends_left"] or v["extends_right"] or not v["overlap"] for v in sp.values()),
    }
    print(f"items with NER output and annotation: {sum(1 for i in ner if i in ann)}")
    for name, rule in rules.items():
        c = Counter()
        for i, d in ner.items():
            if i not in ann:
                continue
            bad = ann[i]["span_ok"] == "no"
            flag = rule(d["spans"])
            c[(flag, bad)] += 1
        tp, fp, fn = c[(True, True)], c[(True, False)], c[(False, True)]
        n_bad = tp + fn
        print(f"{name:42} flags {tp + fp:4}; catches {tp}/{n_bad} span errors ({tp / max(1, n_bad):.2f}); "
              f"{fp} flagged items were fine (precision {tp / max(1, tp + fp):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ann", default="data/annotation/clean_v1/llm/out_pass1_b*.jsonl")
    a = ap.parse_args()
    if a.eval:
        evaluate(a.key, a.out, a.ann)
    else:
        run_ner(a.key, a.out)


if __name__ == "__main__":
    main()
