"""Train/dev/test split grouped by source document (plan 5.3: avoid near-duplicate leakage).

    python scripts/make_splits.py data/processed/annotated.jsonl --ratios 0.7 0.1 0.2

Also writes predicted morphology (meta.morph_pred) with the chosen analyzer so
training does not need the analyzer at run time.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trf.morph import StanzaAnalyzer, ThamizhiMorphAnalyzer, example_features
from trf.schema import load_jsonl, save_jsonl


def doc_of(ex):
    return ex.meta.get("doc") or (ex.source_id or "").split("#")[0] or ex.sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.1, 0.2])
    ap.add_argument("--analyzer", choices=["stanza", "thamizhi", "none"], default="stanza")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    exs = load_jsonl(a.input)
    if a.analyzer != "none":
        an = StanzaAnalyzer() if a.analyzer == "stanza" else ThamizhiMorphAnalyzer()
        for ex in exs:
            ex.meta["morph_pred"] = {k: v.to_dict() for k, v in example_features(ex, an).items()}
            ex.meta["morph_analyzer"] = an.name

    docs = defaultdict(list)
    for ex in exs:
        docs[doc_of(ex)].append(ex)
    keys = sorted(docs); random.Random(a.seed).shuffle(keys)
    n = len(exs); bounds = [a.ratios[0] * n, (a.ratios[0] + a.ratios[1]) * n]
    splits = {"train": [], "dev": [], "test": []}
    count = 0
    for k in keys:
        name = "train" if count < bounds[0] else "dev" if count < bounds[1] else "test"
        splits[name] += docs[k]; count += len(docs[k])
    for name, xs in splits.items():
        save_jsonl(xs, Path(a.out_dir) / f"{name}.jsonl")
        print(f"{name:5} {len(xs):6} examples, {len({doc_of(x) for x in xs}):5} docs, "
              f"labels {dict(Counter(x.label for x in xs).most_common(5))}")


if __name__ == "__main__":
    main()
