"""RQ1: entity lemma/case accuracy of Stanza, ThamizhiMorph and the matcher-alias
lemma against the LLM-checked morphology of all clean_v1 items.

Writes docs/analyzer_scores.json (read by make_report.py).

    python scripts/score_analyzers.py
"""
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trf.morph import ThamizhiMorphAnalyzer, example_features
from trf.schema import parse_example

norm = lambda s: re.sub(r"\s+", " ", (s or "").strip())


def main():
    gold = {}
    for fn in glob.glob(str(ROOT / "data/annotation/gold_morph_v1/out_*.jsonl")):
        for d in map(json.loads, open(fn, encoding="utf-8")):
            gold[d["item_id"]] = d
    th = ThamizhiMorphAnalyzer(ROOT / "data/raw/thamizhi-morph/FST-Models")
    acc, tot = Counter(), Counter()
    for sp in ("train", "dev", "test"):
        plain = [json.loads(l) for l in open(ROOT / f"data/processed/clean_v1/{sp}.jsonl", encoding="utf-8")]
        ml = [json.loads(l) for l in open(ROOT / f"data/processed/clean_v1/{sp}_matchlemma.jsonl", encoding="utf-8")]
        for x, xm in zip(plain, ml):
            g = gold[x["meta"]["item_id"]]
            tf = example_features(parse_example(x), th)
            for role, E, i in (("subj", "E1", 0), ("obj", "E2", 1)):
                gl, gc = norm(g[E]["lemma"]), g[E]["case"]
                st = x["meta"]["morph_pred"][role]
                pred = {"Stanza": (st.get("lemma"), st.get("case")),
                        "ThamizhiMorph": (tf[role].lemma, tf[role].case),
                        "matcher alias, else Stanza": (xm["meta"]["morph_pred"][role].get("lemma"), None)}
                for k in ("all", x["meta"]["match"][i]):
                    tot[k] += 1
                    for name, (l, c) in pred.items():
                        acc[(k, name, "lemma")] += norm(l) == gl
                        if c is not None:
                            acc[(k, name, "case")] += c == gc
    rows = []
    for name in ("Stanza", "ThamizhiMorph", "matcher alias, else Stanza"):
        for k in ("all", "inflected", "exact"):
            rows.append({"analyzer": name, "mentions": k, "n": tot[k],
                         "lemma": round(acc[(k, name, "lemma")] / tot[k], 3),
                         "case": round(acc[(k, name, "case")] / tot[k], 3) if name != "matcher alias, else Stanza" else "-"})
    (ROOT / "docs/analyzer_scores.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
