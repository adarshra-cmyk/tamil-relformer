"""Score a morphology analyzer on data/probe/case_probe.jsonl (lemma + case of one target word).

This is a sanity probe of textbook forms, not a corpus coverage estimate.
Usage: python scripts/score_probe.py [--analyzer stanza]
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trf.morph import StanzaAnalyzer, ThamizhiMorphAnalyzer, entity_features
from trf.schema import Entity

ap = argparse.ArgumentParser()
ap.add_argument("--analyzer", default="stanza")
ap.add_argument("--probe", default="data/probe/case_probe.jsonl")
args = ap.parse_args()
an = {"stanza": StanzaAnalyzer, "thamizhi": ThamizhiMorphAnalyzer}[args.analyzer]()
rows = [json.loads(l) for l in open(args.probe, encoding="utf-8") if l.strip()]
ok_l = ok_c = 0
by_case = Counter(); ok_by_case = Counter()
for r in rows:
    s = r["sentence"].find(r["target"])
    f = entity_features(an.analyze(r["sentence"]), Entity("T", r["target"], "MISC", s, s + len(r["target"])))
    lc, cc = f.lemma == r["lemma"], f.case == r["case"]
    ok_l += lc; ok_c += cc; by_case[r["case"]] += 1; ok_by_case[r["case"]] += cc
    flag = "" if (lc and cc) else "  <-- " + ("lemma " if not lc else "") + ("case" if not cc else "")
    print(f"{r['target']:22} gold={r['lemma']}/{r['case']:4} pred={f.lemma}/{f.case} [{f.status}]{flag}")
n = len(rows)
print(f"\n{an.name}: lemma acc {ok_l}/{n} = {ok_l/n:.2f}; case acc {ok_c}/{n} = {ok_c/n:.2f}")
print("case acc by gold case:", {c: f"{ok_by_case[c]}/{by_case[c]}" for c in sorted(by_case)})
