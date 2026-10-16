"""Entity lemma and case from the distant-supervision match instead of Stanza.

The matcher found the span by matching one of the entity's Wikidata names (in
the span's own spelling) plus a case suffix, so that name is the lemma and the
suffix gives the case (no suffix = Nom). Mentions without such a match keep
Stanza's values.

Writes <split>_matchlemma.jsonl (lemma replaced, case stays Stanza's) and
<split>_matchmorph.jsonl (lemma and case replaced) next to each split, and
prints accuracy against the LLM-checked values when they are available.

    python scripts/matcher_lemmas.py data/processed/clean_v1
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ds_candidates import Gazetteer, WORD_RE, stems, strip_title

norm = lambda s: re.sub(r"\s+", " ", (s or "").strip())
# suffix left after the name's stem (and a y/v glide) -> case; longest first
SUFFIX_CASE = [("ிலிருந்து", "Abl"), ("இலிருந்து", "Abl"), ("ினுடைய", "Gen"), ("ுடைய", "Gen"),
               ("ிற்கு", "Dat"), ("ுக்கு", "Dat"), ("க்கு", "Dat"), ("ினால்", "Ins"), ("ால்", "Ins"),
               ("ுடன்", "Com"), ("ோடு", "Com"), ("ிடம்", "Loc"), ("ினது", "Gen"), ("ின்", "Gen"),
               ("ில்", "Loc"), ("ைய", "Acc"), ("ை", "Acc"), ("ில", "Loc"), ("ின", "Gen"), ("ுக்க", "Dat")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--facts", default="data/processed/wikidata_facts.jsonl")
    ap.add_argument("--gold", default="data/annotation/gold_morph_v1/out_*.jsonl")
    a = ap.parse_args()

    names = defaultdict(set)
    for f in map(json.loads, open(a.facts, encoding="utf-8")):
        names[f["s"]].update({f["s_label"], strip_title(f["s_title"])})
        names[f["o"]].update({f["o_label"], *f["o_aliases"]})
    gold = {}
    for fn in glob.glob(a.gold):
        for d in map(json.loads, open(fn, encoding="utf-8")):
            gold[d["item_id"]] = d

    def matcher_lemma(span, q):
        """Longest name of q that matches the whole span (name + case suffix)."""
        words = [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(span)]
        end = len(span.rstrip(" .,"))
        best = None
        for n in names[q]:
            g = Gazetteer(); g.add(n, q)
            for s, e, _, _ in g.find(words, span):
                if s == 0 and e >= end:
                    cand = strip_title(n)
                    if best is None or len(cand) > len(best):
                        best = cand
        return best

    def suffix_case(span, q):
        """Case from what follows the longest matching name stem in the span's last word."""
        ws = WORD_RE.findall(span)
        if not ws:
            return None
        last, best = ws[-1], None
        for n in names[q]:
            nw = WORD_RE.findall(strip_title(n))
            for st in (stems(nw[-1]) if nw else []):
                if last.startswith(st) and (best is None or len(st) > len(best)):
                    best = st
        if best is None:
            return None
        suf = last[len(best):].lstrip("யவ")
        if not suf:
            return "Nom"
        return next((c for s, c in SUFFIX_CASE if suf.startswith(s)), "OTHER")

    acc, tot, src = Counter(), Counter(), Counter()
    for split in ("train", "dev", "test"):
        p = Path(a.dir) / f"{split}.jsonl"
        out_l, out_m = [], []
        for x in map(json.loads, open(p, encoding="utf-8")):
            xm = json.loads(json.dumps(x))
            for role, q, i, E in (("subj", x["meta"]["wd_subject"], 0, "E1"), ("obj", x["meta"]["wd_object"], 1, "E2")):
                span = x["entities"][i]["text"]
                mp = x["meta"]["morph_pred"][role]
                ml, mc = matcher_lemma(span, q), suffix_case(span, q)
                stanza_l, stanza_c = mp.get("lemma"), mp.get("case")
                for d in (x, xm):
                    d["meta"]["morph_pred"][role]["lemma"] = ml or stanza_l
                    d["meta"].setdefault("lemma_source", {})[role] = "matcher" if ml else "stanza"
                xm["meta"]["morph_pred"][role]["case"] = mc or stanza_c
                xm["meta"].setdefault("case_source", {})[role] = "suffix" if mc else "stanza"
                src["lemma:" + ("matcher" if ml else "stanza")] += 1
                src["case:" + ("suffix" if mc else "stanza")] += 1
                g = gold.get(x["meta"]["item_id"])
                if g:
                    for k in ("all", x["meta"]["match"][i]):
                        tot[k] += 1
                        acc[(k, "stanza_l")] += norm(stanza_l) == norm(g[E]["lemma"])
                        acc[(k, "new_l")] += norm(ml or stanza_l) == norm(g[E]["lemma"])
                        acc[(k, "stanza_c")] += stanza_c == g[E]["case"]
                        acc[(k, "new_c")] += (mc or stanza_c) == g[E]["case"]
            out_l.append(x); out_m.append(xm)
        for suffix, rows in (("matchlemma", out_l), ("matchmorph", out_m)):
            with open(Path(a.dir) / f"{split}_{suffix}.jsonl", "w", encoding="utf-8") as f:
                for x in rows:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("sources:", dict(src))
    for k in ("all", "inflected", "exact"):
        if tot[k]:
            t = tot[k]
            print(f"{k:10} n={t:5}  lemma Stanza {acc[(k, 'stanza_l')] / t:.3f} -> {acc[(k, 'new_l')] / t:.3f}   "
                  f"case Stanza {acc[(k, 'stanza_c')] / t:.3f} -> {acc[(k, 'new_c')] / t:.3f}")


if __name__ == "__main__":
    main()
