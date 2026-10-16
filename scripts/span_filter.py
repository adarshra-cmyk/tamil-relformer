"""Predict span/type errors in DS candidates before annotation.

Logistic regression on cheap features of each matched pair: the words just
before and after each span, match type (exact/inflected), span length in
words, entity types, relation slot and DS label. Trained on one split of an
annotated round (span_ok/type_ok from pass 1), evaluated on another, so the
numbers say how many annotation slots a filter would save at what cost.

    python scripts/span_filter.py --round data/annotation/clean_v1 --train-split train --eval-split test
"""
import argparse
import glob
import json
import re

WORD = re.compile(r"[^\s.,;:!?\"'()\[\]{}<>“”‘’–—-]+")


def feats(c):
    s = c["sentence"]
    out = {f"ds={c['relation']['label']}": 1}
    for role, e, m in zip(("s", "o"), c["entities"], c["meta"]["match"]):
        prev = WORD.findall(s[:e["start"]])[-1:] or ["<BOS>"]
        nxt = WORD.findall(s[e["end"]:])[:1] or ["<EOS>"]
        words = WORD.findall(e["text"])
        out.update({f"{role}.prev={prev[0]}": 1, f"{role}.next={nxt[0]}": 1, f"{role}.type={e['type']}": 1,
                    f"{role}.match={m}": 1, f"{role}.nwords={min(len(words), 4)}": 1,
                    f"{role}.next_type={nxt[0]}|{e['type']}": 1, f"{role}.short": int(len(e["text"]) <= 4),
                    f"{role}.glued_next": int(s[e["end"]:e["end"] + 1].isalpha())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="data/annotation/clean_v1")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="test")
    a = ap.parse_args()
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    key = list(map(json.loads, open(f"{a.round}/key.jsonl", encoding="utf-8")))
    ann = {}
    for f in glob.glob(f"{a.round}/llm/out_pass1_b*.jsonl"):
        for d in map(json.loads, open(f, encoding="utf-8")):
            ann[d["item_id"]] = d
    bad = lambda k: not (ann[k["item_id"]]["span_ok"] == "yes" and ann[k["item_id"]]["type_ok"] == "yes")
    tr = [k for k in key if k["split"] == a.train_split]
    ev = [k for k in key if k["split"] == a.eval_split]
    vec = DictVectorizer()
    Xtr = vec.fit_transform([feats(k["candidate"]) for k in tr]); ytr = [bad(k) for k in tr]
    Xev = vec.transform([feats(k["candidate"]) for k in ev]); yev = [bad(k) for k in ev]
    clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    p = clf.predict_proba(Xev)[:, 1]
    n_bad = sum(yev)
    print(f"train {len(tr)} items ({sum(ytr)} errors); eval {len(ev)} items ({n_bad} errors, rate {n_bad / len(ev):.3f})")
    print(f"{'drop if p >':>12} {'dropped':>8} {'errors caught':>14} {'good lost':>10} {'error rate after':>17}")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        drop = [pi > th for pi in p]
        caught = sum(d and y for d, y in zip(drop, yev)); lost = sum(d and not y for d, y in zip(drop, yev))
        kept = len(ev) - sum(drop)
        print(f"{th:12.1f} {sum(drop):8} {caught:8} ({caught / n_bad:.2f}) {lost:10} {(n_bad - caught) / max(1, kept):17.3f}")
    names = vec.get_feature_names_out()
    top = sorted(zip(clf.coef_[0], names), reverse=True)[:15]
    print("strongest error signals:", ", ".join(f"{n} ({w:+.2f})" for w, n in top))


if __name__ == "__main__":
    main()
