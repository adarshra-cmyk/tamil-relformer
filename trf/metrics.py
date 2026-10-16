"""Evaluation (plan section 8).

Primary: micro-F1 over non-NO_RELATION labels (TACRED convention: a prediction
counts toward precision only when it is not NO_RELATION, and toward recall
only when the gold label is not NO_RELATION).
Also: macro-F1 over relation labels, per-relation P/R/F1, confusion matrix.
"""
from __future__ import annotations

from collections import Counter

from .schema import NO_RELATION


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(gold: list[str], pred: list[str], labels: list[str] | None = None) -> dict:
    assert len(gold) == len(pred)
    labels = labels or sorted(set(gold) | set(pred))
    rels = [l for l in labels if l != NO_RELATION]
    tp_all = sum(g == p != NO_RELATION for g, p in zip(gold, pred))
    n_pred = sum(p != NO_RELATION for p in pred)
    n_gold = sum(g != NO_RELATION for g in gold)
    mp, mr, mf = _prf(tp_all, n_pred, n_gold)

    per = {}
    for l in rels:
        tp = sum(g == p == l for g, p in zip(gold, pred))
        p_, r_, f_ = _prf(tp, sum(p == l for p in pred), sum(g == l for g in gold))
        per[l] = {"p": p_, "r": r_, "f1": f_, "support": sum(g == l for g in gold)}
    present = [l for l in rels if per[l]["support"] > 0]
    macro = sum(per[l]["f1"] for l in present) / len(present) if present else 0.0

    conf = Counter(zip(gold, pred))
    return {
        "micro_p": mp, "micro_r": mr, "micro_f1": mf,
        "macro_f1": macro,
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0,
        "per_relation": per,
        "confusion": {g: {p: conf[(g, p)] for p in labels if conf[(g, p)]} for g in labels},
        "n": len(gold),
    }
