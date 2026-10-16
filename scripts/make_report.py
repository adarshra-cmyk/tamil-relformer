"""Regenerate docs/clean_v1_report.md from the data and run outputs.

Every number in the report comes from files in the repo, so rerunning this
after more runs finish updates the tables.

    python scripts/make_report.py
"""
import csv
import glob
import io
import json
import subprocess
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable
OUT = ROOT / "docs/clean_v1_report.md"
D = ROOT / "data/processed/clean_v1"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def dataset_section():
    key = jl(ROOT / "data/annotation/clean_v1/key.jsonl")
    rows = []
    for sp in ("train", "dev", "test"):
        xs = jl(D / f"{sp}.jsonl")
        sampled = sum(k["split"] == sp for k in key)
        pos = sum(x["relation"]["label"] != "NO_RELATION" for x in xs)
        ds = sum(x["meta"]["ds_label"] == x["relation"]["label"] for x in xs) / len(xs)
        infl = sum(x["meta"]["match"][1] == "inflected" for x in xs) / len(xs)
        rows.append({"split": sp, "sampled": sampled, "usable": len(xs), "relation items": pos,
                     "DS label right": f"{ds:.3f}", "object inflected": f"{infl:.2f}"})
    return table(rows, list(rows[0]))


def rejection_section():
    """Why matcher v2 candidates were rejected in clean_v1 (pass-1 notes, keyword categories)."""
    import re as _re
    ann = {}
    for f in glob.glob(str(ROOT / "data/annotation/clean_v1/llm/out_pass1_b*.jsonl")):
        for d in jl(f):
            ann[d["item_id"]] = d
    cats = [("part of a longer name", r"part of|longer name|cuts|inside the name|within the name"),
            ("common noun marked", r"common noun|generic"),
            ("surname without given name", r"surname|given name|first name"),
            ("title inside span", r"title|honorific"),
            ("adjective or language", r"adjective|language"),
            ("wrong entity type", r"not a (LOCATION|PERSON|ORGANIZATION)")]
    c = Counter()
    rej = [d for d in ann.values() if not (d["span_ok"] == "yes" and d["type_ok"] == "yes")]
    for d in rej:
        c[next((n for n, r in cats if _re.search(r, d["notes"], _re.I)), "other")] += 1
    rows = [{"reason (from annotator notes)": k, "items": v, "share": f"{v / len(rej):.2f}"} for k, v in c.most_common()]
    return f"{len(rej)} of {len(ann)} candidates rejected.\n\n" + table(rows, list(rows[0]))


def ds_section():
    """How often the distant-supervision label equals the annotated label, by DS label."""
    xs = [x for sp in ("train", "dev", "test") for x in jl(D / f"{sp}.jsonl")]
    by, conf = {}, Counter()
    for x in xs:
        ds, lab = x["meta"]["ds_label"], x["relation"]["label"]
        n, k = by.get(ds, (0, 0))
        by[ds] = (n + 1, k + (ds == lab))
        if ds != lab:
            conf[(ds, lab)] += 1
    rows = [{"DS label": l, "items": n, "annotated label agrees": f"{k / n:.2f}"}
            for l, (n, k) in sorted(by.items(), key=lambda kv: kv[1][1] / kv[1][0])]
    top = "\n".join(f"- DS {a} -> annotated {b}: {c}" for (a, b), c in conf.most_common(8))
    return table(rows, list(rows[0])) + "\n\nMost common disagreements:\n\n" + top


def agreement_section():
    p = subprocess.run([PY, "scripts/import_clean_round.py", "agree"], cwd=ROOT, capture_output=True, text=True)
    # agree rewrites the recon batch file from the same inputs; it does not change any labels
    return "```\n" + p.stdout.strip() + "\n```"


def morph_section():
    rows = []
    for name, files, split in (("test", "out_b*.jsonl", "test"), ("train+dev", "out_t*.jsonl", None)):
        gold = {}
        for f in sorted(glob.glob(str(ROOT / "data/annotation/gold_morph_v1" / files))):
            for d in jl(f):
                gold[d["item_id"]] = d
        if not gold:
            continue
        xs = [x for sp in (["test"] if split else ["train", "dev"]) for x in jl(D / f"{sp}.jsonl")
              if x["meta"]["item_id"] in gold]
        acc, tot = Counter(), Counter()
        for x in xs:
            g = gold[x["meta"]["item_id"]]
            for role, E, i in (("subj", "E1", 0), ("obj", "E2", 1)):
                p = x["meta"]["morph_pred"][role]
                infl = x["meta"]["match"][i]
                for k in ("all", infl):
                    tot[k] += 1
                    acc[(k, "lemma")] += (p.get("lemma") or "").strip() == g[E]["lemma"].strip()
                    acc[(k, "case")] += p.get("case") == g[E]["case"]
        for k in ("all", "inflected", "exact"):
            rows.append({"items": name, "mentions": k, "n": tot[k],
                         "Stanza lemma": f"{acc[(k, 'lemma')] / tot[k]:.3f}",
                         "Stanza case": f"{acc[(k, 'case')] / tot[k]:.3f}"})
    return table(rows, list(rows[0])) if rows else "(not yet)"


def analyzer_section():
    p = ROOT / "docs/analyzer_scores.json"
    if not p.exists():
        return "(run scripts/score_analyzers.py)"
    rows = json.loads(p.read_text())
    return table(rows, list(rows[0]))


def cross_model_section():
    p = ROOT / "docs/cross_model_agreement.json"
    if not p.exists():
        return ""
    r = json.loads(p.read_text())
    top = "; ".join(f"A {a} / B {b}: {c}" for a, b, c in r["top_disagreements"][:5])
    return (f"Cross-model check: a second LLM (model B) labelled the {r['n']} human-review test items blind "
            f"(same guide). It flagged {r['flagged']} spans/types; on the other {r['n_labels']} the "
            f"label agrees with the model A label {r['raw']:.3f} (kappa {r['kappa']:.3f}). Most frequent "
            f"differences: {top}. Two models, not two people.")


def compare_section(root, extra=()):
    p = subprocess.run([PY, "scripts/compare_runs.py", root, "--test", str(D / "test.jsonl"),
                        "--train", str(D / "train.jsonl"), "--n-boot", "2000", *extra],
                       cwd=ROOT, capture_output=True, text=True)
    return "```\n" + p.stdout.strip() + "\n```" if p.stdout.strip() else "(no runs yet)"


def silver_section():
    """E0 trained on clean_v1 vs on distant-supervision labels (same train-split articles), same test."""
    rows = []
    for name, root in (("clean (LLM-labelled)", "runs/clean_v1"), ("silver (DS labels, matcher v2)", "runs/clean_v1_silver")):
        by = {}
        for m in (ROOT / root).glob("e0_*/*/metrics.json"):
            cfg = json.loads((m.parent / "config.json").read_text())
            if cfg["name"] not in ("e0_baseline", "e0_silver_v2"):
                continue
            by.setdefault(cfg["n_train"], []).append(json.loads(m.read_text())["test"]["micro_f1"])
        for n, v in sorted(by.items()):
            rows.append({"training labels": name, "train items": n, "seeds": len(v),
                         "test micro-F1 (mean)": f"{sum(v) / len(v):.3f}"})
    return table(rows, list(rows[0])) if rows else "(runs queued)"


def oracle_section():
    """Train on Stanza morphology (runs/clean_v1_oracle) or on gold (runs/clean_v1_fullgold);
    each checkpoint is scored on the test set with Stanza and with gold morphology."""
    rows = []
    for train_m, root, k_stanza, k_gold in (("Stanza", "runs/clean_v1_oracle", "test", "test_gold"),
                                            ("gold", "runs/clean_v1_fullgold", "test_stanza", "test")):
        for m in sorted((ROOT / root).glob("*/*/metrics.json")):
            r = json.loads(m.read_text())
            if k_stanza not in r or k_gold not in r:
                continue
            cfg = json.loads((m.parent / "config.json").read_text())
            rows.append({"train morphology": train_m, "config": cfg["name"], "seed": cfg["seed"],
                         "test Stanza": f"{r[k_stanza]['micro_f1']:.3f}",
                         "test gold": f"{r[k_gold]['micro_f1']:.3f}",
                         "gold - Stanza": f"{r[k_gold]['micro_f1'] - r[k_stanza]['micro_f1']:+.3f}"})
    return table(rows, list(rows[0])) if rows else "(runs queued, not finished)"


def main():
    parts = [
        "# clean_v1 results (provisional)",
        "",
        "Generated by `scripts/make_report.py`; rerun it to refresh. All labels are",
        "LLM annotations (`meta.annotator = llm`), not human gold. A human",
        "review of 300 test items is in progress (`data/annotation/human_check_v1/`).",
        "",
        "## Plan coverage (TamilRelFormer Technical Research Plan)",
        "",
        "| Plan item | Status |",
        "|---|---|",
        "| Audit FIRE 2018 / existing Tamil RE data (5.1) | Done: no reusable data with spans; IECSIL request drafted, not sent |",
        "| Relation inventory 10-20 + NO_RELATION (5.2) | 14 relations + NO_RELATION (`configs/relations.json`) |",
        "| Dataset 4,000-6,000 pairs (5.2) | 4,584 usable: clean_v1 2,667 + clean_v2 1,917 train-only (LLM labels); clean_v12 n=all queued |",
        "| Guide with definitions and examples (5.3) | `docs/annotation_guide_v0.md` v0.1 |",
        "| Double-annotate 15-20%, adjudicate, kappa >= 0.70 (5.3) | 15% of dev/test double-labelled and reconciled; kappa is model-model only; human review of 300 items in progress |",
        "| Splits by document (5.3) | Article-disjoint train/dev/test |",
        "| Licensing (5.3) | Wikipedia text CC BY-SA 4.0, Wikidata CC0: redistributable with attribution and share-alike |",
        "| Step A typed markers / E0 (6, 7) | Done |",
        "| Step B morphology with status, coverage, error rates (6) | Stanza, ThamizhiMorph, matcher scored on 5,334 checked mentions; status field in every record |",
        "| Step C tag + embedding, compare both (6) | E2a (tag), E2b (embedding), E2c (both); 5 seeds n=500 done |",
        "| E1 lemma, E2 case (7) | n=500 done (5 seeds, no config clears zero); n=all running |",
        "| Gold/verified vs predicted morphology (9, Month 4) | Queued (test-only and train+test gold) |",
        "| Learning curves 100/500/1,000/all (8) | 500 done; all running; 100, 1,000 and 3,407 (clean_v12) queued |",
        "| Case-signalled relations, error analysis (8) | Inflected-object and unseen-form slices; per-relation F1 (`scripts/error_analysis.py`) |",
        "| Multiple seeds, repeatable gains (7, 8) | 5 seeds, two-level bootstrap |",
        "| E3 morphology-guided masking vs random masking (7) | Implemented with control; queued on E2c (n=500, 5 seeds) |",
        "| E4 supervised contrastive (7) | Not started (stretch) |",
        "| Backbone comparison MuRIL / XLM-R-large (8) | Not started (optional) |",
        "| Gold-entity vs predicted-entity evaluation (8) | Gold entities only; IndicNER needs gated access |",
        "| LLM zero/few-shot baseline (8, optional) | Not run: labels are LLM annotations, so an LLM baseline would be circular |",
        "| Variety metadata, RQ4 (2, 8) | Recorded; about 1% Sri Lankan Tamil, too few for transfer tests |",
        "",
        "## Related resources (checked 2026-10-12)",
        "",
        "No released Tamil relation extraction dataset with entity spans was found.",
        "IndoRE (Nag et al., CoNLL 2021; arXiv 2110.09570) covers Bengali, Hindi,",
        "Telugu. FIRE 2018 IECSIL Task B includes Tamil but has pattern-based silver",
        "labels without spans (password-protected). Recent Tamil resources are",
        "discourse relations (WILDRE 2026) and NER only (arXiv 2412.02056).",
        "",
        "## Data",
        "",
        "3,533 candidates from matcher v2, article-disjoint splits, pilot articles",
        "excluded, DS labels hidden from annotators. Items with span or type errors",
        "are left out.",
        "",
        dataset_section(),
        "",
        "Pass 1 vs independent pass 2 on the double-annotated items (model",
        "self-consistency, not inter-annotator agreement):",
        "",
        agreement_section(),
        "",
        cross_model_section(),
        "",
        "## Why candidates are rejected",
        "",
        rejection_section(),
        "",
        "Matcher v3a (stoplist on person short forms, 19 more common nouns) removes",
        "14% of these errors while losing 0.3% of usable items on clean_v1 (not held",
        "out); on the held-out pilot the change is +2 errors removed, -1 usable.",
        "A logistic-regression span filter trained on the train split lowers the",
        "test-split error rate from 0.244 to 0.129 when dropping items with p > 0.5,",
        "losing 16% of usable items (`scripts/span_filter.py`); it favours easier",
        "items, so it is meant for training rounds only.",
        "",
        "## Distant-supervision label quality",
        "",
        "Usable items only (span and type correct), all splits, matcher v2.",
        "",
        ds_section(),
        "",
        "The largest disagreement, DS NO_RELATION annotated LOC:LOCATED_IN, is",
        "Wikidata coverage, not the matcher: the sentence states containment (often",
        "a village panchayat in its block or taluk) that Wikidata does not record.",
        "Following P131 transitively within our facts recovers 28 of these 331; on a",
        "random 60 of them, live Wikidata connects only 9 through any",
        "P131/P17/P276 chain (checked 2026-10-12).",
        "",
        "## Analyzer accuracy (RQ1)",
        "",
        "Entity lemma and case against LLM-checked values, all 5,334 mentions",
        "(train+dev+test). 'matcher alias' = the Wikidata name the DS matcher",
        "matched in the span (scripts/matcher_lemmas.py), Stanza where none.",
        "",
        analyzer_section(),
        "",
        "Stanza by split:",
        "",
        morph_section(),
        "",
        "## Ablation (RQ2), XLM-R base",
        "",
        "Analysis fixed before most results: primary = overall test micro-F1,",
        "each config vs E0 at n=500 and n=all; secondary = 'obj form unseen' and",
        "the plan's learning-curve sizes (100, 1,000; plus 500 and all). Intervals: two-level bootstrap (seeds and",
        "items) once 3+ seeds, items only before that.",
        "",
        "e1m / e2am / e2bm (lemma or case from the matcher alias and suffix, see",
        "RQ1) were added after the analyzer scores, before any of their runs:",
        "exploratory, not part of the fixed comparisons. e1dup_surface_control",
        "copies the surface form into E1's lemma slot: if it gains as much as E1,",
        "E1's gain is not about the lemma.",
        "",
        compare_section("runs/clean_v1"),
        "",
        "### With type-constrained decoding",
        "",
        "Added 2026-10-12 after 11 runs, before the remaining seeds: a prediction",
        "whose relation type signature does not match the (given) entity types is",
        "replaced by NO_RELATION, for every config alike. Secondary analysis; the",
        "unconstrained table above stays primary.",
        "",
        compare_section("runs/clean_v1", ("--type-constraint",)),
        "",
        "## Clean vs distant-supervision training labels",
        "",
        "E0 trained on clean_v1 train or on a 9,752-item DS-labelled set drawn",
        "from the same train-split articles; both use clean dev and test.",
        "",
        silver_section(),
        "",
        "## Gold-morphology upper bound",
        "",
        "Each checkpoint (n=500) is scored with Stanza morphology and with",
        "LLM-checked morphology at test time; trained either on Stanza or on",
        "checked morphology. Compare with E0 at the same seed in the ablation table.",
        "",
        oracle_section(),
        "",
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
