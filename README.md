# TamilRelFormer

Morphology-aware entity representations for Tamil relation extraction.

Tamil is agglutinative: an entity like சென்னை appears as சென்னையில், சென்னைக்கு, and so on.
This project tests whether giving a relation-extraction encoder explicit lemma and case
information about each entity mention helps, especially with little training data, and
compares backbones (XLM-R, MuRIL). Data comes from Tamil Wikipedia sentences aligned to
Wikidata facts, relabelled by an LLM annotator with a human-checked test subset.

All comparisons are pre-registered (`docs/preregistration_*.md`) and scored with a paired
seed-and-item bootstrap over 5 seeds (`scripts/compare_runs.py`); a result counts only if
the 95% CI excludes zero. Current results are in `docs/results_p123.md`.

## Layout

| Path | What |
|---|---|
| `trf/schema.py` | dataset format (plan 5.2) plus entity character offsets |
| `trf/morph.py` | Stanza and ThamizhiMorph analyzers; per-entity lemma/case/status |
| `trf/encode.py` | typed entity markers; E1 lemma-in-text, E2a case tag, E2b case ids |
| `trf/model.py` | XLM-R start-marker pair classifier, optional case embedding |
| `trf/metrics.py` | micro-F1 without NO_RELATION, macro-F1, per-relation, confusion |
| `trf/train.py` | one run; writes config, split hashes, dev history, test metrics, predictions |
| `configs/` | `relations.json` (inventory v0) and E0/E1/E2a/E2b/E2c run configs |
| `scripts/fetch_wikidata.py` | Wikidata facts for subjects with Tamil Wikipedia articles |
| `scripts/build_ds_candidates.py` | distant-supervision candidates with inflection-aware matching |
| `scripts/annotation_io.py` | pilot sheet export (DS label hidden), import, Cohen's kappa |
| `scripts/make_splits.py` | document-grouped splits + predicted morphology |
| `scripts/run_ablation.sh`, `summarize_runs.py` | E0 to E2 over seeds and train sizes |
| `scripts/score_probe.py`, `data/probe/` | analyzer sanity probe (26 gold lemma/case items) |
| `docs/annotation_guide_v0.md` | annotation guide (v0.1 rules: spans, types, nationality adjectives) |
| `scripts/fetch_entity_types.py` | entity types from Wikidata P31/P279 (built places = LOCATION, mythological figures = PERSON) |
| `scripts/eval_matcher_on_pilot.py` | matcher vs pilot judgements (`--split tune/held`, relinked outcome) |
| `configs/alias_stoplist.txt` | reviewed common-word / wrong-entity aliases (`~` = alias only) |
| `scripts/make_clean_round.py`, `import_clean_round.py` | clean_v1: sample, LLM batches, agreement, import with morphology |
| `scripts/matcher_lemmas.py` | lemma and case from the matcher alias and suffix (`*_matchlemma`, `*_matchmorph`) |
| `scripts/score_analyzers.py` | RQ1: Stanza vs ThamizhiMorph vs matcher against checked morphology |
| `scripts/compare_runs.py` | paired comparison vs E0, seed+item bootstrap, slices |
| `scripts/error_analysis.py` | exploratory per-relation F1 and item flips |
| `scripts/make_report.py` | regenerates `docs/clean_v1_report.md` from repo files |

Environment: `.venv` (native arm64 Python 3.11, torch with MPS). ThamizhiMorph
needs `brew install foma`; FSTs are cloned into `data/raw/thamizhi-morph`.

```
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/fetch_wikidata.py
.venv/bin/python scripts/build_ds_candidates.py
.venv/bin/python scripts/annotation_io.py export
.venv/bin/python scripts/make_splits.py data/processed/annotated.jsonl
scripts/run_ablation.sh "13 21 42" "100 500 1000 0"
```

## clean_v1 (current)

Labels in `data/processed/clean_v1/` are LLM annotations (provisional, not
human gold); `data/annotation/human_check_v1/` holds the human review. The
`*_goldmorph` files carry LLM-checked lemma/case in `meta.morph`.

```
.venv/bin/python scripts/build_ds_candidates.py --neg-per-article 100000 --out runs/matcher_cmp/cands_v2.jsonl
.venv/bin/python scripts/make_clean_round.py                  # then label llm/batches/*
.venv/bin/python scripts/import_clean_round.py agree          # writes recon batches
.venv/bin/python scripts/import_clean_round.py import
.venv/bin/python scripts/matcher_lemmas.py data/processed/clean_v1
D=data/processed/clean_v1
scripts/run_ablation.sh "13 21 42 7 99" "500 0" --out runs/clean_v1 --set train=$D/train.jsonl dev=$D/dev.jsonl test=$D/test.jsonl
.venv/bin/python scripts/compare_runs.py runs/clean_v1 --test $D/test.jsonl --train $D/train.jsonl
.venv/bin/python scripts/make_report.py
```

## Data and licensing

Sentences are from Tamil Wikipedia (CC BY-SA 4.0) and relation facts from Wikidata (CC0).
Labels are LLM annotations (provisional); `data/annotation/human_check_v1/` holds the
human review. Large intermediate files (distant-supervision candidates, the Wikidata fact
dump, silver splits, morphology caches) are not included; the scripts above regenerate them.
