# Methods (draft)

Status: draft written 2026-10-12. Labels are LLM annotations; a human
check of 300 test items is in progress. Numbers live in
`docs/clean_v1_report.md`, which is generated from the repository.

## Task

Sentence-level relation classification for Tamil. Each item is one sentence
with two marked entities, E1 (subject) and E2 (object), and one of 14
relations or NO_RELATION (`configs/relations.json`). The relation is read as
"E1 relation E2" and must be stated by the sentence itself.

## Candidate generation (distant supervision)

1. Facts: 212k Wikidata statements whose subject has a Tamil Wikipedia
   article (`scripts/fetch_wikidata.py`).
2. Entity types from P31 classes with an upward P279 walk
   (`scripts/fetch_entity_types.py`): PERSON (human; deities, mythological and
   legendary figures), ORGANIZATION, LOCATION (geographic locations, then
   built structures such as temples, dams and airports). Geopolitical classes
   are checked before ORGANIZATION.
3. Matching (`scripts/build_ds_candidates.py`, matcher v2): each subject's
   article is split into sentences; entity names and aliases are matched with
   case suffixes and clitics allowed after the stem. v2 adds a reviewed alias
   stoplist, exact-only matching for names under four characters, a surname
   rule limited to name fragments, extension of a span over the person's own
   name words, and slot types for entities with no P31 class.
4. A candidate pair gets the Wikidata relation if one exists, else
   NO_RELATION.

Matcher changes were tuned on one half of a 680-item pilot (split by
article) and checked once on the other half.

## Annotation (clean_v1)

- 3,533 candidates sampled from matcher v2, balanced over DS labels, with
  articles assigned to train, dev or test by hash so no article is in two
  splits; pilot articles excluded (`scripts/make_clean_round.py`).
- Each item judged for span correctness, type correctness and relation under
  `docs/annotation_guide_v0.md`, with the DS label hidden. 30 independent
  LLM annotators (model A) of 120 items each; 15% of dev and test items
  labelled a second time independently; differing items reconciled by a
  third pass.
- Items with a span or type error are excluded, leaving 1,490 / 290 / 887.
- Checks: pass 1 vs pass 2 agreement (same model); a blind pass by a second LLM (model B)
  on 300 test items (different model); a human review of the same 300 items.

## Morphology

Entity lemma and case come from one of: Stanza (default), ThamizhiMorph,
the matcher (the Wikidata name matched in the span as lemma; the remaining
suffix mapped to case), or LLM-checked values (`*_goldmorph`). Case is
read from the last word of a multi-word name.

## Models

XLM-R base with typed entity start markers; the classifier reads the two
marker positions (`trf/model.py`).

- E0: markers only.
- E1: lemma written inside the entity marker (`<LEM> lemma`).
- E2a: case written as a tag inside the marker (`<CASE-Loc>`).
- E2b: case as a learned embedding concatenated to the marker states.
- Variants with matcher morphology (E1m, E2a-m, E2b-m), with type masking
  in training (`type_mask`), and a combined lemma + case model (E3m), are
  exploratory: they were added after first results.

Training: AdamW (encoder 3e-5, head 1e-4), batch 32, up to 1,500 steps,
dev evaluation every 100 steps, early stopping after 4 evaluations without a
0.005 gain, best dev checkpoint used for test. The n-item training subsets
are label-stratified and depend only on the seed, so all configs see the same
items for a given seed.

## Evaluation

- Metric: micro-F1 over relations (NO_RELATION excluded from precision and
  recall), plus macro-F1 and per-relation F1.
- Fixed before most results: primary comparison = each morphology config vs
  E0 on the full test set at n=500 and n=all, 5 seeds; secondary = the "object
  form unseen" slice (object entity seen in that run's training subset but
  never in this surface form) and the plan's learning-curve sizes 100 and 1,000 (with 500 and all).
- Paired by seed. 95% intervals from a two-level bootstrap (resample seeds,
  then test items) once 3 or more seeds exist.
- Robustness: results without LOC:LOCATED_IN items; type-constrained
  decoding (a prediction whose type signature does not match the entity types
  becomes NO_RELATION), added after 11 runs and reported as secondary.
- Upper bound: the same configs scored with LLM-checked morphology at test
  time, and trained and tested on it.
