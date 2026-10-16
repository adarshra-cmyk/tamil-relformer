# Pre-registration H2: accurate morphology (written 2026-10-12, before any run below)

Motivation (from results already seen): at n=500, 5 seeds, no morphology config
beats E0 (E1 +0.009 [-0.004,+0.022]; E2c +0.007 [-0.008,+0.023]) and the
surface-copy control gains as much as E1. Those configs use Stanza morphology,
whose lemma accuracy on inflected mentions is 0.72-0.75. Matcher-derived lemma
is 0.963 and case 0.972 on the same mentions; LLM-checked gold morphology
exists for all clean_v1 items.

**H2.** Lemma/case features help when the morphology is accurate.

**Comparisons (all n=500, seeds 13 21 42 7 99, same splits, same baseline runs):**
1. E1m  (`e1m_lemma_matcher`, `*_matchlemma` data) vs E0
2. E2cm (`e2cm_case_tag_emb_matcher`, `*_matchmorph` data) vs E0
3. E1-gold (`e1g_lemma_gold`) and E2c-gold (`e2cg_case_tag_emb_gold`), `*_goldmorph` train/dev/test, vs E0: upper bound

**Primary metric:** overall test micro-F1, unconstrained decoding, paired
seed+item bootstrap (`scripts/compare_runs.py`, 2000 resamples). A config
"wins" only if the 95% interval excludes zero with all 5 seeds. No extra seeds
are added after seeing results. Type-constrained numbers are reported alongside,
not substituted. Subsets (inflected, unseen form) are secondary/descriptive.

**Also pre-specified:** the same two matcher configs at n=100 (seeds 13 21 42),
since feature help is expected to be larger with less data.

**Caveat stated in advance:** matcher morphology uses the Wikidata entity the
mention is linked to (every item comes from the DS matcher). A positive result
means "morphology from entity linking helps", not "a Tamil analyser helps";
the E1 surface-copy control still applies when interpreting E1m.

Input check (before runs): vs Stanza, the matcher changes 232/1774 test lemmas and 174/1774 cases; gold changes 217 and 162.
All H2 runs write to runs/clean_v1 so they share the E0 baseline runs.

## Results (2026-10-12 18:10; n=500, 5 seeds, unconstrained, vs E0)

| config | diff | 95% CI | seeds + |
|---|---|---|---|
| E2c, Stanza case (reference) | +0.007 | [-0.008, +0.023] | 3/5 |
| E2cm, matcher case | +0.000 | [-0.014, +0.014] | 2/5 |
| E2cg, gold case | +0.002 | [-0.014, +0.018] | 3/5 |

H2 is not supported for case: accurate case does not beat E0 and does not beat Stanza case.
Consistent with the code audit's information bound (entity suffix characters predict gold case
at 96.7%; gold case adds ~0 to E0 predictions). Separate audit finding: all added special tokens
start near-identical, so case tags were hard to tell apart in all case-tag runs (see P5/P6).
E1m / E1g (lemma part of H2) still queued.
