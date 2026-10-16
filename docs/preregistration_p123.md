# Pre-registration P1-P3 (written 2026-10-12 ~17:30, before any MuRIL or clean_v12 run)

Shared rules: dev/test = clean_v1 (887 test items); seeds 13 21 42 7 99; primary metric =
test micro-F1 excluding NO_RELATION, unconstrained decoding; paired seed+item bootstrap
(scripts/compare_runs.py, 2000 resamples); a win = 95% CI excludes zero with all 5 seeds.
No extra seeds, no reruns of diverged seeds, no hyperparameter changes (lr 3e-5, head_lr 1e-4,
batch 32, max_steps 1500, patience 4, max_len 256). Constrained decoding and subsets are
reported as secondary only. Three separate hypotheses; all three are reported, nulls included.

**P1 (RQ1, plan 8 backbone comparison).** `e0_muril` (google/muril-base-cased) vs `e0_baseline`
(xlm-roberta-base), n=500. MuRIL wins if the CI of MuRIL - XLM-R is above zero; a CI below zero
is reported as XLM-R winning. Secondary: same comparison at n=all.

**P2 (RQ2 on a second backbone).** `e1_muril` vs `e0_muril`, n=500 (`--baseline e0_muril`).
Rationale fixed in advance: on 263 single-word inflected test objects MuRIL's tokenizer puts a
boundary at the stem/suffix split in 27% of cases vs 60% for XLM-R, and keeps the whole word as
one token in 40% vs 15%, so an explicit lemma should carry more new information for MuRIL.
Secondary/descriptive: MuRIL E1-E0 gain vs XLM-R E1-E0 gain (+0.009). If P2 wins, a MuRIL
surface-copy control is required before calling it a lemma effect.

**P3 (plan 8 learning curve, all available data).** `e0_baseline` trained on clean_v12 train
(3,407 items) vs `e0_baseline` on clean_v1 train (1,490), n=all, same seeds/dev/test. Win if the
CI of the difference is above zero. Caveat: clean_v2 labels are also LLM annotations and its
sampling (matcher v3a) differs slightly from clean_v1.

Ranking and odds came from a plan review (P1 ~35-45%, P2 ~15-20%, P3 ~65-75% chance of a CI
excluding zero); H2 and RQ2 at other sizes are expected null.
