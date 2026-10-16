# Results: P1-P3 (pre-registered in docs/preregistration_p123.md)

Scored 2026-10-16 with `scripts/compare_runs.py` (paired seed+item bootstrap, 2000 resamples), test = clean_v1 (887 items),
primary metric = test micro-F1 excluding NO_RELATION, unconstrained decoding, all 5 seeds (13 21 42 7 99) complete.
Labels are LLM annotations (provisional); the human-checked test subset is still pending.

| Hypothesis | Comparison | Train | F1 (cmp vs base) | Diff | 95% CI | Seeds + | Verdict |
|---|---|---|---|---|---|---|---|
| P1 | MuRIL vs XLM-R (E0) | n=500 | 0.838 vs 0.828 | +0.010 | [-0.011, +0.029] | 3/5 | null |
| P1 secondary | MuRIL vs XLM-R (E0) | n=all (1,490) | 0.894 vs 0.898 | -0.004 | [-0.014, +0.007] | 1/5 | null |
| P2 | E1 lemma vs E0, both MuRIL | n=500 | 0.845 vs 0.838 | +0.007 | [-0.005, +0.019] | n/a | null |
| P3 | E0 trained on clean_v12 (3,407) vs clean_v1 (1,490) | n=all | 0.912 vs 0.898 | **+0.014** | **[+0.006, +0.023]** | **5/5** | **win** |

P3 per seed: s7 +0.015, s13 +0.013, s21 +0.011, s42 +0.013, s99 +0.021. Subsets (secondary): obj inflected +0.015
[+0.003, +0.027], obj exact +0.013 [-0.004, +0.029], no LOCATED_IN +0.015 [+0.004, +0.026]. Caveat from the
pre-registration: clean_v2 labels are also LLM annotations and its sampling (matcher v3a) differs slightly.

Also observed at n=all on clean_v1 (primary RQ2 comparison, XLM-R, 5 seeds except E2c 4): E1 lemma vs E0 = -0.016
[-0.034, -0.001], 1/5 seeds positive, i.e. explicit lemma features significantly *hurt* with all training data.
E2a -0.009 [-0.019, +0.001], E2b -0.001 [-0.008, +0.006], E2c -0.005 [-0.015, +0.004] (seed 99 rerun in progress).

How to reproduce P3 (same config trained on two datasets lives in two roots):
```
R=$(mktemp -d); mkdir -p $R/e0_baseline $R/e0_v12data
for s in 13 21 42 7 99; do ln -s $PWD/runs/clean_v1/e0_baseline/seed$s $R/e0_baseline/seed$s
  ln -s $PWD/runs/clean_v12/e0_baseline/seed$s $R/e0_v12data/seed$s; done
.venv/bin/python scripts/compare_runs.py $R --test data/processed/clean_v1/test.jsonl --name-from-dir
```
