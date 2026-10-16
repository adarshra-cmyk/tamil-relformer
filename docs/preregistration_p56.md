# Pre-registration P5-P6: informed marker initialisation (written 2026-10-12 ~18:00, before any run)

Finding (code audit, docs in this commit's message): `resize_token_embeddings` starts all 34
added tokens (typed markers, <LEM>, 13 <CASE-*> tags) as near-identical vectors (pairwise
cosine 0.9999996), and at lr 3e-5 they barely separate. E0 makes type-impossible predictions on
4-6% of test items. Plan Step A asks for "registered special tokens or a carefully tested marker
scheme". Fix (opt-in `marker_init: "informed"`): each new token starts as the mean embedding of
its description ("subject person", the real case suffix இல் / இன் / க்கு ...); deterministic,
no other change. After init: cos(<CASE-Nom>,<CASE-Loc>) 0.26, cos(<OBJ-PER>,<OBJ-LOC>) 0.85.

Rules as in docs/preregistration_p123.md: n=500, seeds 13 21 42 7 99, test micro-F1 excl.
NO_RELATION, unconstrained decoding, paired seed+item bootstrap, win = 95% CI excludes zero with
all 5 seeds, no extra seeds, no hyperparameter changes.

**P5 (plan Step A).** `e0_minit` vs `e0_baseline`. Expected mechanism: fewer type-impossible
predictions. Secondary/descriptive: count of type-impossible predictions per run.

**P6 (RQ2 retest with case tags the model can tell apart).** `e2a_minit` and `e2c_minit` vs
`e0_minit` (`--baseline e0_minit`). Prior stated in advance: the audit's information bound says
gold case adds about 0 beyond E0 (word endings predict case at 96.7%), so P6 is expected null.

## P5 result (2026-10-12 23:15; n=500, 5 seeds, 2000 resamples)

Primary: `e0_minit` vs `e0_baseline`, test micro-F1 **+0.014 [-0.001, +0.028]**, 4/5 seeds
positive (s7 +0.000, s13 +0.003, s21 +0.032, s42 +0.025, s99 +0.008). The CI includes zero, so
by the pre-registered rule P5 is **not a win**.

Secondary (pre-specified, descriptive): type-impossible predictions per run, E0 [35, 49, 40, 41,
41] (mean 41.2) vs informed init [9, 7, 16, 13, 14] (mean 11.8), lower on 5/5 seeds. The
mechanism the fix targets works; the F1 gain is too small to separate from noise at n=500.

Subsets (`obj exact` +0.024 [+0.000,+0.048], `no LOCATED_IN` +0.022 [+0.004,+0.039]) are
secondary/robustness slices and are not claimed.

## P6 result (2026-10-12 23:45; vs `e0_minit`, n=500, 5 seeds)

| config | diff | 95% CI | seeds + |
|---|---|---|---|
| e2a_minit | +0.006 | [-0.008, +0.021] | 4/5 |
| e2c_minit | +0.008 | [-0.008, +0.024] | 4/5 |

Null, as stated in advance. Case tags the model can tell apart still add nothing measurable.
