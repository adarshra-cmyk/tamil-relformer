#!/usr/bin/env bash
# Resumable clean_v1 queue: every cell is skipped if its metrics.json exists,
# and a run dir without metrics.json (killed mid-run) is deleted and redone.
# Usage: nohup scripts/run_queue_clean_v1.sh [PID_TO_WAIT_FOR] > runs/clean_v1_queue.log 2>&1 &
# (pass the PID of a still-running train.py so its run dir is not deleted as unfinished)
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -n "${1:-}" ]; then while kill -0 "$1" 2>/dev/null; do sleep 30; done; fi
D=data/processed/clean_v1
ALL5="13 21 42 7 99"; S3="13 21 42"

# cell OUT CFG SEED SIZE(0=all) DATA_SUFFIX [extra train.py args]
cell() {
  local out=$1 cfg=$2 seed=$3 size=$4 sfx=$5; shift 5
  local tag="seed$seed"; [ "$size" != 0 ] && tag="${tag}_n$size"
  local dir="$out/$cfg/$tag"
  [ -f "$dir/metrics.json" ] && return 0
  rm -rf "$dir"
  local lim=(); [ "$size" != 0 ] && lim=(--train-limit "$size")
  echo "### $(date '+%F %T') start $out $cfg $tag"
  .venv/bin/python -m trf.train --config "configs/$cfg.json" --seed "$seed" ${lim[@]+"${lim[@]}"} \
    --out "$out" --set train=$D/train$sfx.jsonl dev=$D/dev$sfx.jsonl test=$D/test$sfx.jsonl "$@" \
    || echo "### FAILED $out $cfg $tag"
}
MAIN="e0_baseline e1_lemma e2a_case_tag e2b_case_emb e2c_case_tag_emb"

# 0. H2 case configs first (lead from scripts/probe_leads.py: case features help only when case is right)
for s in $ALL5; do
  cell runs/clean_v1 e2cm_case_tag_emb_matcher $s 500 _matchmorph
  cell runs/clean_v1 e2cg_case_tag_emb_gold $s 500 _goldmorph
done
# P5/P6 (docs/preregistration_p56.md): informed marker init, n=500, 5 seeds
for s in $ALL5; do for c in e0_minit e2a_minit e2c_minit; do cell runs/clean_v1 $c $s 500 ""; done; done
# P1/P2 (docs/preregistration_p123.md): MuRIL E0 and E1, n=500, 5 seeds
for s in $ALL5; do for c in e0_muril e1_muril; do cell runs/clean_v1 $c $s 500 ""; done; done
# P3: E0 n=all on clean_v1 and on clean_v12
for s in $ALL5; do cell runs/clean_v1 e0_baseline $s 0 ""; done
D=data/processed/clean_v12
for s in $ALL5; do cell runs/clean_v12 e0_baseline $s 0 ""; done
D=data/processed/clean_v1
# P1 secondary: MuRIL n=all
for s in $ALL5; do cell runs/clean_v1 e0_muril $s 0 ""; done
# 1. n=all, primary (5 seeds)
for s in $ALL5; do for c in $MAIN; do cell runs/clean_v1 $c $s 0 ""; done; done
# 1b. H2 (docs/preregistration_h2.md): accurate morphology, n=500 x5, then n=100 x3
for s in $ALL5; do
  cell runs/clean_v1 e1m_lemma_matcher $s 500 _matchlemma
  cell runs/clean_v1 e2cm_case_tag_emb_matcher $s 500 _matchmorph
  cell runs/clean_v1 e1g_lemma_gold $s 500 _goldmorph
  cell runs/clean_v1 e2cg_case_tag_emb_gold $s 500 _goldmorph
done
for s in $S3; do for c in e1m_lemma_matcher:_matchlemma e2cm_case_tag_emb_matcher:_matchmorph; do
  cell runs/clean_v1 ${c%%:*} $s 100 ${c##*:}
done; done
# 2. low-data, pre-registered secondary (plan sizes 100, 1000; 3 seeds)
for n in 100 1000; do for s in $S3; do for c in $MAIN; do cell runs/clean_v1 $c $s $n ""; done; done; done
# 3. E3 masking on the E2c base vs random-mask control (n=500, 5 seeds)
for s in $ALL5; do for c in e3_e2c_suffixmask e3_e2c_randommask; do cell runs/clean_v1 $c $s 500 ""; done; done
# 3b. combined clean_v1 + clean_v2 training set (3,407 items), n=all, same dev/test
D=data/processed/clean_v12
for s in $ALL5; do for c in $MAIN; do cell runs/clean_v12 $c $s 0 ""; done; done
D=data/processed/clean_v1
# 4. exploratory (n=500)
for s in $ALL5; do cell runs/clean_v1 e1m_lemma_matcher $s 500 _matchlemma; done
for s in $S3; do
  cell runs/clean_v1 e2am_case_tag_matcher $s 500 _matchmorph
  cell runs/clean_v1 e2bm_case_emb_matcher $s 500 _matchmorph
  cell runs/clean_v1 e0_typemask $s 500 ""
  cell runs/clean_v1 e1m_typemask $s 500 _matchlemma
  cell runs/clean_v1 e3m_lemma_case_typemask $s 500 _matchmorph
done
for s in $S3; do for n in 500 0; do
  cell runs/clean_v1 e0_silver_v2 $s $n "" --set train=$D/train_silver_v2.jsonl dev=$D/dev.jsonl test=$D/test.jsonl
done; done
# 5. gold morphology: test-only oracle and train+test gold (n=500, 3 seeds)
for s in $S3; do for c in e1_lemma e2a_case_tag e2b_case_emb e2c_case_tag_emb; do
  cell runs/clean_v1_oracle $c $s 500 "" --extra-test gold=$D/test_goldmorph.jsonl
  [ $c = e2a_case_tag ] || [ $c = e2b_case_emb ] && cell runs/clean_v1_fullgold $c $s 500 _goldmorph
done; done
echo "### $(date '+%F %T') queue done"
