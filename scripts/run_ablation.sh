#!/usr/bin/env bash
# Core ablation (plan section 7: E0, E1, E2) over seeds and train sizes.
# Usage: scripts/run_ablation.sh "13 21 42" "100 500 1000 0"   (0 = all training data)
# Extra args after the two lists go to trf.train, e.g. --set train=... dev=... test=...
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${1:-"13 21 42"}; SIZES=${2:-"0"}; shift 2 || true
for size in $SIZES; do
  for seed in $SEEDS; do
    for cfg in e0_baseline e1_lemma e2a_case_tag e2b_case_emb; do
      lim=(); [ "$size" != "0" ] && lim=(--train-limit "$size")
      .venv/bin/python -m trf.train --config configs/$cfg.json --seed "$seed" ${lim[@]+"${lim[@]}"} "$@"
    done
  done
done
.venv/bin/python scripts/summarize_runs.py
