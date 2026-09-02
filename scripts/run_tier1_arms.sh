#!/bin/bash
# Tier 0+1 pre-registered measurement protocol (plan Task 10).
#
#   scripts/run_tier1_arms.sh --dataset DIR --out DIR [--steps N] [--budget N]
#                             [--max-resolution N] [--seed N] [--splatstats DIR]
#
# WHY THIS IS A COMMITTED SCRIPT AND NOT SHELL HISTORY. A first attempt at this
# protocol was launched from an agent turn and died with it at step 7,500 of 30,000,
# leaving nothing behind -- the protocol existed only in one process's argv. Committed
# means it survives a dead turn, a reviewer can read exactly what was run, and anyone can
# re-run it. Launch it DETACHED (`setsid nohup ... &`) so it outlives whatever started it.
#
# ORDERING IS LOAD-BEARING AND ENFORCED HERE, NOT BY THE OPERATOR'S DISCIPLINE.
# Floors are run, scored, and WRITTEN before any treatment arm is scored. Checkpoint D
# verifies this by file mtime, so the phases below must not be reordered or parallelised.
#
# TWO RULES FOR ANY MUTATION TEST RUN AGAINST THIS WORK. Both were learned the expensive
# way; this project has now produced FIVE test results that looked like evidence and were
# not (a fixture whose plane family made both rules agree; a test asserting over an empty
# array; an orientation test whose geometry put n_z at exactly 0; `**_ignored` swallowing
# `aux_colors`; and a mutation battery run without `--with scikit-image`, where every
# "kill" was really `test_ssim_matches_skimage`'s ModuleNotFoundError):
#
#   1. ASSERT ON THE FAILING TEST'S NAME, never on a failure count. A count cannot tell
#      your mutant's kill from an unrelated broken import.
#   2. VERIFY THE MUTANT ACTUALLY CHANGES BEHAVIOUR. One "killed" mutant here had in fact
#      SURVIVED: it patched a cache's read side but not its write side, so the cache never
#      populated and the mutation was a no-op.
#
# SCORING NOTES.
#   * thin-axis is scored against points3D.tsdf.txt -- the TSDF-ONLY cloud -- NEVER the
#     seed that was trained on. Getting that wrong was an 11.6 deg error once, larger than
#     every recipe gain in CLAUDE.md's table.
#   * thin/thick RATIO is not thin-axis ANGLE. They are different metrics; do not conflate.
#   * P-GEOM carries no masks, so masked PSNR == unmasked PSNR there. Say so; do not let a
#     "masked PSNR" column imply a masked result.
set -euo pipefail

DATASET=""; OUT=""; STEPS=30000; BUDGET=500000; MAXRES=1920; SEED=42
MG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLATSTATS="$(cd "$MG/../../analyze/splatstats" 2>/dev/null && pwd || echo "")"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --budget) BUDGET="$2"; shift 2;;
    --max-resolution) MAXRES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --splatstats) SPLATSTATS="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$DATASET" && -n "$OUT" ]] || { echo "need --dataset and --out" >&2; exit 2; }
DATASET="$(cd "$DATASET" && pwd)"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
TSDF="$DATASET/sparse/0/points3D.tsdf.txt"
[[ -f "$TSDF" ]] || { echo "missing TSDF-only reference cloud $TSDF" >&2; exit 2; }
SEED2=$((SEED + 1))

# Flags every arm shares. Arms differ in ONE variable-group; see the table below.
common=(--colmap "$DATASET/sparse/0" --images "$DATASET/images"
        --max-resolution "$MAXRES" --steps "$STEPS" --budget "$BUDGET"
        --eval-split-every 8 --eval-every 2500)

run_arm() {                      # run_arm NAME SEED [extra flags...]
  local arm="$1" seed="$2"; shift 2
  if [[ -f "$OUT/$arm.json" ]]; then echo "[$(date +%T)] $arm: report exists, skipping"; return; fi
  echo "[$(date +%T)] $arm: training (seed $seed) $*"
  ( cd "$MG" && caffeinate -i uv run --frozen python -m metal_gauss.train \
      "${common[@]}" --seed "$seed" \
      --report "$OUT/$arm.json" --export "$OUT/$arm.ply" --eval-dump "$OUT/$arm.dump" \
      "$@" ) > "$OUT/$arm.log" 2>&1
  echo "[$(date +%T)] $arm: trained"
}

score_arm() {                    # score_arm NAME -- splatstats vs TSDF-only + LPIPS
  local arm="$1"
  if [[ -f "$OUT/$arm.stats.json" && -f "$OUT/$arm.dump/lpips.json" ]]; then
    echo "[$(date +%T)] $arm: already scored"; return; fi
  echo "[$(date +%T)] $arm: scoring"
  ( cd "$SPLATSTATS" && caffeinate -i uv run --frozen python scripts/splat_stats.py \
      "$OUT/$arm.ply" --seed "$TSDF" --json "$OUT/$arm.stats.json" --quiet ) \
      >> "$OUT/$arm.log" 2>&1
  ( cd "$MG" && caffeinate -i uv run scripts/lpips_eval.py "$OUT/$arm.dump" ) \
      >> "$OUT/$arm.log" 2>&1
  echo "[$(date +%T)] $arm: scored"
}

echo "=== PHASE 1: floor arms (B0a/B0b same seed $SEED; B0c seed $SEED2) ==="
run_arm B0a "$SEED"
run_arm B0b "$SEED"
run_arm B0c "$SEED2"

echo "=== PHASE 2: score floors, then WRITE floors.json (must predate any treatment) ==="
score_arm B0a; score_arm B0b; score_arm B0c
( cd "$MG" && uv run --frozen python scripts/tier1_floors.py "$OUT" ) | tee -a "$OUT/floors.log"
[[ -f "$OUT/floors.json" ]] || { echo "floors.json was not written; refusing to grade arms" >&2; exit 1; }
touch "$OUT/FLOORS_DONE"

echo "=== PHASE 3: treatment arms ==="
# F1 = Tier 0 (flatten alone). R1 = F1 + the three geometry weights AS A GROUP -- that is
# three variables at once, deliberately, because the recipe is tested as a unit exactly as
# CLAUDE.md's "Indoor haze" measurement did. Nobody may later call R1-vs-F1 one-variable.
run_arm F1 "$SEED" --flatten-loss-weight 1.0
run_arm R1 "$SEED" --flatten-loss-weight 1.0 \
        --depth-loss-weight 1.0 --normal-loss-weight 0.2 --depth-normal-weight 0.05

echo "=== PHASE 4: score treatments ==="
score_arm F1; score_arm R1
touch "$OUT/ALL_DONE"
echo "[$(date +%T)] protocol complete: $OUT"
