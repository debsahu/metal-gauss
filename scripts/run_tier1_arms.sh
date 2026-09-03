#!/bin/bash
# Tier 0+1 pre-registered measurement protocol (plan Task 10), any scene.
#
#   scripts/run_tier1_arms.sh --dataset DIR   --out DIR [--seed-cloud PATH]
#                             [--colmap DIR] [--images DIR]
#                             [--depth-dir DIR] [--normal-dir DIR]
#                             [--arms A,B,..] [--floors A,B[,C]]
#                             [--steps N] [--budget N] [--max-resolution N] [--seed N]
#   scripts/run_tier1_arms.sh --blender DIR   --out DIR ...        (NeRF-synthetic)
#
# WHY THIS IS A COMMITTED SCRIPT AND NOT SHELL HISTORY. A first attempt at this protocol
# was launched from an agent turn and died with it at step 7,500 of 30,000, leaving nothing
# behind -- the protocol existed only in one process's argv. Launch it DETACHED. `setsid`
# does not exist on macOS; use Python's `start_new_session=True`, which is a real setsid(2).
#
# DO NOT RUN THIS FILE DIRECTLY -- launch it through scripts/launch_tier1.sh, which copies
# it to an immutable snapshot inside the output directory and executes that.
#
# bash reads a script lazily by byte offset and re-reads after each command, so editing a
# running script shifts those offsets and makes it execute garbage. On 2026-09-02 that
# killed a five-arm ARKitScenes run 62 minutes in -- "syntax error near unexpected token
# ')'" on a file `bash -n` passes -- because a one-line arm was added while it ran. This
# warning was already here and was not enough: the edit and the run were requested hours
# apart. The snapshot makes it structural.
#
# FREEZE EVERYTHING A RUNNING JOB READS, NOT JUST THIS FILE. The snapshot above protects
# the script, and that rule proved TOO NARROW on 2026-09-03: equivalence gate #2 launches
# six sequential training processes, each of which recompiles metal_gauss/csrc/*.metal at
# startup, and editing that source mid-gate would have split the arms across two binaries
# -- in a gate whose whole purpose is comparing two binaries. Caught before any arm wrote a
# report, but nothing enforced it. A long job's inputs include the launch script, shaders
# it JIT-compiles, config files, prior directories and datasets. Freeze all of them.
#
# ORDERING IS LOAD-BEARING AND ENFORCED HERE, NOT BY OPERATOR DISCIPLINE. Floors are run,
# scored, and WRITTEN before any treatment arm is scored. Checkpoint D verifies this by file
# mtime, so the phases must not be reordered or parallelised. Never run two scenes at once
# either: GPU contention corrupts every ms/step column.
#
# TWO RULES FOR ANY MUTATION TEST RUN AGAINST THIS WORK, both learned expensively. This
# project has now produced SIX test results that looked like evidence and were not: a
# fixture whose plane family made both rules agree; a test asserting over an empty array; an
# orientation test whose geometry put n_z at exactly 0; `**_ignored` swallowing `aux_colors`;
# a mutation battery run without `--with scikit-image`, where every "kill" was really an
# unrelated ModuleNotFoundError; and a monitor whose failure filter matched "oom" inside
# "playroom".
#
#   1. ASSERT ON THE FAILING TEST'S NAME, never on a failure count.
#   2. VERIFY THE MUTANT ACTUALLY CHANGES BEHAVIOUR -- one "killed" mutant had in fact
#      SURVIVED, having patched a cache's read side but not its write side.
#
# And the generalisation of both, from the flatten double-add: THREE INDEPENDENT
# MEASUREMENTS AGREEING IS ONLY EVIDENCE WHEN THEY CAN FAIL INDEPENDENTLY. The plan's probe,
# this fork's reproduction and the reviewer's all agreed on flatten's effect because all
# three ran the same doubled code.
#
# SCORING NOTES.
#   * thin-axis is scored against a reference cloud that was NEVER TRAINED ON. Scoring it
#     against the trained seed was an 11.6 deg error once, larger than every recipe gain in
#     CLAUDE.md's table. tier1_floors.py refuses to proceed if the arms disagree about the
#     reference or if it is the COLMAP points3D.txt the trainer seeded from.
#   * thin/thick RATIO is not thin-axis ANGLE. Different metrics; never substitute one.
#   * A scene with no masks reports mean coverage 100% and masked PSNR == unmasked PSNR.
#     Say so; do not let a "masked PSNR" column imply a masked result.
#   * ms/step is a SCHEDULE AVERAGE, not a full-resolution figure: --num-downscales
#     defaults to 2, so a 30k run spends a third of its steps at 1/16 of the pixels, a
#     third at 1/4, and a third at full -- mean 0.4375. Identical across arms, so A/Bs are
#     valid, but the number is not "at <max-resolution>".
set -euo pipefail

DATASET=""; BLENDER=""; OUT=""; SEED_CLOUD=""; DEPTH_DIR=""; NORMAL_DIR=""
COLMAP_DIR=""; IMAGES_DIR=""; INIT_PLY=""
STEPS=30000; BUDGET=500000; MAXRES=1920; SEED=42
ARMS="B0a,B0b,B0c,F1,R1"; FLOORS="B0a,B0b,B0c"
# MG_ROOT is set by launch_tier1.sh. The snapshot it executes lives in the OUTPUT
# directory, so deriving the repo root from ${BASH_SOURCE[0]} would point at the output
# tree and every helper script would vanish -- which is exactly what happened first try.
MG="${MG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLATSTATS="$(cd "$MG/../../analyze/splatstats" 2>/dev/null && pwd || echo "")"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    # A dataset does not always lay itself out as <root>/sparse/0 + <root>/images:
    # ARKitScenes keeps poses in sparse_colmap_for_moge/0 and images in ds/images.
    --colmap) COLMAP_DIR="$2"; shift 2;;
    --images) IMAGES_DIR="$2"; shift 2;;
    # Poses and seed do not always live together: ARKitScenes has 656 posed images with
    # ZERO points3D and its 1.13M-point seed in a separate ply.
    --init-ply) INIT_PLY="$2"; shift 2;;
    --blender) BLENDER="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --seed-cloud) SEED_CLOUD="$2"; shift 2;;
    --depth-dir) DEPTH_DIR="$2"; shift 2;;
    --normal-dir) NORMAL_DIR="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;;
    --floors) FLOORS="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --budget) BUDGET="$2"; shift 2;;
    --max-resolution) MAXRES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --splatstats) SPLATSTATS="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$OUT" ]] || { echo "need --out" >&2; exit 2; }
[[ -n "$DATASET" || -n "$BLENDER" ]] || { echo "need --dataset or --blender" >&2; exit 2; }
[[ -z "$DATASET" || -z "$BLENDER" ]] || { echo "--dataset and --blender are exclusive" >&2; exit 2; }
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
SEED2=$((SEED + 1))
IFS=',' read -r -a ARM_LIST <<< "$ARMS"
IFS=',' read -r -a FLOOR_LIST <<< "$FLOORS"
# The third floor arm is the SEED floor; the first two share a seed and give the REPEAT
# floor, which is the correct yardstick for the paired same-seed A/Bs this protocol runs.
SEEDFLOOR_ARM="${FLOOR_LIST[2]:-}"

if [[ -n "$DATASET" ]]; then
  DATASET="$(cd "$DATASET" && pwd)"
  source_flags=(--colmap "${COLMAP_DIR:-$DATASET/sparse/0}"
                --images "${IMAGES_DIR:-$DATASET/images}")
  [[ -n "$INIT_PLY" ]] && source_flags+=(--init-ply "$INIT_PLY")
  [[ -n "$DEPTH_DIR" ]] && source_flags+=(--depth-dir "$DEPTH_DIR")
  [[ -n "$NORMAL_DIR" ]] && source_flags+=(--normal-dir "$NORMAL_DIR")
else
  BLENDER="$(cd "$BLENDER" && pwd)"
  source_flags=(--blender "$BLENDER")
fi
common=("${source_flags[@]}" --max-resolution "$MAXRES" --steps "$STEPS"
        --budget "$BUDGET" --eval-split-every 8 --eval-every 2500)

arm_flags() {                    # extra flags per arm NAME
  case "$1" in
    # B0d: a baseline repeat on a LATER binary. Reusing floors measured on an earlier
    # commit is only sound if the drift between them is inert for arms that carry no
    # geometry weight -- which is an argument, not a measurement. B0d turns it into one:
    # it must land inside the recorded B0a/B0b repeat floor.
    B0a|B0b|B0c|B0d|L0a|L0b|L0c) : ;;
    F1|L1)  echo "--flatten-loss-weight 1.0" ;;
    L2)     echo "--depth-normal-weight 0.05" ;;
    # R1 moves THREE weights at once, deliberately: the recipe is tested as a unit, as
    # CLAUDE.md's "Indoor haze" measurement did. Nobody may later call it one-variable.
    R1)     echo "--flatten-loss-weight 1.0 --depth-loss-weight 1.0 --normal-loss-weight 0.2 --depth-normal-weight 0.05" ;;
    # R1p: the recipe WITHOUT depth-normal consistency. That term was measured on
    # 2026-09-02 to diverge on its own -- lego arm L2 ran it alone, at weight 0.05, on a
    # scene with no priors at all, and lost 17.7 dB against a 0.108 dB repeat floor while
    # its own logged value climbed monotonically. Until --depth-source plane-aux lands, a
    # run with --depth-normal-weight > 0 in `center` mode is a known-broken configuration,
    # so this arm gives the first clean read on the depth and normal PRIORS in isolation.
    R1p)    echo "--flatten-loss-weight 1.0 --depth-loss-weight 1.0 --normal-loss-weight 0.2" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}
arm_seed() { [[ "$1" == "$SEEDFLOOR_ARM" ]] && echo "$SEED2" || echo "$SEED"; }

run_arm() {
  local arm="$1" seed extra
  seed="$(arm_seed "$arm")"; extra="$(arm_flags "$arm")"
  if [[ -f "$OUT/$arm.json" ]]; then echo "[$(date +%T)] $arm: report exists, skipping"; return; fi
  echo "[$(date +%T)] $arm: training (seed $seed) ${extra:-<baseline>}"
  ( cd "$MG" && caffeinate -i uv run --frozen python -m metal_gauss.train \
      "${common[@]}" --seed "$seed" \
      --report "$OUT/$arm.json" --export "$OUT/$arm.ply" --eval-dump "$OUT/$arm.dump" \
      $extra ) > "$OUT/$arm.log" 2>&1
  echo "[$(date +%T)] $arm: trained"
}

score_arm() {
  local arm="$1"
  echo "[$(date +%T)] $arm: scoring"
  if [[ -n "$SEED_CLOUD" && ! -f "$OUT/$arm.stats.json" ]]; then
    ( cd "$SPLATSTATS" && caffeinate -i uv run --frozen python scripts/splat_stats.py \
        "$OUT/$arm.ply" --seed "$SEED_CLOUD" --json "$OUT/$arm.stats.json" --quiet ) \
        >> "$OUT/$arm.log" 2>&1
  elif [[ -z "$SEED_CLOUD" ]]; then
    # No reference cloud for this scene: on-seed and thin-axis are UNDEFINED, not zero.
    echo "[$(date +%T)] $arm: no --seed-cloud, skipping splatstats (geometry metrics undefined)"
  fi
  [[ -f "$OUT/$arm.dump/lpips.json" ]] || \
    ( cd "$MG" && caffeinate -i uv run scripts/lpips_eval.py "$OUT/$arm.dump" ) \
      >> "$OUT/$arm.log" 2>&1
  # Merge LPIPS into the report so it is self-contained. Without this the Stage 4 gate is
  # only half-checked: the number exists on disk but metrics.lpips reads absent.
  ( cd "$MG" && uv run --frozen python scripts/backfill_lpips.py "$OUT" "$arm" ) \
      >> "$OUT/$arm.log" 2>&1
  echo "[$(date +%T)] $arm: scored"
}

echo "=== PHASE 1: floor arms (${FLOOR_LIST[*]}) ==="
for a in "${FLOOR_LIST[@]}"; do run_arm "$a"; done

echo "=== PHASE 2: score floors, then WRITE floors.json (must predate any treatment) ==="
for a in "${FLOOR_LIST[@]}"; do score_arm "$a"; done
rm -f "$OUT/floors.json"          # never let a STALE floors.json satisfy the guard below
if ! ( cd "$MG" && uv run --frozen python scripts/tier1_floors.py "$OUT" "${FLOOR_LIST[@]}" ) \
     >> "$OUT/floors.log" 2>&1; then
  echo "floors computation FAILED; refusing to grade arms (see $OUT/floors.log)" >&2
  tail -5 "$OUT/floors.log" >&2; exit 1
fi
tail -6 "$OUT/floors.log"
[[ -f "$OUT/floors.json" ]] || { echo "floors.json missing; refusing to grade arms" >&2; exit 1; }
touch "$OUT/FLOORS_DONE"

echo "=== PHASE 3: treatment arms ==="
for a in "${ARM_LIST[@]}"; do
  [[ " ${FLOOR_LIST[*]} " == *" $a "* ]] || run_arm "$a"
done

echo "=== PHASE 4: score treatments ==="
for a in "${ARM_LIST[@]}"; do
  [[ " ${FLOOR_LIST[*]} " == *" $a "* ]] || score_arm "$a"
done
touch "$OUT/ALL_DONE"
echo "[$(date +%T)] protocol complete: $OUT"
