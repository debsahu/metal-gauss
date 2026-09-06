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
# SPLATSTATS MUST RESOLVE BEFORE ANY ARM TRAINS, when a reference cloud was given.
# Its default is "$MG/../../analyze/splatstats", which assumes metal-gauss is checked out
# at <repo>/compute/metal-gauss. On a STANDALONE CLONE that path does not exist, the
# `|| echo ""` above leaves SPLATSTATS empty, `cd ""` silently stays in $MG, and the
# scorer runs "$MG/scripts/splat_stats.py" -- which is not there. On 2026-09-06 that took
# a batch down under `set -e` THIRTY-SEVEN MINUTES in, with all three floor arms trained
# and not one of them scored. Nothing failed at launch. Pass --splatstats on a standalone
# clone; this refuses at second zero rather than after the GPU time.
if [[ -n "$SEED_CLOUD" ]]; then
  if [[ -z "$SPLATSTATS" || ! -f "$SPLATSTATS/scripts/splat_stats.py" ]]; then
    echo "--seed-cloud was given but splatstats does not resolve:" >&2
    echo "    SPLATSTATS='${SPLATSTATS}'" >&2
    echo "    expected '${SPLATSTATS:-<empty>}/scripts/splat_stats.py' to exist" >&2
    echo "  Pass --splatstats /path/to/analyze/splatstats. The default assumes this" >&2
    echo "  checkout sits at <repo>/compute/metal-gauss; a standalone clone does not." >&2
    exit 2
  fi
fi
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
    # ---- NEEDLE arms (2026-09-05). metal-gauss produces 3-10x more needle-shaped
    # splats than every other trainer on the same scenes: needle_frac = frac(smid/smax
    # < 0.1) is 16.6% here against Brush 0.55-4.5% and LFS 0.15% on playroom_0821.
    # Flatten is EXONERATED -- it collapses smin and leaves smid/smax at 0.839 -> 0.839
    # -- so these arms probe the remaining flag-reachable candidates. Every one is
    # FLAG-ONLY: no trainer code differs between them and the B0 floors, which is what
    # makes them comparable at all (contrast the Tier 1 protocol deviation, where 18
    # arms spanned 7 commits).
    #
    # N1/N2 are the sub-pixel-dilation hypothesis: a splat thinner than a pixel is
    # widened by the 2D screen-space dilation, so the trainer never pays for a needle it
    # cannot see. PRE-REGISTERED PREDICTION: both buy <= 2-3 pp, because only 14% of
    # measured needles are below 1 px at the nearest training camera. --filter-3d is
    # view-INDEPENDENT (it widens in world space and bakes into the export);
    # --antialias is view-dependent and compensates opacity instead.
    N1)     echo "--filter-3d" ;;
    N2)     echo "--antialias" ;;
    # N3: the MCMC scale regulariser is a mean over exp(log_scales) across ALL THREE
    # axes, so it is dominated by smax and pushes the largest axis down hardest --
    # which is a pressure on the aspect ratio nobody has measured. Direction UNKNOWN
    # and deliberately not predicted.
    N3)     echo "--scale-reg 0.0" ;;
    # N4: the known-positive control. --num-downscales 2 costs +4.1 pp on this scene
    # (n=3) and +3.9 pp on ARKitScenes, so 0 must IMPROVE the needle fraction or the
    # whole battery is mis-wired. It is not a candidate fix -- at 0 metal-gauss is
    # still 3-4x Brush -- it is the arm that proves the instrument responds.
    # N2b: --antialias again, on a binary where it does not emit NaN gradients. N2 ran
    # before that was fixed and exported 31,158 non-finite scale_* values over 2.08% of
    # its splats, which `needle_frac` counted as HEALTHY splats -- so N2's shape columns
    # are bounded, not measured, and the arm has to be repeated rather than reinterpreted.
    # Same flag, different NAME, so N2's artifacts are not overwritten and the two remain
    # comparable side by side.
    N2b)    echo "--antialias" ;;
    N4)     echo "--num-downscales 0" ;;
    # ---- ISOTROPY arms. The in-plane isotropy barrier
    # (--inplane-isotropy-weight, geometry_loss.inplane_isotropy_loss) is a hinge on
    # log(smax/smid): mean(relu(log(smax/smid) - log r0)), r0 = 2 by default, so discs
    # pay nothing. It is the first term in this trainer's objective that mentions the
    # in-plane aspect ratio at all.
    #
    # WHY A DECADE SWEEP AND NOT ONE WEIGHT. The term is dimensionless and its gradient
    # is exactly +-1/N per paying splat in LOG space, whereas flatten's is s_min/N -- a
    # factor of ~1e-3 apart on millimetre splats. So the weight that means "as strong as
    # flatten at 1.0" is not knowable from flatten's scale, and a single guessed weight
    # that came out inert or catastrophic would say nothing about the term. Three decades
    # bracket it.
    I0)     echo "--inplane-isotropy-weight 0.01" ;;
    I1)     echo "--inplane-isotropy-weight 0.1" ;;
    I2)     echo "--inplane-isotropy-weight 1.0" ;;
    I3)     echo "--inplane-isotropy-weight 10.0" ;;
    # Im3: below the sweep. I0 at 0.01 already took needle_frac 16.80% -> 0.078%, so the
    # sweep found the effect but not its MINIMUM DOSE, and the dose matters: at 0.01 smax
    # also fell 25.7 -> 18.1 mm, a 30% change to the whole model that nothing asked for.
    Im3)    echo "--inplane-isotropy-weight 0.001" ;;
    # IR / IRb: THE ORTHOGONALITY CHECK, end to end. The barrier's whole design rests on
    # flatten owning the smin lane and the barrier owning smid/smax, and that is proven so
    # far only at the tensor (the sorted smin lane takes exactly zero gradient). These two
    # arms are the same recipe with and without the barrier, so flatten's own measured
    # effect -- the collapse of smin -- must survive unchanged in IRb, or the tensor-level
    # proof does not transfer to a 30k run.
    IRb)    echo "--flatten-loss-weight 1.0 --depth-loss-weight 1.0 --normal-loss-weight 0.2 --inplane-isotropy-weight 0.01" ;;
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
