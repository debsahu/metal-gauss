#!/bin/bash
# Snapshot-and-run wrapper for scripts/run_tier1_arms.sh. ALWAYS launch a protocol run
# through this, never by executing the runner directly.
#
#   scripts/launch_tier1.sh --out DIR [every other run_tier1_arms.sh argument]
#
# WHY THIS EXISTS. bash reads a script lazily by byte offset and re-reads after each
# command, so editing a running script shifts those offsets and makes it execute garbage.
# On 2026-09-02 the ARKitScenes driver had been running for 36 minutes when a one-line arm
# was added to the runner; 26 minutes later it died with
#
#     ./scripts/run_tier1_arms.sh: line 176: syntax error near unexpected token ')'
#
# on a file that `bash -n` passes cleanly. Five 30k-step arms had already trained; only
# their scoring was lost. "Do not edit the runner while it is running" was written in the
# runner's own header at the time and was still not enough, because the edit and the run
# were requested hours apart by different people.
#
# So the rule is now structural rather than remembered: each invocation copies the runner
# to an immutable, timestamped path inside the output directory and executes THAT. Edits to
# the working tree cannot reach a live run, and every output directory carries a
# byte-exact record of the script that produced it -- provenance the reports did not have.
set -euo pipefail
MG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$MG/scripts/run_tier1_arms.sh"

OUT=""
prev=""
for a in "$@"; do
  [[ "$prev" == "--out" ]] && OUT="$a"
  prev="$a"
done
[[ -n "$OUT" ]] || { echo "launch_tier1.sh: --out is required" >&2; exit 2; }
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAP="$OUT/runner.snapshot.$STAMP.sh"
cp "$RUNNER" "$SNAP"
chmod 500 "$SNAP"          # read+execute only: the snapshot is evidence, not a scratch file
{
  echo "$STAMP  md5=$(md5 -q "$SNAP")  git=$(git -C "$MG" rev-parse --short HEAD 2>/dev/null)" \
       "dirty=$(test -n "$(git -C "$MG" status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "         args: $*"
} >> "$OUT/runner.provenance.log"

echo "[launch] executing snapshot $SNAP"
exec bash "$SNAP" "$@"
