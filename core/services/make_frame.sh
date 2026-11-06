#!/bin/zsh
# Map pack for one frame: Blender passes → numpy maps. The scene is already exported (bench/scenes/X: scene.obj, cameras.json, world.json, figkit.json).
# Usage: core/services/make_frame.sh <scene folder> <scene output folder> <camera 1..> [pack.py flags]
# A failure of any step gives a non-zero exit code (a crash of Blender or pack.py must not yield 0).
set -e; setopt pipefail
HERE="${0:A:h:h:h}"; cd "$HERE"
SD="$1"; OUT="$2"; K="$3"; shift 3
abs() { [[ "$1" = /* ]] && echo "$1" || echo "$HERE/$1"; }
SDA=$(abs "$SD"); OA=$(abs "$OUT")
mkdir -p "$OA/cam$K"; LOG="$OA/cam$K/blender.log"
if ! blender -b --python-exit-code 1 -P core/blender/run.py -- core/outputs/bl_pack.py -- "$SDA" $K "$OA/cam$K/raw" > "$LOG" 2>&1; then
  { grep -E "Error|Traceback|line [0-9]+" "$LOG" | tail -8 >&2 || true; }; echo "make_frame: Blender crashed, log $LOG" >&2; exit 1
fi
grep -E "Error|Traceback" "$LOG" | head -5 || true
.venv/bin/python -m core.outputs.pack "$SDA" $K "$OA/cam$K/raw" "$OA/cam$K" "$@" | cut -c1-200
