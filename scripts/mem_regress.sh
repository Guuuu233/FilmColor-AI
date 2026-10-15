#!/bin/bash
# Memory + pixel regression cases for the shared inversion / render stages (S3-2).
# usage: scripts/mem_regress.sh OUT_DIR   (run from the repo root; cases run one at a time)
# Writes OUT_DIR/<case>/<stem>.{tif,json} and OUT_DIR/memory.tsv (case, seconds, max RSS, peak footprint in bytes).
set -euo pipefail
OUT=$1
PY=.venv/bin/python
BLUENEG=data/raw/public/blueneg/negative-16bit/blue-intact/199807A-06-hk.dng
NEF=data/raw/own/ultramax400_r01/DSC_3826.NEF
RAF=data/raw/public/nlp_official/_DSF9235.RAF
mkdir -p "$OUT"
printf "case\tseconds\tmax_rss_bytes\tpeak_footprint_bytes\n" > "$OUT/memory.tsv"
run() {
  local name=$1; shift
  local log="$OUT/$name.time"
  /usr/bin/time -l "$PY" -I scripts/convert.py "$@" > "$OUT/$name.log" 2> "$log"
  local real rss fp
  real=$(awk '/ real /{print $1}' "$log")
  rss=$(awk '/maximum resident set size/{print $1}' "$log")
  fp=$(awk '/peak memory footprint/{print $1}' "$log")
  printf "%s\t%s\t%s\t%s\n" "$name" "$real" "$rss" "$fp" >> "$OUT/memory.tsv"
}
run blueneg_auto --auto "$OUT/blueneg_auto" "$BLUENEG"
run nef_auto --auto "$OUT/nef_auto" "$NEF"
run nef_default "$OUT/nef_default" "$NEF"
run raf_all_branches --render experiments/s3_memory/all_branches_render.json "$OUT/raf_all_branches" "$RAF"
cat "$OUT/memory.tsv"
